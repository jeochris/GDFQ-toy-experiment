"""
basic trainer
"""
import time

import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils as utils
import numpy as np
import torch

__all__ = ["Trainer"]


class Trainer(object):
	"""
	trainer for training network, use SGD
	"""
	
	def __init__(self, model_teacher, generator, lr_master_G,
	             test_loader, settings, logger, tensorboard_logger=None,
	             opt_type="SGD", optimizer_state=None, run_count=0):
		"""
		init trainer
		"""
		
		self.settings = settings
		
		self.model_teacher = utils.data_parallel(
			model_teacher, self.settings.nGPU, self.settings.GPU)

		self.generator = utils.data_parallel(
			generator, self.settings.nGPU, self.settings.GPU)

		self.test_loader = test_loader
		self.tensorboard_logger = tensorboard_logger
		self.criterion = nn.CrossEntropyLoss().cuda()
		self.bce_logits = nn.BCEWithLogitsLoss().cuda()
		self.MSE_loss = nn.MSELoss().cuda()
		self.lr_master_G = lr_master_G
		self.opt_type = opt_type

		self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.settings.lr_G,
											betas=(self.settings.b1, self.settings.b2))

		self.logger = logger
		self.run_count = run_count
		self.scalar_info = {}
		self.mean_list = []
		self.var_list = []
		self.teacher_running_mean = []
		self.teacher_running_var = []

		self.fix_G = False
	
	def update_lr(self, epoch):
		"""
		update learning rate of optimizers
		:param epoch: current training epoch
		"""
		lr_G = self.lr_master_G.get_lr(epoch)
		# update learning rate of model optimizer
		for param_group in self.optimizer_G.param_groups:
			param_group['lr'] = lr_G
	
	def backward_G(self, loss_G):
		"""
		backward propagation of generator
		"""
		self.optimizer_G.zero_grad()
		loss_G.backward()
		self.optimizer_G.step()

	def hook_fn_forward(self,module, input, output):
		input = input[0]
		mean = input.mean([0])
		# use biased var in train
		var = input.var([0], unbiased=False)

		# fake image batch마다 mean, var 값들 저장
		self.mean_list.append(mean)
		self.var_list.append(var)
		# teacher model의 원래 BNS값 저장 (fixed BNS)
		# eval mode 이기에 running_mean, running_var 써도 업데이트 없이 고정된 값 사용?
		self.teacher_running_mean.append(module.running_mean)
		self.teacher_running_var.append(module.running_var)
	
	def train(self, epoch):
		"""
		training
		"""
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		fp_acc = utils.AverageMeter()

		iters = 20
		self.update_lr(epoch)

		self.model_teacher.eval()
		self.generator.train()
		
		start_time = time.time()
		end_time = start_time
		
		# 맨처음에 한번 batchnorm에 hook 걸어놓기
		# 모델이 실행될때 BN layer마다 해당 hook 실행해서 mean, var 값 계산해서 저장
		if epoch==0:
			for m in self.model_teacher.modules():
				if isinstance(m, nn.BatchNorm1d):
					m.register_forward_hook(self.hook_fn_forward)
		
		for i in range(iters):
			start_time = time.time()
			data_time = start_time - end_time

			# random z 생성
			z = Variable(torch.randn(self.settings.batchSize, self.settings.latent_dim)).cuda()

			# Get labels ranging from 0 to n_classes for n rows
			labels = Variable(torch.randint(0, self.settings.nClasses, (self.settings.batchSize,))).cuda()
			# https://titania7777.tistory.com/3
			z = z.contiguous()
			labels = labels.contiguous()
			# generator 넣고 image 추출
			images = self.generator(z, labels)
		
			# 이전 BN mean, var 값 비우기
			self.mean_list.clear()
			self.var_list.clear()
			# teacher model에 만든 fake image 돌리면서 해당 batch의 BN mean var 값도 다시 채움
			output_teacher_batch = self.model_teacher(images)

			# One hot loss
			loss_one_hot = self.criterion(output_teacher_batch, labels)

			# BN statistic loss
			BNS_loss = torch.zeros(1).cuda()

			for num in range(len(self.mean_list)):
				# teacher_running_mean : teacher 모델의 고정된 ?
				BNS_loss += self.MSE_loss(self.mean_list[num], self.teacher_running_mean[num]) + self.MSE_loss(
					self.var_list[num], self.teacher_running_var[num])

			BNS_loss = BNS_loss / len(self.mean_list)

			# loss of Generator
			loss_G = loss_one_hot + 0.1 * BNS_loss

			# Generator Backprop
			self.backward_G(loss_G)
			
			end_time = time.time()
			
			gt = labels.data.cpu().numpy()
			d_acc = np.mean(np.argmax(output_teacher_batch.data.cpu().numpy(), axis=1) == gt)

			fp_acc.update(d_acc)

		print(
			"[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%] [G loss: %f] [One-hot loss: %f] [BNS_loss:%f]"
			% (epoch + 1, self.settings.nEpochs, i+1, iters, 100 * fp_acc.avg, loss_G.item(), loss_one_hot.item(), BNS_loss.item())
		)

		self.scalar_info['accuracy every epoch'] = 100 * d_acc
		self.scalar_info['G loss every epoch'] = loss_G
		self.scalar_info['One-hot loss every epoch'] = loss_one_hot
		
		if self.tensorboard_logger is not None:
			for tag, value in list(self.scalar_info.items()):
				self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
			self.scalar_info = {}