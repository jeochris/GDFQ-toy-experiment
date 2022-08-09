import argparse
import datetime
import logging
import os
import time
import traceback
import sys
import copy
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from options import Option

from trainer import Trainer
from teacher_trainer import Teacher_Trainer

import utils as utils

class Generator(nn.Module):
	def __init__(self, options=None):
		super(Generator, self).__init__()
		self.settings = options
		
		self.label_emb = nn.Embedding(2, self.settings.latent_dim)
		self.layers = nn.Sequential(
			nn.Linear(self.settings.latent_dim, 32),
			nn.BatchNorm1d(32),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(32, 64),
			nn.BatchNorm1d(64),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(64, 32),
			nn.BatchNorm1d(32),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(32, 2)
		)

	def forward(self, z, labels):
		# input for generator = z & label's embedding -> element-wise multiplied
		gen_input = torch.mul(self.label_emb(labels), z)
		out = self.layers(gen_input)
		return out

class ExperimentDesign:
	def __init__(self, generator=None, options=None, base_model=None, real_dataset=None):
		self.settings = options
		
		# generator 가져옴
		self.generator = generator
		self.real_dataset = real_dataset
		self.base_model = base_model

		self.test_loader = None
		self.model_teacher = None
		
		self.optimizer_state = None
		self.trainer = None
		self.start_epoch = 0

		self.logger = self.set_logger()

		# prepare for experiment
		self.prepare()

	def set_logger(self):
		logger = logging.getLogger('baseline')
		file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
		console_formatter = logging.Formatter('%(message)s')
		# file log
		file_handler = logging.FileHandler(os.path.join(self.settings.save_path, "train_test.log"))
		file_handler.setFormatter(file_formatter)
		
		# console log
		console_handler = logging.StreamHandler(sys.stdout)
		console_handler.setFormatter(console_formatter)
		
		logger.addHandler(file_handler)
		logger.addHandler(console_handler)
		
		logger.setLevel(logging.INFO)
		return logger

	def prepare(self):
		self._set_gpu()
		self._set_dataloader() # dataloader 생성 - datalodaer.py
		self._set_model() # teacher, student model 초기상태 설정
		self._set_trainer()
	
	def _set_gpu(self):
		torch.manual_seed(self.settings.manualSeed)
		torch.cuda.manual_seed(self.settings.manualSeed)
		assert self.settings.GPU <= torch.cuda.device_count() - 1, "Invalid GPU ID"
		cudnn.benchmark = True

	def _set_dataloader(self):
		self.test_loader = torch.utils.data.DataLoader(self.real_dataset, batch_size=32, shuffle=False, num_workers=8)

	def _set_model(self):
		self.model_teacher = self.base_model
		self.model_teacher.eval()

	def _set_trainer(self):
		# set lr master
		lr_master_G = utils.LRPolicy(self.settings.lr_G,
									 self.settings.nEpochs,
									 self.settings.lrPolicy_G)

		params_dict_G = {
			'step': self.settings.step_G,
			'decay_rate': self.settings.decayRate_G
		}
		
		lr_master_G.set_params(params_dict=params_dict_G)

		# set trainer
		self.trainer = Trainer(
			model_teacher=self.model_teacher,
			generator = self.generator,
			test_loader=self.test_loader,
			lr_master_G=lr_master_G,
			settings=self.settings,
			logger=self.logger,
			opt_type=self.settings.opt_type,
			optimizer_state=self.optimizer_state,
			run_count=self.start_epoch)

	def run(self):
		best_top1 = 100
		start_time = time.time()

		try:
			for epoch in range(self.start_epoch, self.settings.nEpochs):
				self.epoch = epoch
				self.start_epoch = 0

				self.trainer.train(epoch=epoch)

		except BaseException as e:
			self.logger.error("Training is terminating due to exception: {}".format(str(e)))
			traceback.print_exc()
		
		end_time = time.time()
		time_interval = end_time - start_time
		t_string = "Running Time is: " + str(datetime.timedelta(seconds=time_interval)) + "\n"
		self.logger.info(t_string)

		return self.trainer.generator


def main():
	parser = argparse.ArgumentParser(description='Baseline')
	# conf_path : hocon file
	parser.add_argument('--conf_path', type=str, metavar='conf_path',
	                    help='input the path of config file')
	parser.add_argument('--id', type=int, metavar='experiment_id',
	                    help='Experiment ID')
	args = parser.parse_args()

	# option -> all hyperparameter
	option = Option(args.conf_path)
	option.manualSeed = args.id + 1
	option.experimentID = option.experimentID + "{:0>2d}_repeat".format(args.id)

	# Build Teacher Model
	teacher_trainer = Teacher_Trainer()
	teacher_trainer.train()
	model_teacher = teacher_trainer.model
	real_dataset = teacher_trainer.dataset

	generator = Generator(option)

    # experiment design init
	experiment = ExperimentDesign(generator, option, model_teacher, real_dataset)
	ret_gen = experiment.run()

	# generator test
	ret_gen.eval()

	with torch.no_grad():
		z = Variable(torch.randn(400, option.latent_dim)).cuda()
		labels = Variable(torch.randint(0, 2, (400,))).cuda()
		z = z.contiguous()
		labels = labels.contiguous()
		images = ret_gen(z, labels)

	labels = labels.cpu().detach().numpy()
	images = images.cpu().detach().numpy()

	tmpx1 = []
	tmpy1 = []
	tmpx2 = []
	tmpy2 = []

	for i in range(400):
		if labels[i]==0:
			tmpx1.append(images[i][0])
			tmpy1.append(images[i][1])
		else:
			tmpx2.append(images[i][0])
			tmpy2.append(images[i][1])

	plt.scatter(tmpx1,tmpy1,c="orange")
	plt.scatter(tmpx2,tmpy2,c="blue")
	plt.savefig('./generator.png')

if __name__ == '__main__':
	main()
