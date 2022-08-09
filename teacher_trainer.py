import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class Real_Dataset(Dataset):
    def __init__(self, x_1, x_2) :
        super(Dataset).__init__()

        self.x_1 = x_1
        self.x_2 = x_2

    def __len__(self) :
        return len(self.x_1) + len(self.x_2)

    def __getitem__(self, idx) : # idx = 0~199 / 200~399
        if idx < 400:
            return self.x_1[idx], 0
        else:
            return self.x_2[idx - 400], 1

class Teacher_Model(nn.Module) :
    def __init__(self):
        super(Teacher_Model, self).__init__()

        self.layers1 = nn.Sequential(
            nn.Linear(2, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.layers2 = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32)
        )
        self.layers3 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self,x):
        x = self.layers1(x)
        identity = x
        x = self.layers2(x)
        x += identity
        x = self.layers3(x)
        return x

class Teacher_Trainer(object):
    def __init__(self):
        self.x_1 = None
        self.x_2 = None
        self.dataset = None
        self.model = None

        self.generate_real_dataset()

    def generate_real_dataset(self):
        x1 = np.concatenate([np.random.uniform(0, 4, 200), np.random.uniform(-4, 0, 200)])
        y1 = np.concatenate([np.random.uniform(-4, 0, 200), np.random.uniform(0, 4, 200)])

        x2 = np.concatenate([np.random.uniform(0, 4, 200), np.random.uniform(-4, 0, 200)])
        y2 = np.concatenate([np.random.uniform(0, 4, 200), np.random.uniform(-4, 0, 200)])

        self.x_1 = np.concatenate((x1[:, np.newaxis], y1[:, np.newaxis]), axis = 1)
        self.x_2 = np.concatenate((x2[:, np.newaxis], y2[:, np.newaxis]), axis = 1)

        plt.scatter(x1,y1,c="orange")
        plt.scatter(x2,y2,c="blue")
        plt.savefig('./real_data.png')
        plt.cla()

        self.x_1 = torch.Tensor(self.x_1)
        self.x_2 = torch.Tensor(self.x_2)
        self.dataset = Real_Dataset(self.x_1, self.x_2)

    def train(self):
        train_dataloader = torch.utils.data.DataLoader(self.dataset,
                                                        batch_size=32,
                                                        shuffle=True,
                                                        num_workers=4)

        self.model = Teacher_Model().cuda()

        learning_rate = 0.0001
        num_epoch = 15

        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        losses = []
        accuracy = []

        for epoch in range(num_epoch):
            self.model.train()
            print(f"-------- Epoch {epoch+1} --------")
            batch_losses = []

            for i, data in enumerate(train_dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()

                optimizer.zero_grad()
                output = self.model.forward(inputs)
                loss = criterion(output, labels)
                batch_losses.append(loss.item())

                loss.backward()
                optimizer.step()
                
            avg_loss = sum(batch_losses) / len(batch_losses)
            losses.append(avg_loss)
            print(f'Epoch {epoch+1} Loss : {avg_loss}')

            correct = 0
            total = 0

            self.model.eval()
            with torch.no_grad():
                for image,label in train_dataloader:
                    x = image.cuda()
                    y_= label.cuda()

                    output = self.model.forward(x)
                    _,output_index = torch.max(output,1)

                    total += label.size(0)
                    correct += (output_index == y_).sum().float()

                accuracy.append(100*correct/total)
                print(f'Epoch {epoch+1} Accuracy : {100*correct/total}')

            print()