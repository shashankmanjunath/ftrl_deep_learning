import os

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from fire import Fire

import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models
import torch.nn as nn
import numpy as np
import torchvision
import torch


class Trainer:
    def __init__(self, run_name):
        self.run_name = run_name
        self.save_dir = os.path.join("runs", self.run_name)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.batch_size = 256
        self.num_epochs = 300

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=False, transform=self.transform
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
        )

        self.test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=False, transform=self.transform
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        self.model = torchvision.models.resnet152(pretrained=False, progress=False)
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=10, bias=True)
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max")

        self.writer = SummaryWriter(self.save_dir)

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            tq = tqdm(self.train_loader, desc=f"Training [{epoch+1}/{self.num_epochs}]")
            epoch_loss = []

            for i, data in enumerate(tq):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                preds = torch.argmax(outputs, axis=-1)

                loss = self.criterion(outputs, labels)
                epoch_loss.append(loss.item())

                loss.backward()
                self.optimizer.step()

                acc = (preds == labels).float().mean()
                tq.set_postfix({"loss": loss.item(), "accuracy": acc.item()})
                self.writer.add_scalar("train/loss", loss.item(), epoch*len(self.train_loader) + i)
                self.writer.add_scalar("train/acc", acc.item(), epoch*len(self.train_loader) + i)

            eval_acc = self.eval(epoch)
            self.scheduler.step(eval_acc)

    def eval(self, epoch):
        with torch.no_grad():
            self.model.eval()
            tq = tqdm(self.test_loader, desc=f"Testing [{epoch+1}/{self.num_epochs}]")

            loss_arr = []
            acc_arr = []

            for data in tq:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                preds = torch.argmax(outputs, axis=-1)

                acc_arr += (preds == labels).detach().cpu().numpy().astype(int).tolist()

                loss = self.criterion(outputs, labels)
                loss_arr.append(loss.item())
                tq.set_postfix({"test_loss": np.mean(loss_arr), "test_acc": np.mean(acc_arr)})

            self.writer.add_scalar("train/loss", np.mean(loss_arr), epoch)
            self.writer.add_scalar("train/acc", np.mean(acc_arr), epoch)
        return np.mean(acc_arr)


def main(name):
    x = Trainer(run_name=name)
    x.train()


if __name__ == "__main__":
    Fire(main)
