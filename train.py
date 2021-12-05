import os

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from fire import Fire

import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models
import torch.nn as nn
import numpy as np
import torchvision
import torch

from madgrad_optimizer import MADGRADOptimizer
from mda_optimizer import MDAOptimizer


class Trainer:
    def __init__(self, run_name, optim_name):
        self.run_name = run_name
        self.save_dir = os.path.join("runs", self.run_name)

        self.train_transform = transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.test_transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.batch_size = 128
        self.num_epochs = 300

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=self.train_transform
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
        )

        self.test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=False, transform=self.test_transform
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        self.model = torchvision.models.resnet152(pretrained=False, progress=False, num_classes=10)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.maxpool = nn.Identity()
        self.model = self.model.to(self.device)

        self.criterion = nn.NLLLoss()

        if optim_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        elif optim_name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=2.5e-4, weight_decay=0.0001)
        elif optim_name == "mda":
            self.optimizer = MDAOptimizer(self.model.parameters(), lr=2.5e-4, momentum=0.9, weight_decay=0.0001)
        elif optim_name == "madgrad":
            self.optimizer = MADGRADOptimizer(self.model.parameters(), lr=2.5e-4, momentum=0.9, weight_decay=0.0001)
        else:
            raise RuntimeError(f"Optimizer {optim_name} not recognized")

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[150, 225], gamma=0.1)

        self.best_test_acc = -1
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

                logits = F.log_softmax(self.model(inputs), dim=1)
                preds = torch.argmax(logits, axis=-1)

                loss = self.criterion(logits, labels)
                epoch_loss.append(loss.item())

                loss.backward()
                self.optimizer.step()

                acc = (preds == labels).float().mean()
                tq.set_postfix({"loss": loss.item(), "accuracy": acc.item()})
                step = epoch*len(self.train_loader) + i
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/acc", acc.item(), step)

            torch.save({
                "state_dict": self.model.state_dict(),
                "optim_state_dict": self.optimizer.state_dict(),
            }, os.path.join(self.save_dir, "last_model.pth"))

            self.eval(epoch)

            if self.scheduler:
                self.scheduler.step()

    def eval(self, epoch):
        with torch.no_grad():
            self.model.eval()
            tq = tqdm(self.test_loader, desc=f"Testing  [{epoch+1}/{self.num_epochs}]")

            loss_arr = []
            acc_arr = []

            for data in tq:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                logits = F.log_softmax(self.model(inputs), dim=1)
                preds = torch.argmax(logits, axis=-1)

                acc_arr += (preds == labels).detach().cpu().numpy().astype(int).tolist()

                loss = self.criterion(logits, labels)
                loss_arr.append(loss.item())
                tq.set_postfix({"test_loss": np.mean(loss_arr), "test_acc": np.mean(acc_arr)})

            self.writer.add_scalar("test/loss", np.mean(loss_arr), epoch)
            self.writer.add_scalar("test/acc", np.mean(acc_arr), epoch)

        if np.mean(acc_arr) > self.best_test_acc:
            torch.save({
                "state_dict": self.model.state_dict(),
                "optim_state_dict": self.optimizer.state_dict(),
            }, os.path.join(self.save_dir, "best_model.pth"))
            self.best_test_acc = np.mean(acc_arr)
        return np.mean(acc_arr)


def main(name, optim):
    x = Trainer(run_name=name, optim_name=optim.lower())
    x.train()


if __name__ == "__main__":
    Fire(main)
