import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models, transforms
import os

from model.util import train_net, val_net, Net1, Net0


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True, help="name of the experiment")
parser.add_argument('--data_root', type=str,
                    default="/home/oem/Mingle/OracleMNIST/data/oracle-mnist-size224",
                    help="the path to dataset with trian and val")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--device', type=str, default="0")
parser.add_argument('--input_size', type=int, default=28, help="size of input")
parser.add_argument('--if_val', type=int, default=1, help="if val when training")
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.5)
args = parser.parse_args()

# Define transforms for the training and validation sets
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(args.input_size),
    # transforms.RandomRotation(15),
    # transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
    ])

val_transform = transforms.Compose([
    transforms.Resize((args.input_size, args.input_size)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
    ])

# Load the datasets with ImageFolder
data_dir = args.data_root
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'test')

train_dset = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dset = torchvision.datasets.ImageFolder(root=val_dir, transform=val_transform)

train_loader = torch.utils.data.DataLoader(
    train_dset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    val_dset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True
)
class_names = train_dset.classes
print(class_names)

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

model = Net1(num_classes=10).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
# optimizer = optim.Adam(model.parameters(), lr=args.lr)
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[80])

epochs = 100
for epoch in range(epochs):
    # train
    train_loss, train_acc = train_net(model, train_loader, criterion, optimizer, device)

    if args.if_val:
        val_loss, val_acc = val_net(model, val_loader, criterion, device)
        print(f'Epoch {epoch} train Loss: {train_loss:.4f} Acc: {train_acc:.4f},'
              f' test Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
    else:
        print(f'Epoch {epoch} train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')

    # scheduler.step()
