import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy


def train_net(model, loader, criterion, optimizer, device):
    running_loss = 0.0
    running_corrects = 0
    num_sample = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        num_sample += inputs.size(0)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        _, preds = torch.max(logits, 1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    loss = running_loss / num_sample
    acc = running_corrects.double() / num_sample
    return loss, acc


def val_net(model, loader, criterion, device):
    with torch.no_grad():
        running_loss = 0.0
        running_corrects = 0
        num_sample = 0
        for inputs, labels in loader:
            inputs = inputs.to(device)
            num_sample += inputs.size(0)
            labels = labels.to(device)
            logits = model(inputs)
            _, preds = torch.max(logits, 1)
            loss = criterion(logits, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        loss = running_loss / num_sample
        acc = running_corrects.double() / num_sample
        return loss, acc


class Net1(nn.Module):
    def __init__(self, num_classes=6):
        super(Net1, self).__init__()
        ndf = 32
        self.model = nn.Sequential(
            nn.Conv2d(1, ndf, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([ndf, 14, 14]),
            nn.Conv2d(ndf, ndf * 2, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([ndf * 2, 7, 7]),
            nn.Conv2d(ndf * 2, ndf * 4, 7, 1, padding=0),
        )
        self.fc = nn.Sequential(
            nn.Linear(ndf * 4, ndf * 2),
            nn.LeakyReLU(),
            nn.Linear(ndf * 2, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Net0(nn.Module):
    def __init__(self):
        super(Net0, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_ood_score(logit, prob, method='max_logit'):
    logit = logit.detach().cpu().numpy()
    prob = prob.detach().cpu().numpy()
    if method == 'max_logit':
        return -np.max(logit, axis=1).flatten()
    elif method == 'max_prob':
        return -np.max(prob, axis=1).flatten()
    elif method == 'shannon_entropy':
        known_entropy = entropy(prob, axis=1).flatten()
        return known_entropy
    elif method == 'energy':
        return -np.log(np.exp(logit).sum(axis=1)).flatten()
    elif method == 'GEN':
        M = 6
        gamma = 0.1
        sorted_prob = np.sort(prob, axis=-1)[:, ::-1]
        gen_entropy = np.sum(sorted_prob[:, :M] ** gamma * (1 - sorted_prob[:, :M]) ** gamma, axis=-1).flatten()
        return gen_entropy
    else:
        raise NotImplemented('Please check the method to compute ood score')


def eval_ood(model, known_loader, unknown_loader, device, method='max_logit'):
    model.eval()
    ##################################################################
    # for known
    ##################################################################
    correct, num_known, = 0, 0
    known_ood_scores = []
    with torch.no_grad():
        for data, labels in known_loader:
            data = data.to(device)
            labels = labels.to(device)
            logits = model(data)
            probs = F.softmax(logits, dim=1)
            pred = logits.max(1)[1]
            num_known += data.size(0)
            correct += (pred == labels).sum()
            known_ood_scores.append(get_ood_score(logits, probs, method))
    known_ood_scores = np.hstack(known_ood_scores).flatten()
    assert known_ood_scores.shape[0] == num_known
    ##################################################################
    # for unknown
    ##################################################################
    num_unknown = 0
    unknown_ood_scores = []
    with torch.no_grad():
        for data, labels in unknown_loader:
            data = data.to(device)
            logits = model(data)
            probs = F.softmax(logits, dim=1)
            num_unknown += data.size(0)
            unknown_ood_scores.append(get_ood_score(logits, probs, method))
    unknown_ood_scores = np.hstack(unknown_ood_scores)
    assert unknown_ood_scores.shape[0] == num_unknown
    ##################################################################
    # compute output: acc and auroc
    ##################################################################
    acc = float(correct) / float(num_known)
    labels = np.hstack((np.zeros_like(known_ood_scores), np.ones_like(unknown_ood_scores)))
    ood_scores = np.hstack((known_ood_scores, unknown_ood_scores))
    auroc = roc_auc_score(labels, ood_scores)
    return acc, auroc
