import torch
import torch.nn as nn
import torch.nn.functional as F


class Softmax(nn.Module):
    def __init__(self, args):
        super(Softmax, self).__init__()
        self.args = args

    def forward(self, logits, labels=None, reduction='mean'):
        if labels is None:
            return logits, 0, 0
        loss = F.cross_entropy(logits, labels, reduction=reduction)
        prob = torch.softmax(logits, dim=1)
        # prob for non-maximal probability class
        prob_nmpc = torch.sort(prob, dim=1)[0][:, :-1]
        std = torch.std(prob_nmpc, dim=1)
        if self.args.only_corr:
            pred = torch.argmax(prob, dim=1)
            std = std * (pred == labels)
        loss_nmpc = std.mean() if reduction == 'mean' else std
        return logits, loss, loss_nmpc
