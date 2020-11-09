import torch
import torch.nn.functional as F
import numpy as np
import pdb

def one_hot_embedding(labels, num_classes=10):
    y = torch.eye(num_classes)
    neg = labels < 0 # negative labels
    labels[neg] = 0  # placeholder label to class-0
    y = y[labels] # create one hot embedding
    y[neg, 0] = 0 # remove placeholder label
    return y

def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)

def edl_mse_loss(output, target, device='cuda'):
    evidence = F.relu(output)
    alpha = evidence + 1
    target = target.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    err = torch.sum(
        (target - (alpha / S)) ** 2, dim=1, keepdim=True)
    var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    return err + var

def edl_mae_loss(output, target, device='cuda'):
    evidence = F.relu(output)
    alpha = evidence + 1
    target = target.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    err = torch.sum(
        torch.abs(target - (alpha / S)), dim=1, keepdim=True)
    var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    return err + var

def edl_soft_mse_loss(output, target, device='cuda'):
    alpha = F.softmax(output)
    target = target.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    err = torch.sum(
        (target - (alpha / S)) ** 2, dim=1, keepdim=True)
    var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    return err + var