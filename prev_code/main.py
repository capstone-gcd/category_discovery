import os
import sys

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

import utils
from utils.config import parse_option
from utils.data import set_loader
from utils.losses import SupConLoss, UnSupConLoss
from model import set_model

def get_loader(args):
    train_loader, test_loader = set_loader(args)
    return train_loader, test_loader


def get_model(pretrain=True, **kwargs):
    model = set_model(**kwargs)
    if pretrain:
        pretrained_weight = torch.load("/root/default/kcc/category_discovery/backbone.pth")
        for name, param in model.named_parameters():
            if name in pretrained_weight:
                param = pretrained_weight[name]
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


def train(train_loader, model, sup_criterion, unsup_criterion, optimizer, epoch, args):
    model.train()
    
    loss_avg = 0
    for idx, (images, labels) in enumerate(train_loader):
        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        batch_size = labels.shape[0]
        
        # warm-up learning rate
        utils.warmup_learning_rate(args, epoch, idx, batch_size, optimizer)
        
        # compute loss
        features = model(images)
        print("shape of feature right away", features.shape)
        f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
        print("shape of f1, f2", f1.shape, f2.shape)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        print("feature shape after concat", features.shape)
        sup_loss = sup_criterion(features, labels)
        print("Sup loss", sup_loss.item())
        # unsup_loss = unsup_criterion(features)
        
        # loss = args.loss_lambda * sup_loss + (1-args.loss_lambda) * unsup_loss
        # loss_avg += loss.item()
        
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
        # if (idx + 1) % args.print_freq == 0:
        #     print('Train : [{0}][{1}/{2}]\t'
        #           'loss {loss:.3f}'.format(
        #               epoch, idx+1, len(train_loader), loss=loss.item()))
    return loss_avg/len(train_loader)

def test(test_loader, model, args):
    return None

def main():
    # configuration
    args = parse_option()
    device = torch.cuda.is_available()
    # load dataset
    train_loader, test_loader = get_loader(args)
    # load model
    model = get_model(args)
    sup_criterion = SupConLoss(args.temp)
    unsup_criterion = UnSupConLoss(args.temp)
    optimizer = torch.optim.AdamW(utils.get_params_groups(model))
    # send to cuda
    if torch.cuda.is_available():
        model = model.cuda()
        sup_criterion = sup_criterion.cuda()
        unsup_criterion = unsup_criterion.cuda()
    
    for epoch in range(1, args.epochs + 1):
        utils.adjust_learning_rate(args, optimizer, epoch)
        
        loss = train(train_loader, model, sup_criterion, unsup_criterion, optimizer, epoch, args)
        # print('Epoch {}: Loss {:.4f}'.format(epoch, loss))
        if epoch == 10:
            break
    return None

if __name__ == "__main__":
    main()