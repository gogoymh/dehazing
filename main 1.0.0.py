import os
import torch
import numpy as np
from torchvision import transforms
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable
import argparse

import functions as fn
import config as cfg
from network import Hazenet as net

######################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="Number of epoch")
parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="Adam parameter")
parser.add_argument("--b2", type=float, default=0.999, help="Adam parameter")
parser.add_argument("--exp", type=int, default=9, help="Number of experiment")

opt = parser.parse_args()
######################################################################################################################       
torch.manual_seed(1)
np.random.seed(1)
tensor_hazed = fn.hazed_jpg2tensor(cfg.path_hazed,
                                   transform=transforms.Compose([
                                           fn.Rescale(150),
                                           fn.CenterCrop(128),
                                           fn.ToTensor()
                                           ]))
torch.manual_seed(1)
np.random.seed(1)
Hazed_loader = Data.DataLoader(tensor_hazed, batch_size=cfg.batch_size, shuffle=False, pin_memory=True)

torch.manual_seed(1)
np.random.seed(1)
tensor_GT = fn.GT_jpg2tensor(cfg.path_GT,
                             transform=transforms.Compose([
                                     fn.Rescale(150),
                                     fn.CenterCrop(128),
                                     fn.ToTensor()
                                     ]))
torch.manual_seed(1)
np.random.seed(1)
GT_loader = Data.DataLoader(tensor_GT, batch_size=cfg.batch_size, shuffle=False, pin_memory=True)

######################################################################################################################
Dehazer = net().cuda()
optimizer = torch.optim.Adam(Dehazer.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
L1 = nn.L1Loss()

######################################################################################################################
tot_loss = torch.zeros((opt.n_epochs))

for epoch in range(opt.n_epochs):
    
    for batch, Hazy in enumerate(Hazed_loader):
        
        GT = GT_loader.__iter__().next()
    
        Hazy = Variable(Hazy).float().cuda()
        GT = Variable(GT).float().cuda()
            
        optimizer.zero_grad()
        sudo = Dehazer(Hazy)
        loss = L1(sudo, GT)    
        loss.backward()    
        optimizer.step()
    
        tot_loss[epoch] += loss
    
    tot_loss[epoch] /= len(Hazed_loader)
    print("[Epoch: %d][Loss: %f]" % (epoch+1, tot_loss[epoch]))
    
    if (epoch+1) % 10 == 0:
        save_path = os.path.join(cfg.main_path, "exp %d" % opt.exp)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
    
        fn.generate_batch_train_image(Hazy, sudo, GT, save_path, epoch)
        
        print("Saved!")
        
        
        
        
        
        
        
        