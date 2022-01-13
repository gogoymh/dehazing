import os
import torch
import numpy as np
from torchvision import transforms
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable

import functions as fn
import config as cfg

exp = 6

######################################################################################################################
L1 = nn.L1Loss()
tot_loss = torch.zeros((361))
cnt = 0

for i in range(1,20):
    for j in range(1,20):
        print("[Step: (%d,%d)]" % (i,j))
        
        tensor_hazed = fn.hazed_jpg2tensor(cfg.path_hazed,
                                           transform=transforms.Compose([
                                                   fn.Rescale(800),
                                                   fn.CenterCrop(700),
                                                   fn.ToTensor()
                                                   ]))
        torch.manual_seed(1)
        np.random.seed(1)
        Hazed_loader = Data.DataLoader(tensor_hazed, batch_size=cfg.batch_size, shuffle=False, pin_memory=True)
        
        tensor_dehazed = fn.hazed_jpg2tensor(cfg.path_hazed,
                                             transform=transforms.Compose([
                                                     fn.Rescale(800),
                                                     fn.CenterCrop(700),
                                                     fn.normalize(0.05*i, 0.05*j),
                                                     fn.ToTensor()
                                                     ]))
        
        torch.manual_seed(1)
        np.random.seed(1)
        Dehazed_loader = Data.DataLoader(tensor_dehazed, batch_size=cfg.batch_size, shuffle=False, pin_memory=True)
    
        tensor_GT = fn.GT_jpg2tensor(cfg.path_GT,
                                     transform=transforms.Compose([
                                             fn.Rescale(800),
                                             fn.CenterCrop(700),
                                             fn.ToTensor()
                                             ]))
        torch.manual_seed(1)
        np.random.seed(1)
        GT_loader = Data.DataLoader(tensor_GT, batch_size=cfg.batch_size, shuffle=False, pin_memory=True)
    
        print("Loaded")
    
        Hazy = Hazed_loader.__iter__().next()
        Normalized = Dehazed_loader.__iter__().next()
        GT = GT_loader.__iter__().next()
        
        # ---- L1 loss ---- #
        Normalized = Variable(Normalized).float().cuda()
        GT = Variable(GT).float().cuda()
        loss = L1(Normalized, GT)
        tot_loss[cnt] = loss
        cnt += 1
        print("Loss is %f" % loss)
        
        
        # ---- save ---- #
        
        save_path = os.path.join(cfg.main_path, "exp %d" % exp)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
    
        fn.generate_batch_train_image(Hazy, Normalized, GT,  save_path, 0.05*i, 0.05*j)
        print("Saved")
        print("=" * 50)
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        