import torch
import torch.utils.data as Data
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable

from network2 import Hazenet as net
import functions as fn

import sys
sys.path.append("C:/Dehazing/pyclass23/")
import JpgTensoring23 as jpt


Dehazer = net().cuda()
optimizer = torch.optim.Adam(Dehazer.parameters(), lr=0.0002, betas=(0.5, 0.999))
L1 = nn.L1Loss()

checkpoint_D = torch.load("C:/유민형/개인 연구/dehazing/exp 10/exp10.pkl")
            
Dehazer.load_state_dict(checkpoint_D['model_state_dict'])
optimizer.load_state_dict(checkpoint_D['optimizer_state_dict'])
epoch = checkpoint_D['epoch']
loss = checkpoint_D['loss']




tensor_hazed_test = jpt.hazed_jpg2tensor("C:/유민형/개인 연구/dehazing/hazy_test/",
                                    transform=transforms.Compose([
                                            jpt.Rescale(150),
                                            jpt.CenterCrop(128),
                                            jpt.ToTensor()
                                            ]))
Hazy_test = Data.DataLoader(tensor_hazed_test, batch_size=32, shuffle=False,
                                   pin_memory=True)


tensor_GT_test = jpt.clear_jpg2tensor("C:/유민형/개인 연구/dehazing/GT_test/",
                                      transform=transforms.Compose([
                                              jpt.Rescale(150),
                                              jpt.CenterCrop(128),
                                              jpt.ToTensor()
                                              ]))

GT_test = Data.DataLoader(tensor_GT_test, batch_size=32, shuffle=False,
                          pin_memory=True)

Hazy_test = Hazy_test.__iter__().next()
GT_test = GT_test.__iter__().next()

Hazy_test = Variable(Hazy_test).float().cuda()
GT_test = Variable(GT_test).float().cuda()

Dehazer.eval()
sudo_test = Dehazer(Hazy_test)

print("L1: %f" % L1(sudo_test, GT_test))

fn.generate_batch_train_image(Hazy_test, sudo_test, GT_test, "C:/유민형/개인 연구/dehazing/exp 10/", 5010)