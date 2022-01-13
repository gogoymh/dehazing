#import os
#import torch
import numpy as np
from torchvision import transforms
#import torch.utils.data as Data
#import torch.nn as nn
#from torch.autograd import Variable
import matplotlib.pyplot as plt

import functions as fn
import config as cfg


tensor_hazed = fn.hazed_jpg2tensor(cfg.path_hazed,
                                     transform=transforms.Compose([
                                             fn.Rescale(800),  
                                             fn.CenterCrop(700)
                                             ]))

tensor_dehazed = fn.hazed_jpg2tensor(cfg.path_hazed,
                                     transform=transforms.Compose([
                                             fn.Rescale(800),  
                                             fn.CenterCrop(700)
                                             ]))

tensor_GT = fn.GT_jpg2tensor(cfg.path_GT,
                             transform=transforms.Compose([
                                             fn.Rescale(800),
                                             fn.CenterCrop(700)
                                             ]))

idx = 1

a = tensor_hazed[idx]
b = tensor_dehazed[idx]
c = tensor_GT[idx]

plt.imshow(a)
plt.show()
plt.close()
print(np.sum(np.abs(c.copy()-a.copy())))
#############################################################################

'''
# padding
b_p = np.zeros((700+2,700+2,3), dtype='float')
b_p[1:701, 1:701, 0] = b[:,:,0]
b_p[1:701, 1:701, 1] = b[:,:,1]
b_p[1:701, 1:701, 2] = b[:,:,2]

b_hat = np.empty((700,700,3), dtype='float')
#
for channel in range(3):
    for i in range(702-3+1):
        for j in range(702-3+1):
            b_hat[i,j,channel] = (b[i,j,channel] - b_p[i:(3+i), j:(3+j), channel].mean())/b_p[i:(3+i), j:(3+j), channel].std()

# minmax
b_hat[:,:,0] = (b_hat[:,:,0] - b_hat[:,:,0].min())/(b_hat[:,:,0].max()-b_hat[:,:,0].min())
b_hat[:,:,1] = (b_hat[:,:,1] - b_hat[:,:,1].min())/(b_hat[:,:,1].max()-b_hat[:,:,1].min())
b_hat[:,:,2] = (b_hat[:,:,2] - b_hat[:,:,2].min())/(b_hat[:,:,2].max()-b_hat[:,:,2].min())

print("done")
'''





# Noramlize

b[:,:,0] = (b[:,:,0] - b[:,:,0].mean())/b[:,:,0].std()
b[:,:,1] = (b[:,:,1] - b[:,:,1].mean())/b[:,:,1].std()
b[:,:,2] = (b[:,:,2] - b[:,:,2].mean())/b[:,:,2].std()

# minmax
b[:,:,0] = (b[:,:,0] - b[:,:,0].min())/(b[:,:,0].max()-b[:,:,0].min())
b[:,:,1] = (b[:,:,1] - b[:,:,1].min())/(b[:,:,1].max()-b[:,:,1].min())
b[:,:,2] = (b[:,:,2] - b[:,:,2].min())/(b[:,:,2].max()-b[:,:,2].min())


plt.imshow(b)
plt.show()
plt.close()
print(np.sum(np.abs(c.copy()-b.copy())))

##############################################################################
b_2 = b.copy()

b_2[:,:,0] = (b_2[:,:,0] - b_2[:,:,0].mean())/b_2[:,:,0].std()
b_2[:,:,1] = (b_2[:,:,1] - b_2[:,:,1].mean())/b_2[:,:,1].std()
b_2[:,:,2] = (b_2[:,:,2] - b_2[:,:,2].mean())/b_2[:,:,2].std()

# minmax
b_2[:,:,0] = (b_2[:,:,0] - b_2[:,:,0].min())/(b_2[:,:,0].max()-b_2[:,:,0].min())
b_2[:,:,1] = (b_2[:,:,1] - b_2[:,:,1].min())/(b_2[:,:,1].max()-b_2[:,:,1].min())
b_2[:,:,2] = (b_2[:,:,2] - b_2[:,:,2].min())/(b_2[:,:,2].max()-b_2[:,:,2].min())


plt.imshow(b_2)
plt.show()
plt.close()
print(np.sum(np.abs(c.copy()-b_2.copy())))




#plt.imshow(b_hat)
#plt.show()
#plt.close()

'''
##############################################################################
d = b.copy() - a.copy()

# minmax
d[:,:,0] = (d[:,:,0] - d[:,:,0].min())/(d[:,:,0].max()-d[:,:,0].min())
d[:,:,1] = (d[:,:,1] - d[:,:,1].min())/(d[:,:,1].max()-d[:,:,1].min())
d[:,:,2] = (d[:,:,2] - d[:,:,2].min())/(d[:,:,2].max()-d[:,:,2].min())


plt.imshow(d)
plt.show()
plt.close()
print(np.sum(np.abs(c.copy()-d.copy())))

##############################################################################
e = d + d - a

# minmax
e[:,:,0] = (e[:,:,0] - e[:,:,0].min())/(e[:,:,0].max()-e[:,:,0].min())
e[:,:,1] = (e[:,:,1] - e[:,:,1].min())/(e[:,:,1].max()-e[:,:,1].min())
e[:,:,2] = (e[:,:,2] - e[:,:,2].min())/(e[:,:,2].max()-e[:,:,2].min())

plt.imshow(e)
plt.show()
plt.close()
'''
###############################################################################


plt.imshow(c)
plt.show()
plt.close()

print(np.sum(np.abs(c.copy()-c.copy())))
'''
# minmax
t[:,:,0] = (t[:,:,0] - t[:,:,0].min())/(t[:,:,0].max()-t[:,:,0].min())
t[:,:,1] = (t[:,:,1] - t[:,:,1].min())/(t[:,:,1].max()-t[:,:,1].min())
t[:,:,2] = (t[:,:,2] - t[:,:,2].min())/(t[:,:,2].max()-t[:,:,2].min())

plt.imshow(t)
plt.show()
plt.close()





'''





