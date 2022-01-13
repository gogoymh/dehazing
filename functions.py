import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from skimage import io, transform
import math

import config as cfg

class ToTensor(object):
  def __call__(self, image):
    image = image.transpose((2,0,1))
    return torch.from_numpy(image).double()

class CenterCrop(object):
  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size)
    else:
      assert len(output_size) == 2
      self.output_size = output_size
      
  def __call__(self, image):
    h = image.shape[0:2][0]
    w = image.shape[0:2][1]
    new_h, new_w = self.output_size
    
    top = math.floor(h/2)  - math.floor(new_h/2)
    left = math.floor(w/2) - math.floor(new_w/2)
    
    assert top > 0 and left > 0
    
    bottom = top + new_h
    right = left + new_w
    
    crop = image[top:bottom,left:right, :]
    
    return crop

class Rescale(object):
  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size
      
  def __call__(self, image):
    h, w = image.shape[0:2]
    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = math.floor(self.output_size * h / w), self.output_size
      else:
        new_h, new_w = self.output_size, math.floor(self.output_size * w / h)
    else:
      new_h, new_w = self.output_size
      
    new_h, new_w = int(new_h), int(new_w)
    
    image = transform.resize(image, (new_h, new_w))
    
    return image

class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h = image.shape[0:2][0]
        w = image.shape[0:2][1]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return image

class hazed_jpg2tensor(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
    
    def __len__(self):
        num = 30 #폴더에 있는 사진 갯수
        return num
    
    def __getitem__(self, idx):
        file = ("%02d" % (idx+1)) + '_indoor_hazy.jpg'
        img_name = os.path.join(self.path, file)
        
        image = io.imread(img_name)/255
        
        if self.transform:
            image = self.transform(image)
            
        return image

class GT_jpg2tensor(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
    
    def __len__(self):
        num = 30 #폴더에 있는 사진 갯수
        return num
    
    def __getitem__(self, idx):
        file = ("%02d" % (idx+1)) + '_indoor_GT.jpg'
        img_name = os.path.join(self.path, file)
        
        image = io.imread(img_name)/255
        
        if self.transform:
            image = self.transform(image)
            
        return image

class normalize(object):
    # img_type: numpy
    #img = img * 1.0 / 255
    def __init__(self, mean, sigma):
        assert mean >= 0 and mean <= 1
        assert sigma > 0
        
        self.mean = mean
        self.sigma = sigma
    def __call__(self, image):
        return (image - self.mean) / self.sigma

def generate_img_batch(syn_batch, ref_batch, real_batch, png_path):
    # syn_batch_type: Tensor, ref_batch_type: Tensor
    def tensor_to_numpy(img):
        img = img.numpy()
        img += max(-img.min(), 0)
        if img.max() != 0:
            img /= img.max()
        img *= 255
        img = img.astype(np.uint8)
        img = np.transpose(img, [1, 2, 0])
        return img # fine 

    syn_batch = syn_batch[:15]
    ref_batch = ref_batch[:15]
    real_batch = real_batch[:15]

    a_blank = torch.zeros(cfg.img_height, cfg.img_width, 3).numpy().astype(np.uint8)

    nb = syn_batch.size(0)
    #print(syn_batch.size())
    #print(ref_batch.size())
    vertical_list = []

    for index in range(0, nb, cfg.pics_line):
        st = index
        end = st + cfg.pics_line

        if end > nb:
            end = nb

        syn_line = syn_batch[st:end]
        ref_line = ref_batch[st:end]
        real_line = real_batch[st:end]
        #print('====>', nb)
        #print(syn_line.size())
        #print(ref_line.size())
        nb_per_line = syn_line.size(0)

        line_list = []

        for i in range(nb_per_line):
            #print(i, len(syn_line))
            syn_np = tensor_to_numpy(syn_line[i])
            ref_np = tensor_to_numpy(ref_line[i])
            real_np = tensor_to_numpy(real_line[i])
            a_group = np.concatenate([syn_np, ref_np, real_np], axis=1) # hazed, dehazed, clear
            line_list.append(a_group)


        fill_nb = cfg.pics_line - nb_per_line
        while fill_nb:
            line_list.append(a_blank)
            fill_nb -= 1
        #print(len(line_list))
        #print(line_list[0].shape)
        #print(line_list[1].shape)
        #print(line_list[2].shape)
        #print(line_list[3].shape)
        #print(line_list[4].shape)
        line = np.concatenate(line_list, axis=1)
        #print(line.dtype)
        vertical_list.append(line)

    imgs = np.concatenate(vertical_list, axis=0)
    if imgs.shape[-1] == 1:
        imgs = np.tile(imgs, [1, 1, 3])
    #print(imgs.shape, imgs.dtype)
    img = Image.fromarray(imgs)

    img.save(png_path, 'png')


def generate_batch_train_image(hazy, normalized, GT, path, epoch):
        #print('=' * 50)
        #print('Generating a batch of training images...')
                
        pic_path = os.path.join(path, 'step_%d.png' % (epoch+1))
        generate_img_batch(hazy.cpu().data, normalized.cpu().data, GT.cpu().data, pic_path)
        #print('=' * 50)
        
        
        
        
        
        
        
        
        
        