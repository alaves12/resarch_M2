import torch
from PIL import Image
from torchvision import transforms as T
import random
from torchvision.transforms import functional as TF
import os

def readfile(name, dataset, train=True):
    hazyfiles = []
    cleanfiles = []
    depthfiles = []
    lightfiles = []

    if train:
        with open('./datas/datasets/{}.txt'.format(dataset), 'r') as f:
            for line in f:
                filenames = line.split()
                cleanfiles.append(filenames[1])
                hazyfiles.append(filenames[2])
                depthfiles.append(filenames[3])   
                lightfiles.append(filenames[4])
        
    if not train:
        with open('./datas/datasets/Val_1.txt', 'r') as f:
            for line in f:
                filenames = line.split()
                cleanfiles.append(filenames[1])
                hazyfiles.append(filenames[2])
                depthfiles.append(filenames[3])         

    if name == 'clean':
        return cleanfiles

    elif name == 'hazy':
        return hazyfiles

    elif name == 'depth':
        return depthfiles

    elif name == 'light':
        return lightfiles

class MyDataset():
    def __init__(self,dataset,crop_size,length=None,flip=True,rotate=True):
        self.clean = readfile('clean', dataset)
        self.hazy = readfile('hazy', dataset)
        self.depth = readfile('depth', dataset)
        self.light = readfile('light', dataset)

        self.crop_size = crop_size
        self.flip = flip
        self.rotate = rotate
        self.length = length

    def transform(self,images):
        y,x = images[0].size
        if y <= self.crop_size[0] or x <= self.crop_size[0]:
            images = [T.Resize((2*x,2*y))(image) for image in images]

        if self.crop_size:
            i,j,h,w = T.RandomCrop.get_params(images[0],output_size=self.crop_size)
            images = [TF.crop(image,i,j,h,w) for image in images]
        
        if (random.random() > 0.5) and self.flip:
            images = [TF.hflip(image) for image in images]
        
        seed = random.random()
        if (seed <= 0.25) and self.rotate:
            images = [TF.rotate(image,90) for image in images]
        
        elif (0.25 < seed <= 0.5) and self.rotate:
            images = [TF.rotate(image,180) for image in images]
        
        elif (0.5 < seed <= 0.75) and self.rotate:
            images = [TF.rotate(image,270) for image in images]
        
       

        images = [TF.to_tensor(image) for image in images]

        return images

    def __getitem__(self,index):
        cleanfile = Image.open(self.clean[index]).convert('RGB')
        hazyfile = Image.open(self.hazy[index])

        if self.depth[index] == 'None':
            images = [hazyfile, cleanfile]
            images = self.transform(images)
            input_dict = {'hazy':images[0], 'clean':images[1], 'depth':self.depth[index], 'A': self.light[index]}

        elif self.light[index] == 'None':
            depthfile = Image.open(self.depth[index])
            images = [hazyfile, cleanfile, depthfile]
            images = self.transform(images)
            input_dict = {'hazy':images[0], 'clean':images[1], 'depth':images[2], 'A': self.light[index]}


        else:
            depthfile = Image.open(self.depth[index])
            lightfile = Image.open(self.light[index])

            images = [hazyfile,cleanfile,depthfile,lightfile]
            images = self.transform(images)
            
            input_dict = {'hazy':images[0], 'clean':images[1], 'depth':images[2], 'A':images[3]} 
        
        return input_dict

    def __len__(self):
        if self.length is not None:
            return self.length
        else:
            return(len(self.clean))