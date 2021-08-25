import os
import torch
import torch.nn as nn
import numpy as np
import time
import torch.optim as optim
import natsort
import bezier

from sklearn.decomposition import PCA
from dataset import encoder
from model import get_pretrained_model
from dataset import jpg2np, get_loader
from torchvision import models
from bezier.hazmat.curve_helpers import evaluate_hodograph, get_curvature
from torch.utils.data import TensorDataset, DataLoader
from torch.utils import data

class UCFdataset(data.Dataset):
    def __init__(self, upper, lower, dir_path): # 'train', 'validation'
        super(UCFdataset, self).__init__()
        
        self.file_list, self.y, self.video_names = self.file_load(upper, lower, dir_path)
        
        
    def __getitem__(self, index):
        
        x = jpg2np(self.file_list[index]) / 255. # (30, 3, 240, 320)
        self.x_data = torch.from_numpy(x).float()
        self.y_data = torch.from_numpy(self.y[index]).float()
        return self.x_data, self.y_data, self.video_names[index]

    def __len__(self):
        return self.y.shape[0]
    
    def file_load(self, upper, lower, dir_path):
        """
        return the input file path list
        """
        data_path = []
        video_imgs_path = os.path.join(os.getcwd(), dir_path)
        folders = os.listdir(video_imgs_path)

        frames = {}
        for folder in folders:
            path = os.path.join(video_imgs_path, folder)
            frames[folder] = len(os.listdir(path))

        video_names = []
        for video_name, num_of_frames in zip(list(frames.keys()), list(frames.values())):
            if upper <= num_of_frames and num_of_frames <= lower:
                video_names.append(video_name)
        video_names = natsort.natsorted(video_names)
                
        print("Select The number of frames between [%d, %d] of UCF101 Dataset" %(upper, lower))
        print('The number of selected videos is', len(video_names))


        data_path = [os.path.join(video_imgs_path, video_name) for video_name in video_names]

        labels = []
        for label in video_names:
            loc1 = label.find('_')
            loc2 = loc1 + label[loc1+1:].find('_')
            labels.append(label[loc1+1:loc2+1])
        y, _ = encoder(labels)
#         y = torch.tensor(y, dtype=torch.float) # [N(13320, 1)

        return data_path, y, video_names

