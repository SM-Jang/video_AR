import os
import numpy as np
from PIL import Image
import argparse
import torch

from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import natsort 

import pdb

#############################################################
################### for predict.py ##########################
#############################################################


def encoder(label):
    
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)

    # define example
    values = np.array(label)
    
    # integer encode
    integer_encoded = label_encoder.fit_transform(values)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    
    # onehot encode
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    
    return integer_encoded, onehot_encoded



def get_dataset(data_dir):

    ## list all data path ##
    data_path = list()
    for file in os.listdir(data_dir):
        if file[-4:] == '.npy': 
            data_path.append(os.path.join(data_dir, file))
            
            
    ## Step1. load .npy data as list ##
    x = list()
    temp = list()
    
    for file in data_path:
        video = np.load(file)
        temp.append(video) # [Number of Videos, Frames, Dimension]

        
    ## Fix the number of frames from 0 to min_size ##
    m = 1000000
    for frame in temp:
        if m > frame.shape[0]  : m = frame.shape[0]
            
    for t in temp:
        x.append(t[:m])
    
    ## Step2. make Groud True(=y) vector ##
    ## list .npy data path ##
    data_path = list()
    for file in os.listdir(data_dir):
        if file[-4:] == '.npy': 
            data_path.append(os.path.join(data_dir, file))
            
            
    ## make label vector ##
    label = list()
    for file in data_path:
        loc1 = file.find('_')
        loc2 = loc1 + file[loc1+1:].find('_')
        if file[loc1+1:loc2] == 'checkpoint': continue
        label.append(file[loc1+1:loc2+1])

        
    y, _ = encoder(label) # One-hot Vector

    return np.transpose(np.stack(x), (0, 2, 1) ), y



#############################################################
#####################  for main.py ##########################
#############################################################


def jpg2np(dir_path): 
    """
    dir_path: 데이터 이미지 폴더의 path
    
    video frame(jpg)을 numpy형태로 전환시키는 함수
    shape = [frames, channel, width, height]
    """
#     pdb.set_trace()
    
    result = list()
    cur_path = os. getcwd()
    path = os.path.join(cur_path, dir_path)
    data_list = os.listdir(path)
    data_list = natsort.natsorted(data_list,reverse=False) # sorting the files

    for i, data in enumerate(data_list[::5]):
        if i == 30: break # 5칸씩 30개까지만
        data_path = os.path.join(path, data)

        # Open the image form working directory
        image = Image.open(data_path)

        # convert image to numpy array
        data = np.asarray(image)
        data = np.transpose(data, (2, 0, 1))

        result.append(data)

    result = np.stack(result, axis=0)
    
    return result



def get_loader(videoFrames):
    """
    video: 비디오를 프레임 셋으로 전환한 numpy 형태
    
    1개의 video frames를 받아 1frame씩 load
    return shape is [1-frame, channel, width, height]
    """
    video = torch.from_numpy(videoFrames)
    
    loader = DataLoader(dataset=video,
                             batch_size=1,
                             shuffle=True)
    return loader


#############################################################
#############################################################
#############################################################


def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str) 
    
    
    return parser.parse_args()
    
    
    
if __name__ == '__main__':
    
    args = get_arguments()
    
    video = jpg2np(args.dir_path)
    loader = get_loader(video)
    
    # Loader Test
    for i, frame in enumerate(loader):
        if i == 10: break
        
        print(frame.shape)
