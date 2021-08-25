import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model import get_pretrained_model, FineTuneModel
from dataset import jpg2np, get_loader


from bezier.hazmat.curve_helpers import evaluate_hodograph, get_curvature
from sklearn.preprocessing import StandardScaler


import argparse
import pdb
# from sklearn.decomposition import PCA



def embedding(model, path, device, name):

    ## return Frame-Wise Embedding Matrix ##
    result = list()

    ## model ##
    model = model.to(device)
    
    
    ## loader ##
    videoFrames = jpg2np(path) # 한 비디오의 이미지 프레임을 5칸씩 정렬된 순서로 np로 전환
    loader = get_loader(videoFrames)
    
    
    total = len(os.listdir(path))
    
    
    ## Frame-wise Embedding ##
    for i, frame in enumerate(loader):
        if i+1 % 10 == 0:
            print("[%d, %d]-th frame is embedding by %s" %(i+1, total, name))


        frame = frame.float().to(device)
        
        ## model embedding ##
        embedding = model(frame).detach().cpu().numpy().squeeze()
        result.append(embedding)

    
    return np.stack(result)




def bezier(embeddings, name, frames):
    import bezier # 여기서 해야지 curve 부분이 실행이 됨...

#     pdb.set_trace()
    ## return Cuvature on bezier curve ##
    kappa = list()
    
    embeddings = embeddings.squeeze().T # make 2 dimension
    print("The Embedding points shape is", embeddings.shape)
    

    # Bezier Curve Approximation
    curve = bezier.Curve.from_nodes(embeddings)


    if embeddings.shape[0] == 2:
        curve.plot(num_pts=256)
        print('save img {}'.format(name))
        plt.savefig('./kappa/img/{}.png'.format(name)) 

        
    print("The information of the Bezier Curve is", curve)


    ## Calculate kappa (t: 0~100) ##
    for s in range(frames):
        t = s / frames
        tangent_vec = curve.evaluate_hodograph(t)
        kappa.append(abs(get_curvature(embeddings, tangent_vec, t)))
    
    return kappa
    
    
    
def get_arguments():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', type=str)# 예시: 'resnet18'
    parser.add_argument('--dir_path', type=str)# 예시: './50salads_image/rgb-27-2'
    parser.add_argument('--cuda', type=int)
    parser.add_argument('--dims', type=int)
    parser.add_argument('--frames', type=int)
    parser.add_argument('--model_path', type=str, default='./record/bin/')
    
    
    return parser.parse_args()


if __name__ == '__main__':

    
    args = get_arguments()

    
    if os.path.isdir('./kappa/{}-{}'.format(args.name, args.dims)) == False: 
        os.mkdir('./kappa/{}-{}'.format(args.name, args.dims))

        
    ## gpu setting ##
    GPU_NUM = args.cuda # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print('Current cuda device ', torch.cuda.current_device())
    

    ## select 150 ~ 300 frames video ##
    video_imgs_path = os.path.join(os.getcwd(), args.dir_path)
    folders = os.listdir(video_imgs_path) ## all videos path
    
    
    frames = dict()
    for folder in folders:
        ## Calculate the number of video frames
        path = os.path.join(video_imgs_path,folder)
        frames[folder] = len(os.listdir(path)) 

        
        
    ## Select the sub-set ##
    video_names = list()
    for k, f in zip(list(frames.keys()), list(frames.values())):
        ## save 150 ~ 200 frames video names ##
        if (150 <= f) :
            video_names.append(k)

    print('The number of selected videos is', len(video_names)) # 500

    with open("./kappa/{}-{}/labels.txt".format(args.name, args.dims), 'a+') as f:
        for name in video_names: f.write(name+'\n')


    ## Modify The Pre-Trained Model for embedding ##
#     model = get_pretrained_model(args.name)
#     if args.name == 'vgg16':
#         model = nn.Sequential(model.features, 
#                               nn.AdaptiveAvgPool2d((1,1)))
        
    if args.name == 'fine_tune':
        model = FineTuneModel(args.dims, 101)
        model.load_state_dict(torch.load(args.model_path+'FT_epoch3.pth'))
        model = model.model
        

    ## Embedding for one video ##
    embeddings_list = list()
    for i, file in enumerate(video_names):
        print("[{}/{}] {}".format(i, len(video_names), file))
        path = os.path.join(args.dir_path, file)
        embeddings_list.append(embedding(model, path, device, args.name)) # 1-video = [frames, embedding_dims(512 by vgg16)]

    embeddings = np.stack(embeddings_list)


    ## PCA reduce dimension of embeddings matrix ##
#     embeddings = embeddings.reshape(embeddings.shape[0]*embeddings.shape[1], -1)
#     pca = PCA(args.dims)
#     embeddings = pca.fit_transform(embeddings).reshape(len(video_names), -1, args.dims)
    

    ## Bezier Approximation ##
    kappa = list()
    for e in embeddings:
        kappa.append(bezier(e, path[path.rfind('/')+1:], args.frames))
    kappa = np.stack(kappa) # [frames,]


## kappa normalize 필요 & modify 필요 ##
#     elementwise = list()
#     for i, e in enumerate(embeddings):
#         elementwise.append(e*kappa[i])
#     elementwise = np.stack(elementwise) # [frames(30), embedding_dims]


    ## save the recordings ##
    np.save("./kappa/{}-{}/kappa.npy".format(args.name, args.dims), kappa)
#     np.save("./kappa/{}-{}/elementwise.npy".format(args.name, args.dims), elementwise)
    np.save("./kappa/{}-{}/embeddings.npy".format(args.name, args.dims), embeddings)
    
    print("Kappa, Embeddings, Elementwise  Done!")

    
