import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as data


import matplotlib.pyplot as plt
import pickle
import pdb

from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from model import FineTuneModel, resnet152FineTune

def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    model.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    model.train()
    for epoch in range(epochs):
        for batch_idx, (Xs, y) in enumerate(train_loader):
            y = y.to(device).view(-1, )
            N_count += Xs.size(0)

            output = []
            for i, X in enumerate(Xs):
                X = X.to(device)
                output.append(model(X))  
                
            output = torch.stack(output)  # output size = (batch, number of classes)

            optimizer.zero_grad()
            loss = F.cross_entropy(output, y)
            losses.append(loss.item())

            # to compute accuracy
            y_pred = torch.max(output, 1)[1]  # y_pred != output
            step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
            scores.append(step_score)         # computed on CPU

            loss.backward()
            optimizer.step()

            # show information
            if (batch_idx + 1) % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}%'.format(
                    epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

    return losses, scores

def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for i, (Xs, y) in enumerate(test_loader):
            # distribute data to device
            y =  y.to(device).view(-1, )
            
            output = []
            for i, X in enumerate(Xs):
                X = X.to(device)
                output.append(model(X))  

            output = torch.stack(output)  # output size = (batch, number of classes)


            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # to compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))
    
    # save Pytorch models of best record
    torch.save({
        'model': model.state_dict(),
        'optimizer':optimizer.state_dict()
    }, './record/FT_{}_epoch_{}.pth'.format(dim, epoch + 1))
    
    print("Epoch {} model saved!".format(epoch + 1))


    return test_loss, test_score

if __name__ == '__main__':

    ## set path ##
    data_path = "./ucf_image/"    # define UCF-101 spatial data path
    action_name_path = "./UCF101actions.pkl"  # load preprocessed action names
    save_model_path = "./record/"  # save Pytorch models

    ## gpu setting ##
    GPU_NUM = 0
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print('Current cuda device ', torch.cuda.current_device())


    ## training parameters ##

    epochs = 10
    batch_size = 1
    learning_rate = 1e-4
    log_interval = 10
    img_x, img_y = 240, 320  # resize video 2d frame size

    ## model parameter ##
    dim = 100
    k = 101                 # number of target category

    ## Loader parameter ##
    params = {'batch_size': batch_size, 'shuffle': True}

    ## load UCF101 actions names ##
    with open(action_name_path, 'rb') as f:
        action_names = pickle.load(f)   # load UCF101 actions names    
    action_names[58] = 'HandStandPushups' # fix the label name to match the folder name

    ## convert labels -> category ##
    le = LabelEncoder()
    le.fit(action_names)
    action_category = le.transform(action_names).reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(action_category)
    actions = []
    fnames = os.listdir(data_path)
    all_names = []
    for f in fnames:
        loc1 = f.find('v_')
        loc2 = f.find('_g')
        actions.append(f[(loc1 + 2): loc2]) # 저장되어 있는 파일 순서대로 labeling

        all_names.append(f)

    ## list all data files ##
    all_X_list = all_names              # all video file names
    all_y_list = labels2cat(le, actions)    # all video labels

    ## train, test split ##
    train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.25, random_state=42)

    ## total 25 frames (min size) ##
    begin_frame, end_frame, skip_frame = 1, 26, 1 
    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()
    print("The total Number of selected frame is", len(selected_frames))


    ## image transformation ##
    transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])

    ## UCF101 30 images dataset ##
    train_set, valid_set = Dataset_3DCNN(data_path, train_list, train_label, selected_frames, transform=transform), \
                           Dataset_3DCNN(data_path, test_list, test_label, selected_frames, transform=transform)


    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)

    model = resnet152FineTune(dim, k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)   # optimize all cnn parameters


    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_test_losses = []
    epoch_test_scores = []


    for epoch in range(epochs):
        # train, test model
        train_losses, train_scores = train(log_interval, model, device, train_loader, optimizer, epoch)
        epoch_test_loss, epoch_test_score = validation(model, device, optimizer, valid_loader)

        # save results
        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_test_losses.append(epoch_test_loss)
        epoch_test_scores.append(epoch_test_score)

        # save all train test results
        A = np.array(epoch_train_losses)
        B = np.array(epoch_train_scores)
        C = np.array(epoch_test_losses)
        D = np.array(epoch_test_scores)
        np.save('./record/FTmodel_epoch_training_losses{}.npy'.format(dim), A)
        np.save('./record/FTmodel_epoch_training_scores{}.npy'.format(dim), B)
        np.save('./record/FTmodel_epoch_test_loss{}.npy'.format(dim), C)
        np.save('./record/FTmodel_epoch_test_score{}.npy'.format(dim), D)

        # plot
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
    plt.plot(np.arange(1, epochs + 1), C)         #  test loss (on epoch end)
    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc="upper left")
    # 2nd figure
    plt.subplot(122)
    plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train accuracy (on epoch end)
    plt.plot(np.arange(1, epochs + 1), D)         #  test accuracy (on epoch end)
    # plt.plot(histories.losses_val)
    plt.title("training scores")
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'], loc="upper left")
    title = "./record/fig_UCF101_FT{}.png".format(dim)
    plt.savefig(title, dpi=600)
    plt.close(fig)
#     plt.show()
