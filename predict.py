import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from model import UCF_DNN, UCF_CNN1D, UCF_CNN2D
from dataset import encoder, get_dataset
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


import argparse
import pdb

from dataset import encoder

    
    
def train(loader,test_loader, model, epochs, criterion, optimizer, device, check_point, bs, mode, data_dir):
    model.train()
    for epoch in range(epochs):
        bs_step = 0
        correct = 0
        for x, y in loader: 
            
            x, y_true = x.to(device), y.squeeze().long().to(device)
            optimizer.zero_grad()
            
#             pdb.set_trace()
            y_pred = model(x)
            _, predicted = torch.max(y_pred, 1)
            loss = criterion(y_pred, y_true)
            
            loss.backward()
            optimizer.step()
            bs_step += bs
            correct += (predicted == y_true).sum().item()


        train_acc = 100 * correct / bs_step
        test_acc = test(model, test_loader, device)    
        
            
        print("Epoch [{:3d} / {}] Loss: {:.5f} | Train acc: {:2.2f} | Test acc {:2.2f}".format(epoch+1, epochs, loss.item(), train_acc, test_acc))
    with open('{}result.txt'.format(data_dir), 'a') as f:
        f.write("data: {}\n".format(mode))
        f.write("Epoch [{:3d} / {}] Loss: {:.5f} | Train acc: {:2.2f} | Test acc {:2.2f}\n\n".format(epoch+1, epochs, loss.item(), train_acc, test_acc))
        

                

def test(model,test_loader,device):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        
        flag = 0

        for x, y in test_loader:
            x, labels = x.to(device), y.squeeze().long().to(device)

            y_pred = model(x)
            _, predicted = torch.max(y_pred, 1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()

    return 100 * correct / total







def get_arguments():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cuda', type=int, default=3)
    parser.add_argument('--num_of_classes', type=int, default=101)
    parser.add_argument('--data_dir', type=str, default = './kappa/fine_tune-100/')
    parser.add_argument('--mode', type=str, default = 'elementwise_minmax')

    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default = 0.0001)
    parser.add_argument('--bs', type=int, default = 100)
    parser.add_argument('--check_point', type=str, default = './check_point/vgg16-512')
    parser.add_argument('--name', type=str)
    
    
    return parser.parse_args()
    


if __name__ == '__main__':
    
    w
    args = get_arguments()
    
    if os.path.isdir(args.check_point) == False: 
        os.mkdir(args.check_point)
        os.mkdir(args.check_point+'/predict')
        
    ## gpu setting ##
    GPU_NUM = args.cuda # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print('Current cuda device ', torch.cuda.current_device())
    
    

    
    ## load dataset ##
    x = np.load(args.data_dir+'{}.npy'.format(args.mode)) 
    with open(args.data_dir+'labels.txt', 'r') as f:
        y = f.readlines()

    labels = []
    for label in y:
        loc1 = label.find('_')
        loc2 = loc1 + label[loc1+1:].find('_')
        labels.append(label[loc1+1:loc2+1])
#     pdb.set_trace()
    y, _ = encoder(labels)
    x = torch.tensor(x, dtype=torch.float) # [N(13320), D, Tm(30)]
    y = torch.tensor(y, dtype=torch.float) # [N(13320, 1)

    ## train / test split ##
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123)
    
    train_dataset = TensorDataset(x_train,y_train)
    test_dataset = TensorDataset(x_test,y_test)
    
    train_loader = DataLoader(train_dataset, args.bs)
    test_loader = DataLoader(test_dataset, args.bs)
#     pdb.set_trace()
    if args.name == 'UCF_CNN2D':
        ## UCF_CNN2D ##
        h_in, h_out = 30, args.num_of_classes # UCF101
        h1, h2, h3, h4, h5 = 256, 128, 64, 1000, 500
        model = UCF_CNN2D(h_in, h1, h2, h3, h4, h5, h_out).to(device)
        name = '1. model: {}'.format(args.name)
        
    if args.name == 'UCF_CNN1D':
        ## UCF_CNN1D ##
        h_in, h_out = 30, args.num_of_classes # UCF101
        h1, h2, h3, h4, h5 = 256, 128, 64, 1000, 500
        model = UCF_CNN1D(h_in, h1, h2, h3, h4, h5, h_out).to(device)
        name = '2. model: {}'.format(args.name)

    if args.name == 'UCF_DNN':
        ## DNN ##
        h_in, h_out = x.shape[1]*x.shape[2], args.num_of_classes # UCF101
        h1, h2, h3, h4, h5 = 512, 256, 128, 128, 100 
        model = UCF_DNN(h_in, h1, h2, h3, h4, h5, h_out).to(device)
        name = '3. model: {}'.format(args.name)
    
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,  weight_decay=1e-5)
    print(model)
#     pdb.set_trace()
    with open('{}result.txt'.format(args.data_dir), 'a') as f:
        f.write(name+'\n')

    train(train_loader, test_loader, model, args.epochs, criterion, optimizer, device, args.check_point, args.bs, args.mode, args.data_dir)


    
    
    

