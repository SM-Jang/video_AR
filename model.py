import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import argparse


def get_pretrained_model(name):
    
    if name == 'resnet':
        model = models.resnet152(pretrained=True)
    elif name == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif name == 'squeezenet':
        model = models.squeezenet1_0(pretrained=True)
    elif name == 'densenet':
        model = models.densenet161(pretrained=True)
    elif name == 'inception':
        model = models.inception_v3(pretrained=True)
    elif name == 'googlenet':
        model = models.googlenet(pretrained=True)
    elif name == 'shufflenet':
        model = models.shufflenet_v2_x1_0(pretrained=True)
    elif name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
    elif name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained=True)
    elif name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained=True)
    elif name == 'resnext50_32x4d':
        model = models.resnext50_32x4d(pretrained=True)
    elif name == 'wide_resnet50_2':
        model = models.wide_resnet50_2(pretrained=True)
    elif name == 'mnasnet':
        model = models.mnasnet1_0(pretrained=True)
        
    print("Get Pre-trained Model:", name)
    
    return model

#############################################################################################
#############################################################################################
#############################      FineTuneModels        ####################################
#############################################################################################
#############################################################################################


class FineTuneModel(nn.Module):
    def __init__(self, dim, num_classes):
        super(FineTuneModel, self).__init__()
        

        
        vgg16 = get_pretrained_model('vgg16')
        vgg16.features[-3] = nn.Conv2d(512, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        vgg16.classifier[0] = nn.Linear(in_features=25*dim, out_features=4096, bias=True)
        vgg16.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        
        
        self.model = nn.Sequential(
            vgg16.features,
            nn.AdaptiveAvgPool2d((1,1)),
            
        )
        self.classifier = vgg16.classifier

#         self.vgg16.classifier[0] = nn.Linear(dim*49, 4096)
#         self.vgg16.classifier[-1] = nn.Linear(4096, num_classes)
        
    def forward(self, x):

        
        x = self.model(x)
        x = x.reshape(-1)
        x = self.classifier(x)
        
        return x

    
class FinetuneResNet(nn.Module):
    def __init__(self, dim, k):
        # dim: Embedding dimension
        # k: the number of category
        super(FinetuneResNet, self).__init__()

        resnet = models.resnet152(pretrained=True)
        module = resnet.children()
        self.model = nn.Sequential(
            nn.Sequential(*list(module))[:-2],
            
            
            nn.Conv2d(2048, dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        self.fc = nn.Linear(in_features=dim, out_features=k, bias=True)

    def forward(self, x):
        x = self.model(x)
        x = x.squeeze()
        x = self.fc(x)
        
        return x # [number of frame, k(category)]
    
#############################################################################################
#############################################################################################
#############################         Classifier         ####################################
#############################################################################################
#############################################################################################

        


class UCF_DNN(nn.Module):
    def __init__(self, h_in, h1, h2, h3, h4, h5, h_out):
        super(UCF_DNN, self).__init__()
        self.h_in = h_in
        
        self.fc = nn.Sequential(
            nn.Linear(h_in,h1),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            
            nn.Linear(h1,h2),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            
            nn.Linear(h2,h3),
            nn.ReLU(),
            
            nn.Linear(h3,h4),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            
            nn.Linear(h4,h_out)
        )
        
    def forward(self, x):
        x = x.reshape(-1, self.h_in) # T * D 
        x = self.fc(x)
        return x

    
class UCF_CNN1D(nn.Module):
    ## toy version ## 
    def __init__(self, h_in, h_1, h_2, h_3, h_4, h_5, h_out):
        super(UCF_CNN1D, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(h_in, h_1, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv1d(h_1, h_2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(h_2, h_3, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(32768, h_4),
            nn.ReLU(),
            nn.Linear(h_4, h_5),
            nn.ReLU(),
            nn.Linear(h_5, h_out)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, x.shape[1]*x.shape[2])
        x = self.fc(x)
        return x
    


    
class UCF_CNN2D(nn.Module):
    def __init__(self, h_in, h_1, h_2, h_3, h_4, h_5, h_out):
        super(UCF_CNN2D, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(h_in, h_1, kernel_size=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(0.3),
            nn.Conv2d(h_1, h_2, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(h_2, h_3, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(4096, h_4),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.Linear(h_4, h_5),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.Linear(h_5, h_out)
        )
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
        
        
        



def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)# 예시: 'resnet18'

    
    return parser.parse_args()

if __name__ == '__main__':
    
    args = get_arguments()
    model = get_pretrained_model(args.name)
    print(model)
    
    if args.name == 'squeezenet':
        model.classifier[1] = nn.Conv2d(512, 100, kernel_size=(1,1), stride=(1,1))
        print(model)