import torch
import torch.nn as nn
import numpy as np
#from data import *



class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None,stride=1):
        super(block, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.identity_downsample = identity_downsample
        self.relu = nn.ReLU()


    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity
        x = self.relu(x)

        return x
    
class ResNet1D(nn.Module):
    # Resnet 18 [2, 2, 2, 2]
    def __init__(self, block, num_classes):
        super(ResNet1D, self).__init__()
        # for resnet18
        layers = [2, 2, 2, 2]
        self.expansion = 1

        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, layers[0], 64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
        # tagliare qui per prendere 
        # size after avgpool = [32, 512, 1]
        #self.fc = nn.Linear(512*self.expansion, num_classes)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        #x = x.reshape(x.shape[0], -1)
        #x = self.fc(x)

        return x



    def _make_layer(self, block, num_residual_block, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1:
            identity_downsample = nn.Sequential(nn.Conv1d(self.in_channels, 
                                                out_channels*self.expansion,
                                                kernel_size=1,
                                                stride=stride,
                                                bias=False),
                                                nn.BatchNorm1d(out_channels*self.expansion),
                                                )
        layers.append(
            block(self.in_channels,out_channels, identity_downsample, stride)
        )
        self.in_channels = out_channels * self.expansion

        for i in range(1, num_residual_block):
            layers.append(block(self.in_channels,out_channels ))

        return nn.Sequential(*layers)



def CreateResNet1D( num_classes = 10):
    return ResNet1D(block, num_classes=num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm






""" 
def model_pipeline():


    #make the model, data and optimization problem
    model, criterion, optimizer, trainloader, testloader, validationloader = create()

    #train the model
    train(model, trainloader, criterion, optimizer, validationloader)

    #test the model
    print("Accuracy test: ",test(model, testloader))
        
    #return model

def create():
    resnet1d = CreateResNet1D(num_classes=35)
    #Create a model
    model = resnet1d.to(device)
    nparameters = sum(p.numel() for p in model.parameters())
    print(nparameters)
    #Create the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainloader,testloader,validationloader = getData()

    return model, criterion, optimizer,trainloader, testloader, validationloader

# Function to train a model.
def train(model, trainloader, criterion, optimizer, validationloader):
 

    model.train()
    losses, valacc = [], []  

    for epoch in range(5):
        
        progress_bar = tqdm(trainloader, desc=f'Training epoch {epoch}', leave=False)
        
        for batch, (images, labels) in enumerate(progress_bar):
        
            loss = train_batch(images, labels,model, optimizer, criterion)
            progress_bar.update(1)
            
            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
        
            losses.append(loss.item())

    return np.mean(losses)

def train_batch(images, labels, model, optimizer, criterion):

    #insert data into cuda if available
    images,labels = images.to(device), labels.to(device)
    
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    #backward pass
    loss.backward()

    #step with optimizer
    optimizer.step()

    return loss

def test(model, test_loader):
    model.eval()

    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            oututs = model(images)
            _, predicated = torch.max(oututs.data, 1)
            total += labels.size(0)

            correct += (predicated == labels).sum().item()

        return correct/total
    

model_pipeline()"""