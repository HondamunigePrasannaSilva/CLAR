import torch.nn as nn
from resnet.resnet_1D import *
from resnet.resnet_2D import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EvaluationHead(nn.Module):
    r"""
     Evaluation Head, has the same architecture of projection head
     3 MLP with relu activation
    """
    def __init__(self, num_classes = 35):
        super(EvaluationHead, self).__init__()

        self.linear_1 = nn.Linear(256,num_classes)
        #self.linear_2 = nn.Linear(256,128)
        #self.linear_3 = nn.Linear(128,num_classes)

        #self.relu = nn.ReLU()
                
    def forward(self,x):
        
        x = self.linear_1(x)
        #x = self.relu(x)
        #x = self.linear_2(x)
        #x = self.relu(x)
        #x = self.linear_3(x)
        return x # [batch_size, num_classes]
