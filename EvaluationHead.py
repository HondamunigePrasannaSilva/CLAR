import torch.nn as nn
from resnet.resnet_1D import *
from resnet.resnet_2D import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EvaluationHead(nn.Module):
    r"""
     Evaluation Head, 
        - Linear evaluation is only a linear without any activation

        - Non linear evaluation has the same architecture of projection
          head.
    """

    def __init__(self, num_classes = 35):
        super(EvaluationHead, self).__init__()

        self.evaluation = nn.Sequential(
                    # for linear evaluation
                    nn.Linear(256,num_classes)

                    # for non linear evaluation
                    #nn.RelU(),
                    #nn.Linear(256, 128),
                    #nn.RelU(),
                    #nn.Linear(256, 128),
        )
        
    def forward(self,x):
        x = self.evaluation(x)
        return x # [batch_size, num_classes]
