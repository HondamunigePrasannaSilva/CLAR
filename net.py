import torch.nn as nn
from resnet.resnet_1D import *
from resnet.resnet_2D import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    r"""
        CLAR model
            input audio and spectogram augmented
            act as input to the encoders made by resnets
            finnaly the outputs are resized by projection heads to 128 vectors
        Returns:
            - Embedding of audio and spectograms
            - Output of resnet1d and resnet2d to traing the evaluationhead
    """
    def __init__(self,img_channels = 3, num_classes = 35):
        super(Net, self).__init__()

        self.resnet_1D = CreateResNet1D(num_classes=num_classes).to(device)
        self.resnet_2D = CreateResNet2D(img_channels=img_channels,num_classes=num_classes).to(device)
        self.projectionHead = nn.Sequential(
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                    )
        self.relu = nn.ReLU()
        self.linear = nn.Linear(128, num_classes)
        
    def forward(self, input_spectogram, input_audio):

        audio = self.resnet_1D(input_audio)
        audio = audio.squeeze()
        audio_emb = self.projectionHead(audio)

        spectograms = self.resnet_2D(input_spectogram)
        spectograms = spectograms.squeeze() #[BATCH_SIZE, 512, 1, 1]
        specs_emb = self.projectionHead(spectograms)
        
        audio_emb = self.relu(audio_emb)
        specs_emb = self.relu(specs_emb)
        
        output = self.linear(torch.cat([audio_emb, specs_emb], dim=0))
        
        # should be 128 size 
        return audio_emb, specs_emb, audio, spectograms, output #[batch_size, feature_dim]

