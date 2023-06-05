import torch.nn as nn
from resnet.resnet_1D import *
from resnet.resnet_2D import *




class Net(nn.Module):
    r"""
        CLAR model
            input audio and spectogram augmented
            act as input to the encoders made by resnets
            finnaly the outputs are resized by projection heads to 128 vectors
    """
    def __init__(self,img_channels = 3, num_classes = 35):
        super(Net, self).__init__()

        self.resnet_1D = CreateResNet1D(num_classes=num_classes)
        self.resnet_2D = CreateResNet2D(img_channels=img_channels,num_classes=num_classes)
        self.projectionHead_audio = nn.Sequential(
                                                   nn.Linear(512, 256),
                                                   nn.ReLU(),
                                                   nn.Linear(256, 256),
                                                   nn.ReLU(),
                                                   nn.Linear(256, 128),
                                                   nn.ReLU()
                                                )
        
        self.projectionHead_spectogram = nn.Sequential(
                                                nn.Linear(512, 256),
                                                nn.ReLU(),
                                                nn.Linear(256, 256),
                                                nn.ReLU(),
                                                nn.Linear(256, 128),
                                                nn.ReLU()
                                               )
        
    def forward(self, input_spectogram, input_audio):

        audio = self.resnet_1D(input_audio)
        audio = audio.squeeze()
        audio_emb = self.projectionHead_audio(audio)

        spectograms = self.resnet_2D(input_spectogram)
        #[BATCH_SIZE, 512, 1, 1]
        spectograms = spectograms.squeeze()
        specs_emb = self.projectionHead_spectogram(spectograms)

        # should be 128 size 
        return audio_emb, specs_emb





model = Net()

spect = torch.rand(size=[32,3,100,100])
audio = torch.rand(size=[32,1,16000])

output = model(spect, audio)

