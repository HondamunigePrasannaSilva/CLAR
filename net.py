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
    """
    def __init__(self,img_channels = 3, num_classes = 35):
        super(Net, self).__init__()

        self.resnet_1D = CreateResNet1D(num_classes=num_classes).to(device)
        self.resnet_2D = CreateResNet2D(img_channels=img_channels,num_classes=num_classes).to(device)
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
        return audio_emb, specs_emb #[batch_size, feature_dim]


"""model = Net().to(device)

audio = torch.rand(size=[8,1,16000]).to(device)
spectogram = torch.rand(size=[8,3,129, 126]).to(device)

for i in range(1000):
    audio_emb, spec_emb = model(spectogram, audio)

print(1)
"""