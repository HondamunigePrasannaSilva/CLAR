import torch
import torch as nn
from torch import optim
import wandb
from tqdm import tqdm
import utils as utils
from data import *
from net import *
from augmentation import *

#check for cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


hyperparameters = {
        'LR': 0.001,
        'EPOCHES': 100,
        'BATCH_SIZE': 32,
        'IMG_CHANNEL': 3,
        'CLASSES': 35,
        'AUGMENTATION_1': 'TEST1',
        'AUGMENTATION_2': 'TEST2'
}


def model_pipeline(classifierName, datasetname, trainloader, testloader):

    
    with wandb.init(project="CLAR-train", config=hyperparameters, mode="disabled"):
        #access all HPs through wandb.config
        config = wandb.config

        #make the model, data and optimization problem
        model, loss, optimizer, trainloader, testloader, valloader = create(config)

        #train the model
        train(model, loss, optimizer, trainloader)

        #test the model
        #print("Accuracy test: ",test(model, testloader))
        
    
def contrastiveLoss():
    return 0

def create(config):

    # Get dataloaders
    trainloader,testloader, valloader  = getData(batch_size=config.BATCH_SIZE)
    
    # Create model
    model = Net(img_channels=config.IMG_CHANNEL, num_classes = config.CLASSES)

    # Define the constrastive loss
    loss = contrastiveLoss()

    # Define the optimizer, the paper use  
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    return model, loss, optimizer, trainloader, testloader, valloader


def train(model, loss, optimizer, trainloader,config, modeltitle = "test"):

    #telling wand to watch
    if wandb.run is not None:
        wandb.watch(model, optimizer, log="all", log_freq=1)

    losses = []

    for epoch in range(config.EPOCHS):
        progress_bar = tqdm.tqdm(total=len(trainloader), unit='step')
        
        for audio,labels in trainloader:

            optimizer.zero_grad()

            # CALCUALTE AUGMENTATION 1
            audio_1 = getTransform(config.AUGMENTATION_1, audio)
            # CALCUALTE AUGMENTATION 2
            audio_2 = getTransform(config.AUGMENTATION_2, audio)
            
            # CREATE THE FINAL BATCH
            audio = createFinalbatch(audio_1, audio_2)
            
            # Create the augmented spectograms size [BATCH_SIZE, 3, 200, 200]
            spectograms = createSpectogtrams(audio)

            # Model's ouput two emb vectors
            audio_emb, spect_emb = model(spectograms,audio)
            
            # Calculate loss and backward
            loss = loss(audio_emb, spect_emb)
            loss.backward()
            optimizer.step()

            
            #progress bar stuff
            progress_bar.set_description(f"Epoch {epoch+1}/{config.EPOCHS}")
            progress_bar.set_postfix(loss=loss.item())  # Update the loss value
            progress_bar.update(1)

            # save loss to statistics
            losses.append(loss.item())
            
        # end for batch 
        
        # Log on wandb at each epoch
        if wandb.run is not None:
            wandb.log({"epoch":epoch, "loss":np.mean(losses)}, step=epoch)
        
        # save the model
        torch.save(model.state_dict(), "models/model"+str(modeltitle)+".pt")
    
    return 0






def createSpectogtrams(audio):
    return torch.rand(size=[32,3,200,200])


def createFinalbatch(bs1, bs2):
    bs1 = torch.rand(size=[32,1,16000])
    bs2 = torch.rand(size=[32,1,16000])
    return torch.cat([bs1, bs2], dim = 0) # sould be [2*BATCH_SIZE, 1, 16000]