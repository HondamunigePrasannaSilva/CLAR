import torch
import torch as nn
from torch import optim
import wandb
from tqdm import tqdm
import utils as utils
from data import *
from net import *
from augmentation import *
from contrastive_loss import *
#check for cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


hyperparameters = {
        'LR': 0.001,
        'EPOCHS': 5,
        'BATCH_SIZE': 16,
        'IMG_CHANNEL': 3,
        'CLASSES': 35,
        'AUGMENTATION_1': 'pitchshift',
        'AUGMENTATION_2': 'pitchshift',
        'MODEL_TITLE':'TEST1'
}


def model_pipeline():

    
    with wandb.init(project="CLAR-train", config=hyperparameters, mode="disabled"):
        #access all HPs through wandb.config
        config = wandb.config

        #make the model, data and optimization problem
        model, loss, optimizer, trainloader, testloader, valloader, Augmentation = create(config)

        #train the model
        train(model, loss, optimizer, trainloader,config, Augmentation)

        #test the model
        #print("Accuracy test: ",test(model, testloader))
        
    
def contrastiveLoss():
    return 0

def create(config):

    # Get dataloaders
    trainloader,testloader, valloader  = getData(batch_size=config.BATCH_SIZE)
    
    # Create model
    model = Net(img_channels=config.IMG_CHANNEL, num_classes = config.CLASSES).to(device)

    # Define the constrastive loss
    loss = ContrastiveLoss(batch_size=config.BATCH_SIZE)

    # Create Augmentation
    Augmentation = Augment(config.AUGMENTATION_1, config.AUGMENTATION_2)

    # Define the optimizer, the paper use  
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    return model, loss, optimizer, trainloader, testloader, valloader, Augmentation


def train(model, closs, optimizer, trainloader,config, Augmentation):

    #telling wand to watch
    if wandb.run is not None:
        wandb.watch(model, optimizer, log="all", log_freq=1)

    losses = []
    closs = nn.MSELoss()
    for epoch in range(config.EPOCHS):
        progress_bar = tqdm(total=len(trainloader), unit='step')
        
        for audio,labels in trainloader:
            optimizer.zero_grad()
            # Create augmentation 
            spectograms,audios = createModelInput(Augmentation, audio)            

            # Model's ouput two emb vectors
            audio_emb, spect_emb = model(spectograms,audios)
            #audio_emb, spect_emb = torch.rand(size=[audio.shape[0], 128]).requires_grad_(), torch.rand(size=[audio.shape[0], 128]).requires_grad_()
           #audio_emb , spect_emb = audio_emb.to(device) , spect_emb.to(device)
            # Calculate loss and backward
            loss = closs(audio_emb, spect_emb)
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
        torch.save(model.state_dict(), f"models/model_{config.MODEL_TITLE}.pt")
    
    return

def createModelInput(Augmentation_, audio):

    # CALCUALTE AUGMENTATION 1 AND AUGMENTATION 2
    #audio_1= Augmentation_(audio)
    audio_1 = fade_in_out(audio)
    # CREATE THE FINAL BATCH
    #audios = torch.cat([audio_1, audio_2], dim=0)

    # Create the augmented spectograms size [BATCH_SIZE, 3, 200, 200]
    #spectograms = createSpectograms(audio_1)
    spectograms = torch.rand(size=[audio.shape[0],3, 128, 126])
    # Insert them on GPU
    audios = audio_1.to(device)
    spectograms = spectograms.to(device)

    return  spectograms, audios



model_pipeline()