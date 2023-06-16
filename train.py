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
from EvaluationHead import *
#check for cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


hyperparameters = {
        'LR': 3e-4,
        'WEIGHT_DECAY': 1e-6,
        'B1':0.9,
        'B2':0.999,
        'EPOCHS': 201,
        'BATCH_SIZE': 128,
        'IMG_CHANNEL': 3,
        'CLASSES': 35,
        'EVAL_BATCH':128,
        'EVAL_EPOCHS':10,
        'N_LABELS': 20,
        'DATASET': 'SpeechCommand',
        'MODEL_TITLE':'fade_tm'
}


def model_pipeline():

    
    with wandb.init(project="CLAR", config=hyperparameters):
        #access all HPs through wandb.config
        config = wandb.config

        #make the model, data and optimization problem
        model, loss,ce_loss, optimizer, trainloader, testloader, valloader = create(config)

        #train the model
        train(model, loss,ce_loss, optimizer, trainloader,config)

        #test the model
        #print("Accuracy test: ",test(model, testloader))
        
    
def create(config):

    # Get dataloaders
    trainloader,testloader, valloader  = getData(batch_size=config.BATCH_SIZE, num_workers=8, pin_memory=False)
    
    # Create model
    model = Net(img_channels=config.IMG_CHANNEL, num_classes = config.CLASSES).to(device)

    # Define the constrastive loss
    loss = ContrastiveLoss(batch_size=config.BATCH_SIZE)
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    # Define the optimizer, the paper use  
    optimizer = optim.Adam(model.parameters(), lr=config.LR, betas=(config.B1, config.B2), weight_decay=config.WEIGHT_DECAY)

    return model, loss,ce_loss, optimizer, trainloader, testloader, valloader


def train(model, closs,ce_loss, optimizer, trainloader,config):

    #telling wand to watch
    if wandb.run is not None:
        wandb.watch(model, optimizer, log="all", log_freq=100)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.EPOCHS):
        progress_bar = tqdm(total=len(trainloader), unit='step')
        losses = []

        for audio,lab in trainloader:
            optimizer.zero_grad()
            
            n_label_ = int(lab.shape[0] * (1-config.N_LABELS/100))
            select_lab_to_CE_loss = random.sample(range(lab.shape[0]), n_label_)
            lab[select_lab_to_CE_loss] = -1

            labels = torch.cat([lab, lab], dim=0).to(device)

            # Create augmentation and spectograms!
            spectograms,audios = createModelInput(audio)            

            # Model's ouput two emb vectors
            with torch.cuda.amp.autocast():
                audio_emb, spect_emb, _, _, output = model(spectograms,audios)
                loss = closs(audio_emb, spect_emb)+ce_loss(output, labels)

            # Calculate loss and backward
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            #progress bar stuff
            progress_bar.set_description(f"Epoch {epoch+1}/{config.EPOCHS}")
            progress_bar.set_postfix(loss=loss.item())  # Update the loss value
            progress_bar.update(1)

            # save loss to statistics
            losses.append(loss.item())
     
           
            
        # end for batch 
        

        # Log on wandb at each epoch
        if wandb.run is not None:
            #wandb.log({"epoch":epoch, "loss":np.mean(losses)}, step=epoch)
            wandb.log({"epoch":epoch, "loss":np.mean(losses)})
        
            
        if epoch%10==0: 
            # EVALUATION HEAD  
            torch.save(model.state_dict(), f"models/model_{config.MODEL_TITLE}.pt")
            accuracy_test, accuracy_validation = evaluationphase(model, config)
            wandb.log({"accuracy_test":accuracy_test, "accuracy_validation":accuracy_validation})
    
    return

def createModelInput(audio, augmentation=True):

    # CALCUALTE AUGMENTATION 1 AND AUGMENTATION 2
    if(augmentation == True):
        audio = fade_in_out(audio)
        audio = timemasking(audio,audio.shape[0])
    
    # Create the augmented spectograms size [BATCH_SIZE, 3, 200, 200]
    spectograms = createSpectograms_(audio)

    # Insert them on GPU
    audios = audio.to(device)
    spectograms = spectograms.to(device)

    return  spectograms, audios


def evaluationphase(model, config):
    
    model.eval()
    # Get dataloaders
    trainloader,testloader, valloader  = getData(batch_size=config.EVAL_BATCH, num_workers=8, pin_memory=False)
    # Freeze the gradients for model1
    for param in model.parameters():
        param.requires_grad = False
    
    def train(model, trainloader):
    
        # Create model
        modelEvaluation = EvaluationHead(num_classes = config.CLASSES).to(device)

        # Define the optimizer, the paper use  
        optimizer = optim.Adam(modelEvaluation.parameters(), lr=config.LR)

        criterion = torch.nn.CrossEntropyLoss()

        scaler = torch.cuda.amp.GradScaler()
    
        
        
        for epoch in range(config.EVAL_EPOCHS):
            progress_bar = tqdm(total=len(trainloader), unit='step', leave=False)
            losses = []
            for audio,labels in trainloader:
                
                optimizer.zero_grad()

                # Create augmentation and spectograms!
                spectograms,audios = createModelInput(audio)

                labels_cat = torch.cat([labels, labels], dim = 0).to(device)

                # Model's ouput two emb vectors
                with torch.cuda.amp.autocast():
                    # Use frozen encoder
                    _, _, frozen_audio, frozen_spects, _ = model(spectograms, audios)
                    inputs = torch.cat([frozen_audio, frozen_spects], dim = 0) 
                    outputs = modelEvaluation(inputs)
                    loss = criterion(outputs, labels_cat)


                # Calculate loss and backward
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
               
            
            

                #progress bar stuff
                progress_bar.set_description(f"Epoch {epoch+1}/{config.EVAL_EPOCHS}")
                progress_bar.set_postfix(loss=np.mean(losses))  # Update the loss value
                progress_bar.update(1)

                
            # end for batch 
            
            
        return modelEvaluation
    
    def evaluation(model, model_eval, dataloader):
        model_eval.eval()
        progress_bar = tqdm(total=len(dataloader), unit='step')
        total = 0
        correct = 0
        with torch.no_grad():
            for i, (audio,labels) in enumerate(dataloader):
                
                spectograms,audios = createModelInput(audio, augmentation=False)
                labels_cat = torch.cat([labels, labels], dim = 0).to(device)

                # Model's ouput two emb vectors
                with torch.cuda.amp.autocast():
                    # Use frozen encoder
                    _, _, frozen_audio, frozen_spects, _ = model(spectograms, audios)
                    inputs = torch.cat([frozen_audio, frozen_spects], dim = 0) 
                    outputs = model_eval(inputs)
                    _, predicated = torch.max(outputs.data, 1)
                    total += labels_cat.size(0)
                    
                    correct += (predicated == labels_cat).sum().item()
                    # finire di calcolare il max!                    
            
                #progress bar stuff
                progress_bar.set_description(f"Epoch {i+1}/{len(dataloader)}")
                progress_bar.update(1)

            # end for batch 
        
        return correct/total
    

    model_ = train(model, trainloader)
    accuracy_test = evaluation(model, model_, testloader)
    validation_accuracy = evaluation(model, model_, valloader)

    model.train()
    for param in model.parameters():
        param.requires_grad = True
    

    return accuracy_test, validation_accuracy


model_pipeline()