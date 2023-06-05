import matplotlib.pyplot as plt
from torchaudio.datasets import SPEECHCOMMANDS
import os
import torch
import torchaudio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

#labels of the dataset, (35)
labels =  ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow','forward','four',
           'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on','one', 'right',
           'seven', 'sheila', 'six', 'stop', 'three','tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))

def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number
    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    #
    #   INSERT AUGMENTATION HERE!
    #

    targets = torch.stack(targets)
    return tensors, targets

def getData(batch_size = 32):


    train_set = SubsetSC("training")
    test_set = SubsetSC("testing")
    val_set = SubsetSC("validation")
    
    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False


    # creating Dataloaders
    train_loader = torch.utils.data.DataLoader( train_set, batch_size=batch_size,shuffle=True,collate_fn=collate_fn, num_workers=num_workers,pin_memory=pin_memory)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=batch_size,shuffle=True,collate_fn=collate_fn,num_workers=num_workers,pin_memory=pin_memory,)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False,drop_last=False,collate_fn=collate_fn,num_workers=num_workers,pin_memory=pin_memory,)
    
    return train_loader, test_loader, val_loader



from augmentation import *
if __name__ == "__main__":

    #testing dataloaders!
    train_loader,test_loader, val_loader  = getData()
    audios, labels = next(iter(train_loader))
    #spectogram = torchaudio.transforms.MelSpectrogram()

    #a = spectogram(audios[0])
    plt.plot(audios[0].T)
    a = noise_injection(audios[0])
    plt.plot(a.T.detach().numpy()) 
    plt.show()