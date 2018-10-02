import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import sys
import time
import pickle

class CNN(nn.Module):    
    """
    
    Convolutional Neural Model used for training the models. The total number of kernels that will be used in this
    CNN is Co * len(Ks). 
    
    Args:
        weights_matrix: numpy.ndarray, the shape of this n-dimensional array must be (words, dims) were words is
        the number of words in the vocabulary and dims is the dimensionality of the word embeddings.
        Co: integer, stands for channels out and it is the number of kernels of the same size that will be used.
        Hu: integer, stands for number of hidden units in the hidden layer.
        C: integer, number of units in the last layer (number of classes)
        Ks: list, list of integers specifying the size of the kernels to be used. 
     
    """
    
    def __init__(self, vocab_size, emb_dim, Co, Hu, C, Ks, name = ''):
        
        super(CNN, self).__init__()
        
        self.num_embeddings = vocab_size
        
        self.embeddings_dim = emb_dim
        
        self.cnn_name = 'cnn_' + str(emb_dim) + str(Co) + '_' + str(Hu) + '_' + str(C) + '_' + str(Ks) + '_' + name
        
        self.Co = Co
        
        self.Hu = Hu
        
        self.C = C
        
        self.Ks = Ks
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embeddings_dim)       
                       
        self.convolutions = nn.ModuleList([nn.Conv2d(1,self.Co,(k, self.embeddings_dim)) for k in self.Ks])
            
        self.relu = nn.ReLU()
        
        self.linear1 = nn.Linear(self.Co * len(self.Ks), self.Hu)
        
        self.linear2 = nn.Linear(self.Hu, self.C)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        
        #size(N,1,length) to size(N,1,length,dims)
        
        x = self.embedding(x)
        
        #size(N,1,length,dims) to size(N,1,length)
        
        x = [self.relu(conv(x)).squeeze(3) for conv in self.convolutions]
        
        #size(N,1,length) to (N, Co * len(Ks))
        
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        
        x = torch.cat(x,1)
        
        #size(N, Co * len(Ks)) to size(N, Hu)
        
        x = self.linear1(x)
        
        x = self.relu(x)
        
        #size(N, Hu) to size(N, C)
        
        x = self.linear2(x)
        
        x = self.sigmoid(x)
        
        return x
    
    def load_pretrained_embeddings(self, weights_matrix):
                
        self.embedding = self.embedding.from_pretrained(torch.tensor(weights_matrix).float())
    
    def save_cnn_params(self):
        
        cnn_params = {'vocab_size': self.num_embeddings,'emb_dim': self.embeddings_dim , 'Co': self.Co, 'Hu': self.Hu, 'C': self.C, 'Ks': self.Ks}
        
        output_file = open(self.name + "_params.pkl", "wb")
        
        pickle.dump(cnn_params, output_file)

def train_cnn(model, train_dataloader, lr = 0.02, epochs_num = 100, momentum = 0.9):
    """
    
    This function trains a CNN model using gradient descent with the posibility of using momentum. 
    
    Args:
        model: cnn.CNN, an instance of a model of the class cnn.CNN 
        train_dataloader: Dataloader, Dataloader instance built using PrivacyPoliciesDataset instance
        lr: double, learning rate that we want to use in the learning algorithm
        epochs_num: integer, number of epochs
        momentum: double, momentum paramenter that tunes the momentum gradient descent algorithm
    
    Returns:
        epochs: list, list containing all the epochs
        losses: list, list containing the loss at the beginning of each epoch
    
    """

    optimizer = SGD(model.parameters(), lr = lr, momentum = momentum)

    criterion = nn.BCELoss()

    losses = []

    epochs = []

    start = time.time()

    remaining_time = 0

    for epoch in range(epochs_num):

        for i_batch, sample_batched in enumerate(train_dataloader):

            input = sample_batched[0]

            target = sample_batched[1].float()

            model.zero_grad()

            output = model(input)

            loss = criterion(output, target)

            loss.backward()

            optimizer.step()

        end = time.time()

        remaining_time = remaining_time * 0.90 + ((end - start) * (epochs_num - epoch + 1) / (epoch + 1)) * 0.1

        remaining_time_corrected = remaining_time / (1 - (0.9 ** (epoch + 1)))

        epoch_str = "last epoch finished: " + str(epoch)

        progress_str = "progress: " + str((epoch + 1) * 100 / epochs_num) + "%"

        time_str = "time: " + str(remaining_time_corrected / 60) + " mins"

        sys.stdout.write("\r" + epoch_str + " -- " + progress_str + " -- " + time_str)

        sys.stdout.flush()

        losses.append(loss.item())

        epochs.append(epoch)

    print("\n" + "Training completed. Total training time: " + str(round((end - start) / 60, 2)) + " mins")
    
    return epochs, losses
