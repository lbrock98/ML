#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 18:16:26 2022

@author: Lucy
"""

import numpy as np


def one_hot_encoder(encoded_text, num_uni_chars):
    
    # Create a placeholder for zeros.
    one_hot = np.zeros((encoded_text.size, num_uni_chars))
    
    # Convert data type for later use
    one_hot = one_hot.astype(np.float32)

    one_hot[np.arange(one_hot.shape[0]), encoded_text.flatten()] = 1.0
    
    # Reshape it so it matches the batch shape
    one_hot = one_hot.reshape((*encoded_text.shape, num_uni_chars))
    
    return one_hot

def generate_batches(encoded_text, samp_per_batch=10, seq_len=50):
    
    char_per_batch = samp_per_batch * seq_len
    
    num_batches_avail = int(len(encoded_text)/char_per_batch)
    
    # Cut off end of encoded_text that won't fit evenly 
    encoded_text = encoded_text[:num_batches_avail * char_per_batch]
    
    # Reshape text into rows the size of a batch
    encoded_text = encoded_text.reshape((samp_per_batch, -1))
    
    # Go through each row in array.
    for n in range(0, encoded_text.shape[1], seq_len):
        
        # Grab feature characters
        x = encoded_text[:, n:n+seq_len]
        
        # y is the target shifted over by 1
        y = np.zeros_like(x)
       
        try:
            y[:, :-1] = x[:, 1:]
            y[:, -1]  = encoded_text[:, n+seq_len]
            
        # FOR POTENTIAL INDEXING ERROR AT THE END    
        except:
            y[:, :-1] = x[:, 1:]
            y[:, -1] = encoded_text[:, 0]
            
        yield x, y
        
class CharModel(nn.Module):
    
    def __init__(self, all_chars, num_hidden=256, num_layers=4,drop_prob=0.5,use_gpu=False):
        
        
        # SET UP ATTRIBUTES
        super().__init__()
        self.drop_prob = drop_prob
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.use_gpu = use_gpu
        
        #CHARACTER SET, ENCODER, and DECODER
        self.all_chars = all_chars
        self.decoder = dict(enumerate(all_chars))
        self.encoder = {char: ind for ind,char in decoder.items()}
        
        
        self.lstm = nn.LSTM(len(self.all_chars), num_hidden, num_layers, dropout=drop_prob, batch_first=True)
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.fc_linear = nn.Linear(num_hidden, len(self.all_chars))
      
    
    def forward(self, x, hidden):
                  
        
        lstm_output, hidden = self.lstm(x, hidden)
        drop_output = self.dropout(lstm_output)
        drop_output = drop_output.contiguous().view(-1, self.num_hidden)
        final_out = self.fc_linear(drop_output)
        
        return final_out, hidden
    
    
    def hidden_state(self, batch_size):

        hidden = (torch.zeros(self.num_layers,batch_size,self.num_hidden),
                     torch.zeros(self.num_layers,batch_size,self.num_hidden))
        
        return hidden

def predict_next_char(model, char, hidden=None, k=1):
        
        # Encode raw letters with model
        encoded_text = model.encoder[char]
        encoded_text = np.array([[encoded_text]])
        encoded_text = one_hot_encoder(encoded_text, len(model.all_chars))
        # Convert to Tensor
        inputs = torch.from_numpy(encoded_text)
        # Grab hidden states
        hidden = tuple([state.data for state in hidden])
        # Run model and get predicted output
        lstm_out, hidden = model(inputs, hidden)

        # Convert lstm_out to probabilities
        probs = F.softmax(lstm_out, dim=1).data

        if(model.use_gpu):
            # move back to CPU to use with numpy
            probs = probs.cpu()
            
        # Return k largest probabilities in tensor
        probs, index_positions = probs.topk(k)
        index_positions = index_positions.numpy().squeeze()
        # Create array of probabilities
        probs = probs.numpy().flatten()
        # Convert to probabilities per index
        probs = probs/probs.sum()
        # randomly choose a character based on probabilities
        char = np.random.choice(index_positions, p=probs)
        # return the encoded value of the predicted char and the hidden state
        return model.decoder[char], hidden
    

def generate_text(model, size, seed='The', k=1):
    model.cpu()
    model.eval()
    
    # begin output from initial seed
    output_chars = [c for c in seed]
    # intiate hidden state
    hidden = model.hidden_state(1)
    
    # predict the next character for every character in seed
    for char in seed:
        char, hidden = predict_next_char(model, char, hidden, k=k)
    
    # add initial characters to output
    output_chars.append(char)
    
    # Now generate for size requested
    for i in range(size):
        # predict based off very last letter in output_chars
        char, hidden = predict_next_char(model, output_chars[-1], hidden, k=k)
        # add predicted character
        output_chars.append(char)
    
    return ''.join(output_chars)    

        