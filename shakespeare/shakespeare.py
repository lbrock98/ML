#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 18:13:53 2022

@author: Lucy
"""

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from helper import one_hot_encoder, generate_batches, CharModel, generate_text

# import data
with open('shakespeare.txt','r',encoding='utf8') as f:
    text = f.read()
    
# encode text    
all_characters = set(text) #gets unique characters

decoder = dict(enumerate(all_characters))
encoder = {char: ind for ind,char in decoder.items()}
encoded_text = np.array([encoder[char] for char in text])
    
one_hot_encoder(np.array([1,2,0]),3)

# instantiate model
model = CharModel(
    all_chars=all_characters,
    num_hidden=512,
    num_layers=3,
    drop_prob=0.5,
    use_gpu=False,
)

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()
train_percent = 0.9
train_ind = int(len(encoded_text) * (train_percent))
train_data = encoded_text[:train_ind]
val_data = encoded_text[train_ind:]

# train network
epochs = 50
batch_size = 128
seq_len = 100 
tracker = 0
num_char = max(encoded_text)+1

model.train()
for i in range(epochs):
    
    hidden = model.hidden_state(batch_size)
    for x,y in generate_batches(train_data,batch_size,seq_len):
        
        tracker += 1
        # One Hot Encode incoming data
        x = one_hot_encoder(x,num_char)
        inputs = torch.from_numpy(x)
        targets = torch.from_numpy(y)
       
        # Reset Hidden State
        hidden = tuple([state.data for state in hidden])
        
        model.zero_grad()
        
        lstm_output, hidden = model.forward(inputs,hidden)
        loss = criterion(lstm_output,targets.view(batch_size*seq_len).long())
        
        loss.backward()
        
        # clip
        nn.utils.clip_grad_norm_(model.parameters(),max_norm=5)
        
        optimizer.step()

        
        ###################################
        ### CHECK ON VALIDATION SET ######
        #################################
        
        if tracker % 25 == 0:
            
            val_hidden = model.hidden_state(batch_size)
            val_losses = []
            model.eval()
            
            for x,y in generate_batches(val_data,batch_size,seq_len):
                
                # One Hot Encode incoming data
                x = one_hot_encoder(x,num_char)
                inputs = torch.from_numpy(x)
                targets = torch.from_numpy(y)

                # Reset Hidden State
                val_hidden = tuple([state.data for state in val_hidden])
                
                lstm_output, val_hidden = model.forward(inputs,val_hidden)
                val_loss = criterion(lstm_output,targets.view(batch_size*seq_len).long())
        
                val_losses.append(val_loss.item())
            
            # Reset to training model after val for loop
            model.train()
            
            print(f"Epoch: {i} Step: {tracker} Val Loss: {val_loss.item()}")
            
torch.save(model.state_dict(),'example.net')

# load model
model = CharModel(
    all_chars=all_characters,
    num_hidden=512,
    num_layers=3,
    drop_prob=0.5,
    use_gpu=True,
)
model.load_state_dict(torch.load('example.net'))
model.eval()

# print predictions
print(generate_text(model, 1000, seed='The ', k=3))
