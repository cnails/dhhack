"""
LSTM sampling
"""


import requests
from collections import defaultdict
import torch
print('torch version', torch.__version__)
from torch import nn, zeros, cat
from torch.autograd import Variable
import argparse
import string
import numpy as np
import re
import random
from itertools import cycle
import json
import warnings
warnings.filterwarnings("ignore")

from_list = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
to_list = 'a,b,v,g,d,e,jo,zh,z,i,y,k,l,m,n,o,p,r,s,t,u,f,kh,c,ch,sh,shh,jhh,ih,jh,eh,ju,ja,A,B,V,G,D,E,JO,ZH,Z,I,Y,K,L,M,N,O,P,R,S,T,U,F,KH,C,CH,SH,SHH,JHH,IH,JH,EH,JU,JA'.split(',')

def detransliterate(string):
    detransliterated_string = ''
    index_of_symbol = 0
    while index_of_symbol < len(string):
        was_combination = 0
        for i in range(min(3, len(string)-index_of_symbol), 0, -1):
            if string[index_of_symbol:index_of_symbol+i] in to_list:
                index = to_list.index(string[index_of_symbol:index_of_symbol+i])
                detransliterated_string += from_list[index]
                index_of_symbol += i
                was_combination = 1
                break
        if not was_combination:
            detransliterated_string += string[index_of_symbol]
            index_of_symbol += 1
    return re.findall('[^a-zA-Z]+', detransliterated_string)[0]

# Turning a string into a tensor
def string_to_tensor(string, symbols):
    tensor = torch.zeros(len(string)).to(torch.int64)
    for index in range(len(string)):
        try:
            tensor[index] = symbols.index(string[index])
        except ValueError:
            continue
    return Variable(tensor)

class CharModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size=1, type_of_model=0,
                 num_layers=2, bias=True, dropout=0, bidirectional=False, output_bias=True):
        super().__init__()        
        # initializing attributes
        # the number of expected features in the input
        self.input_size = input_size

        # the number of features in the hidden state
        self.hidden_size = hidden_size

        # the number of features in the output
        self.output_size = output_size

        # set batch_size
        self.batch_size = batch_size

        # type of model (LSTM/LSTMCell/GRU)
        self.type_of_model = type_of_model

        # stacked LSTM/GRU or not? if yes, than how many layers?
        self.num_layers = num_layers
        #if self.type_of_model == 1: # LSTMCell
        #    self.num_layers = 1

        self.bias = bias

        # dropout parameter for LSTM and GRU
        self.dropout = dropout

        # bidirectional LSTM or GRU?
        self.bidirectional = bidirectional

        # bias in Linear layer
        self.output_bias = output_bias

        # define metrics
        self.metrics = nn.CrossEntropyLoss()

        # set optimizer to None
        self.optimizer = None

        # define embedding for input
        self.encoder = nn.Embedding(num_embeddings=self.input_size, embedding_dim=self.hidden_size)

        # define model (LSTM/LSTMCell/GRU)
        if self.type_of_model == 0:
            self.model = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, \
                                 num_layers=self.num_layers, bias=self.bias, \
                                 dropout=self.dropout, bidirectional=self.bidirectional)
        elif self.type_of_model == 1:
            self.model = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size, \
                                     bias=self.bias)
        elif self.type_of_model == 2:
            self.model = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, \
                                num_layers=self.num_layers, bias=self.bias, \
                                dropout=self.dropout, bidirectional=self.bidirectional)
        else:
            raise ValueError("Unknown type of model's number")
        if self.bidirectional:
            self.decoder = nn.Linear(2 * self.hidden_size, self.output_size, bias=output_bias)
        else:
            self.decoder = nn.Linear(self.hidden_size, self.output_size, bias=output_bias)

    def forward(self, input_vector, hidden_state):
        batch_size = input_vector.shape[0]
        encoded_vector = self.encoder(input_vector)

        # time dimension X batch dimension X feature dimension
        # timesteps X batch_size X features
        # one batch per epoch => first dimension: 1

        output_vector, hidden_state = self.model(encoded_vector.view(1, batch_size, -1), hidden_state)

        # bidir lstm
        if self.bidirectional:
            forward_output_vector, backward_output_vector = \
                output_vector[:1, :, :self.hidden_size], output_vector[0:, :, self.hidden_size:]
            output_vector = cat((forward_output_vector, backward_output_vector), dim=-1)

        # view is an analogue of numpy reshape
        output_vector = self.decoder(output_vector.view(batch_size, -1))
        return output_vector, hidden_state

    def hidden_state_initialization(self):
        # GPU
        if self.type_of_model == 2:
            # return only hidden state's vector
            if self.bidirectional:
                return cat((Variable(zeros(self.num_layers, self.batch_size, self.hidden_size)),
                            Variable(zeros(self.num_layers, self.batch_size, self.hidden_size))))
            else:
                return Variable(zeros(self.num_layers, self.batch_size, self.hidden_size))
        # LSTM and LSTMCell
        # return hidden state and cell's vectors
        if self.bidirectional:
            return (cat((Variable(zeros(self.num_layers, self.batch_size, self.hidden_size)),
                         Variable(zeros(self.num_layers, self.batch_size, self.hidden_size)))),
                    cat((Variable(zeros(self.num_layers, self.batch_size, self.hidden_size)),
                         Variable(zeros(self.num_layers, self.batch_size, self.hidden_size)))))
        else:
            return (Variable(zeros(self.num_layers, self.batch_size, self.hidden_size)),
                    Variable(zeros(self.num_layers, self.batch_size, self.hidden_size)))


def count_loss(char_model, input_symbols, correct_output_symbols):
    hidden_state = char_model.hidden_state_initialization()

    if arguments.get('cuda'):
        if arguments.get('type_of_model') == 0:
            hidden_state = (hidden_state[0].cuda(), hidden_state[1].cuda())
        else:
            hidden_state = hidden_state.cuda()

    # We need to zero out the gradients because 
    # we don't want to accumulate them on subsequent backward passes
    char_model.zero_grad()

    loss = 0
    for index in range(arguments.get('size_of_learned_sequence')):
        # forward pass
        output_vector, hidden_state = char_model.forward(input_symbols[:, index], hidden_state)
        loss += char_model.metrics(output_vector.view(arguments.get('batch_size'), -1), \
                                   correct_output_symbols[:, index])
    # backward pass
    loss.backward()

    # optimize parameters
    char_model.optimizer.step()

    # return mean loss
    return loss.item() / arguments.get('size_of_learned_sequence')

def sampling(char_model, symbols, arguments):
    # save previous value
    tmp = char_model.batch_size

    char_model.batch_size = 1
    hidden_state = char_model.hidden_state_initialization()
    char_model.zero_grad()

    # set back previous value
    char_model.batch_size = tmp

    # insert a singleton dimension at the first place and wrap by Variable
    input_symbols = Variable(string_to_tensor(arguments.get('start_of_sequence'), symbols).unsqueeze(0))

    if arguments.get('cuda'):
        if char_model.type_of_model == 0:
            hidden_state = (hidden_state[0].cuda(), hidden_state[1].cuda())
        else:
            hidden_state = hidden_state.cuda()
        input_symbols = input_symbols.cuda()

    for index in range(len(arguments.get('start_of_sequence')) - 1):
        # forward pass
        output_vector, hidden_state = char_model.forward(input_symbols[:, index], hidden_state)

    input_symbols = input_symbols[:, -1]
    generated_text = arguments.get('start_of_sequence')

    for _ in range(arguments.get('size_of_generated_sequence')):
        # forward pass
        output_vector, hidden_state = char_model.forward(input_symbols, hidden_state)

        # divide by temperature so that to get more conservative or more arbitrary samples
        # the lower the temperature, the more conservative output
        # the higher is temperature, the more random output
        # scaling logits before applying softmax

        # we add 0.1 in order to avoid INF
        bias = 0.1

        try:
            # output_vector = torch.softmax(output_vector.div(arguments.get('temperature')), 0)
            # output_vector = torch.exp(output_vector / arguments.get('temperature')).view(-1)
            output_vector = output_vector.data.view(-1).div(arguments.get('temperature')).exp()
        except RuntimeError:
            output_vector = torch.exp(output_vector / (arguments.get('temperature') + bias)).view(-1)

        # The rows of input do not need to sum to one (in which case we use the values as weights)
        # and it doesn't in this case
        index_of_symbol = torch.multinomial(input=output_vector, num_samples=1)[0]

        # print(index_of_symbol)
        # also possible via softmax and argmax prediction but terrible quality
        # probs = torch.softmax(output_vector, 0)
        # index_of_symbol = torch.max(probs, 0)[1].item()

        index_of_symbol = index_of_symbol.data
        if index_of_symbol >= 0 and index_of_symbol < len(symbols):
            generated_text += symbols[index_of_symbol]
        else:
            # in case of unprintable symbol
            generated_text += symbols[np.random.randint(0, len(symbols))] # right border not included
        # generated_symbol
        input_symbols = Variable(string_to_tensor(generated_text[-1], symbols).unsqueeze(0))

        if arguments.get('cuda'):
            input_symbols = input_symbols.cuda()
        # stop when \n is encountered
        # if generated_text[-2:] == '\n\n':
        #     break

    return generated_text
