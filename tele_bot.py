from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import apiai, json
import requests as req
import wit_ai
updater = Updater(token='1041190715:AAFAP8UnMiQznA2ea1GG2iAg9laYiY9yb8E')
dispatcher = updater.dispatcher


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
import matplotlib.pyplot as plt
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

"""
GPT-2
"""

import gpt_2_simple as gpt2
import tensorflow as tf

"""
Bot
"""

def startCommand(bot, update):
	bot.send_message(chat_id=update.message.chat_id, text='Привет, давай пообщаемся?')

def textMessage2(bot, update):
    input_string = '\n\n' + update.message.text.strip() + '\n'
    # TODO: вынести наружу
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess)
    output_string = gpt2.generate(sess, return_as_list=True, prefix=input_string)[0]
    output_string = output_string[len(input_string):]
    output_string = re.sub('\n.*', '', output_string)
    bot.send_message(chat_id=update.message.chat_id, text=output_string)

def textMessage(bot, update):
    input_string = '\n\n' + update.message.text.strip() + '\n'
    arguments = {
        'file': "LSTM_4000.model",
        'start_of_sequence': input_string,
        'size_of_generated_sequence': 500,
        'temperature': 0.5,
        'cuda': 0
    }
    # TODO: вынести наружу
    char_model = torch.load(arguments.get('file'), map_location='cpu')
    detransliterated_text = detransliterate(sampling(char_model, string.printable, arguments))
    detransliterated_text = detransliterated_text[len(input_string):]
    detransliterated_text = re.sub('\d+', ' ', detransliterated_text)
    detransliterated_text = re.sub(' +', ' ', detransliterated_text)
    detransliterated_text = '\n'.join([sent.strip() for sent in detransliterated_text.split('\n')])
    detransliterated_text = re.sub('^\n+', '', detransliterated_text)
    detransliterated_text = re.sub('\n.*', '', detransliterated_text)
    output_string = detransliterated_text.strip()
    bot.send_message(chat_id=update.message.chat_id, text=output_string)

def text1MMessage(bot, update):
	print(update)
	request = apiai.ApiAI('cf5966ecf7894ea6ab4fa5f6955bdc54').text_request
	request.lang = 'ru'
	request.session_id = 'dhhack_bot'
	request.query = update.message.text 
	responseJson = json.loads(request.getresponse().read().decode('utf-8'))
	response = responseJson['result']['fulfillment']['speech']
	if response:
		bot.send_message(chat_id=update.message.chat_id, text=response)
	else:
		bot.send_message(chat_id=update.message.chat_id, text='Я Вас не совсем понял!')

def extract_audio(file_type):
	import subprocess
	command = "ffmpeg -i video.mp4 -ab 160k -ac 2 -ar 44100 -vn audio.wav"
	subprocess.call(command, shell=True)
	print('all good!')

def videoMessage(bot, update):
	type_file = update.message.video.mime_type.split('/')[-1]
	file_info = bot.get_file(update.message.video.file_id)
	video = req.get(file_info.file_path)
	with open('video.' + str(type_file), 'wb') as f:
		try:
			f.write(video.content)
		except Exception as e:
			print(e)
	with open('video.wav', 'wb') as f:
		try:
			f.write(video.content)
		except Exception as e:
			print(e)
	bot.send_message(chat_id=update.message.chat_id, text=f'это видеозапись {type_file} формата')
	extract_audio(type_file)

def audioMessage(bot, update):
	type_file = update.message.audio.mime_type.split('/')[-1]
	# type_file = 'mp3' if type_file == 'mpeg' else type_file
	file_info = bot.get_file(update.message.audio.file_id)
	audio = req.get(file_info.file_path)
	print(audio)
	print(type_file)
	try:
		with open('audio.wav', 'wb') as f:
			try:
				f.write(audio.content)
			except Exception as e:
				print(e)
	except Exception as e:
		print(e)
	words_in_wav = wit_ai.read_voice('audio.wav')
	bot.send_message(chat_id=update.message.chat_id, text=f'{words_in_wav}')

def voiceMessage(bot, update):
	type_file = update.message.voice.mime_type.split('/')[-1]
	file_info = bot.get_file(update.message.voice.file_id)
	audio = req.get(file_info.file_path)
	with open('voice.mp3', 'wb') as f:
		try:
			f.write(audio.content)
		except Exception as e:
			print(e)
	# try:
		# words_in_wav = wit_ai.read_voice('voice.wav')
		# bot.send_message(chat_id=update.message.chat_id, text=f'{words_in_wav}')
	# except Exception as e:
		# print(e)
	# bot.send_message(chat_id=update.message.chat_id, text=f'это голосовое сообщение {type_file} формата')

start_command_handler = CommandHandler('start', startCommand)
text_message_handler = MessageHandler(Filters.text, textMessage)
audio_message_handler = MessageHandler(Filters.audio, audioMessage)
voice_message_handler = MessageHandler(Filters.voice, voiceMessage)
video_message_handler = MessageHandler(Filters.video, videoMessage)
dispatcher.add_handler(start_command_handler)
dispatcher.add_handler(audio_message_handler)
dispatcher.add_handler(voice_message_handler)
dispatcher.add_handler(text_message_handler)
dispatcher.add_handler(video_message_handler)
updater.start_polling(clean=True)
updater.idle()
