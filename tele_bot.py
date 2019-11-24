from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import apiai, json
import requests as req
import wit_ai
updater = Updater(token='1041190715:AAFAP8UnMiQznA2ea1GG2iAg9laYiY9yb8E')
dispatcher = updater.dispatcher

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


def textMessage(bot, update):
    try:
        input_string = '\n\n' + update.message.text.strip() + '\n'
        print('input_string:', input_string)
        # TODO: вынести наружу
        tf.reset_default_graph()
        sess = gpt2.start_tf_sess()
        gpt2.load_gpt2(sess)
        output_string = gpt2.generate(sess, return_as_list=True, prefix=input_string)[0]
        output_string = output_string[len(input_string):]
        output_string = re.sub('\n.*', '', output_string)
        print('output_string:', output_string)
        bot.send_message(chat_id=update.message.chat_id, text=output_string)
    except Exception as e:
        print("ERROR2:", e)

"""
def textMessage3(bot, update):
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
"""

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
