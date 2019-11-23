from wit import Wit

def read_voice(file_name):
	client = Wit(access_token='WWFXLDLV5232NULJ4P7PXO2RUSMLMDUT')
	resp = None
	with open('voice.wav', 'rb') as f:
	  resp = client.speech(f, None, {'Content-Type': 'audio/wav'})
	#   print(resp)
	#   client.speech(resp)
	return(str(resp))

print(read_voice('voice.wav'))
