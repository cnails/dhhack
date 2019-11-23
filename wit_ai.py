from wit import Wit

def read_voice(voice):
	client = Wit(access_token='WWFXLDLV5232NULJ4P7PXO2RUSMLMDUT')
	resp = None
	with open('test.mp3', 'rb') as f:
	  resp = client.speech(f, None, {'Content-Type': 'audio/mpeg3'})
	  client.speech(d)
	reutrn(str(resp))
