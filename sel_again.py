import requests as req, wit_ai

def write_voice(words):
	num = len(words)
	for n in range(num):
		a = req.get("https://tts.voicetech.yandex.net/generate?key=22fe10e2-aa2f-4a58-a934-54f2c1c4d908&text=" + words[n] + "&format=wav&lang=ru-RU&speed=1&emotion=neutral&speaker=alyss&robot=1")
		with open('wav'+ str(n) +'.wav', 'wb') as fd:
			fd.write(a.content)
