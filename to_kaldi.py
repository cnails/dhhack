import subprocess

def run_kaldi(decoder, voice):
    print ("START KALDI")
    subprocess.call(decoder + " " +  voice + "| grep 'voice' | cut -d ' ' -f 3-4 > timecodes.txt", shell=True)
    print ("END KALDI")

run_kaldi("./decode.sh", "voice.vaw")
