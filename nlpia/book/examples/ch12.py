import speech_recognition as sr
import pyaudio
import wave
from io import BytesIO
import pyttsx3


def record_audio(source='Microphone'):
    r = sr.Recognizer()
    with getattr(sr, source, sr.Microphone)() as source:  # use the default microphone as the audio source
        audio = r.listen(source)     # listen for the first phrase and extract it into audio data
    return audio


def play_audio(audio):
    p = pyaudio.PyAudio()
    f = wave.openfp(BytesIO(audio.get_wav_data()))

    stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                    channels=f.getnchannels(),
                    rate=f.getframerate(),
                    output=True)
    # play stream
    data = f.readframes(1024)
    while data:
        stream.write(data)
        data = f.readframes(1024)


def stt(audio, api='google'):
    r = sr.Recognizer()
    text = getattr(r, 'recognize_{}'.format(api), 'recognize_google')(audio)
    try:
        print("This is what {} thinks you said: ".format(api=api) + text)
    except LookupError:                            # speech is unintelligible
        print("ERROR: {} couldn't understand that.".format(api))
    return text


def tts(text, rate=200, voice='Alex'):
    engine = pyttsx3.init()
    voices = [v.id for v in engine.getProperty('voices')]
    voice_names = [v.split('.')[-1] for v in voices]
    if rate:
        engine.setProperty('rate', int(rate))
    if voice in voice_names:
        engine.setProperty('voice', voices[voice_names.index(voice)])
    else:
        voice = engine.getProperty('voice').split('.')[-1]
        print("WARN: Voice name '{}' not found.\n  Valid voice names: {}".format(voice, ' '.join(voices)))
        print("Using default voice named '{}'.".format(voice))
    engine.say(text)
    engine.runAndWait()
    return voice


if __name__ == '__main__':
    print('Listening...')
    audio = record_audio()

    print('This is what your microphone picked up...')
    play_audio(audio)

    text = stt(audio)

    print('And this is what I sound like saying that...')
    print('Used the voice named {}'.format(tts(text)))
