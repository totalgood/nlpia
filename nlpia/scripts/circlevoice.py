#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from traceback import format_exc

import speech_recognition as sr
import pyaudio
import wave
from io import BytesIO
import pyttsx3

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Record an audio clip.')
    parser.add_argument('-n', '--num', '--num_recordings', type=int, default=0, dest='num_recordings',
                        help='Record N audio clips.')
    parser.add_argument('-r', '--record', '--recordpath', type=str, default='', dest='recordpath',
                        help='Record N audio clips (delimitted by silence).')

    parser.add_argument('-b', '--begin', '--begining', type=int, default=0, dest='begin',
                        help='Sample number to begin clip')
    parser.add_argument('-e', '--end', type=int, default=-1, dest='end',
                        help='Sample number to end clip')
    parser.add_argument('-p', '--play', type=str, default='', dest='playpath',
                        help='Path of audio file to play.')
    args = parser.parse_args()
    return args


def record_audio(source='Microphone'):
    r = sr.Recognizer()
    audio = r.listen(sr.Microphone)
    return audio


def play_audio(audio, start=0, stop=None, save=None, batch_size=1024):
    player = pyaudio.PyAudio()
    if isinstance(audio, str):
        input_stream = wave.openfp(open(audio, 'rb'))
    else:
        input_stream = wave.openfp(BytesIO(audio.get_wav_data()))

    output_stream = player.open(
        format=player.get_format_from_width(input_stream.getsampwidth()),
        channels=input_stream.getnchannels(),
        rate=input_stream.getframerate(),
        output=True)
    # play stream
    batch = input_stream.readframes(batch_size)
    save = open(save, 'wb') if save is not None else save
    i = 0
    while batch and (i + 1) * batch_size >= start and (stop is None or (i - 1) * batch_size < stop):
        if stop is not None and i * batch_size >= stop:
            batch = batch[:(stop % batch_size)]
        if start and i * batch_size < start:
            batch = batch[(start % batch_size):]
        output_stream.write(batch)
        if save is not None:
            save.write(batch)
        batch = input_stream.readframes(batch_size)
    return audio


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


def save_audio(audio, path='audio.wav'):
    data = getattr(audio, 'get_{}_data'.format(path.lower().split('.')[-1].strip()), 'get_wav_data')()
    with open(path, 'wb') as fout:
        fout.write(data)
    return path


def try_tts(audio):
    text = []
    try:
        text.append(stt(audio, api='sphinx'))
    except sr.RequestError:
        try:
            text = text.append(stt(audio, api='google'))
        except Exception:
            print(format_exc())
    return text


def main_full_circle():
    args = parse_args()
    i = 0
    while True:
        print("Say something! I'm listening...")
        audio = record_audio()
        if args.num_recordings > i:
            i += 1
            save_audio(audio, 'record{}.wav'.format(i))
        else:
            break

    print('This is what your microphone picked up...')
    play_audio(audio)

    text = try_tts(audio)

    print("This is what I understood:\n{}\n".format(text))

    print("And this is what I sound like saying that...")


if __name__ == '__main__':
    main_full_circle()
