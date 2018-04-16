#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Round trip STT -> TTS demo using pocketSphinx, deepspeech, speech_recognition, and pyttsx3 """
from __future__ import print_function, unicode_literals, division, absolute_import
from builtins import (bytes, dict, int, list, object, range, str,  # noqa
    ascii, chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)
from future import standard_library
standard_library.install_aliases()  # noqa: Counter, OrderedDict,

from timeit import default_timer as timer

import speech_recognition as sr
# import pyaudio
# import wave
# from io import BytesIO
import pyttsx3

import simpleaudio as sa
import argparse
import sys
import scipy.io.wavfile as wav

from deepspeech.model import Model


def record_audio(source='Microphone'):
    r = sr.Recognizer()
    with getattr(sr, source, sr.Microphone)() as source:  # use the default microphone as the audio source
        audio = r.listen(source)     # listen for the first phrase and extract it into audio data
    return audio


def play_audio(audio, start=0, stop=None, save=None, batch_size=1024):
    if isinstance(audio, str):
        # wave_stream = wave.openfp(open(audio, 'rb'))
        wave_obj = sa.WaveObject.from_wave_file(audio)
    else:
        wave_obj = sa.WaveObject.from_wave_read(audio)
    play_obj = wave_obj.play()
    play_obj.wait_done()
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
    voice = voice or engine.getProperty('voice').split('.')[-1]
    if voice in voice_names:
        engine.setProperty('voice', voices[voice_names.index(voice)])
    else:
        voice = engine.getProperty('voice').split('.')[-1]
        print("WARN: Voice name '{}' not found.\n  Valid voice names: {}".format(voice, ' '.join(voices)))
        print("Using default voice named '{}'.".format(voice))
    engine.say(text)
    engine.runAndWait()
    return text


def save_audio(audio, path='audio.wav'):
    data = getattr(audio, 'get_{}_data'.format(path.lower().split('.')[-1].strip()), 'get_wav_data')()
    with open(path, 'wb') as fout:
        fout.write(data)
    return path


# These constants control the beam search decoder

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_WEIGHT = 1.75

# The beta hyperparameter of the CTC decoder. Word insertion weight (penalty)
WORD_COUNT_WEIGHT = 1.00

# Valid word insertion weight. This is used to lessen the word insertion penalty
# when the inserted word is part of the vocabulary
VALID_WORD_COUNT_WEIGHT = 1.00

# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9


def deep_parse_args():
    parser = argparse.ArgumentParser(description='Benchmarking tooling for DeepSpeech native_client.')
    parser.add_argument('model', type=str,
                        help='Path to the model (protocol buffer binary file)')
    parser.add_argument('audio', type=str,
                        help='Path to the audio file to run (WAV format)')
    parser.add_argument('alphabet', type=str,
                        help='Path to the configuration file specifying the alphabet used by the network')
    parser.add_argument('lm', type=str, nargs='?',
                        help='Path to the language model binary file')
    parser.add_argument('trie', type=str, nargs='?',
                        help='Path to the language model trie file created with native_client/generate_trie')
    parser.add_argument('-r', '--record', type=int, default=1, dest='num_recordings',
                        help='Just record audio clips and exit after N clips')

    args = parser.parse_args()
    return args


def parse_args():
    parser = argparse.ArgumentParser(description='Record an audio clip.')
    parser.add_argument('-r', '--record', type=int, default=1, dest='num_recordings',
                        help='Just record audio clips and exit after N clips')
    args = parser.parse_args()
    return args


def deepspeech_main(args):
    print('Loading model from file %s' % (args.model), file=sys.stderr)
    model_load_start = timer()
    ds = Model(args.model, N_FEATURES, N_CONTEXT, args.alphabet, BEAM_WIDTH)
    model_load_end = timer() - model_load_start
    print('Loaded model in %0.3fs.' % (model_load_end), file=sys.stderr)

    if args.lm and args.trie:
        print('Loading language model from files %s %s' % (args.lm, args.trie), file=sys.stderr)
        lm_load_start = timer()
        ds.enableDecoderWithLM(args.alphabet, args.lm, args.trie, LM_WEIGHT,
                               WORD_COUNT_WEIGHT, VALID_WORD_COUNT_WEIGHT)
        lm_load_end = timer() - lm_load_start
        print('Loaded language model in %0.3fs.' % (lm_load_end), file=sys.stderr)

    fs, audio = wav.read(args.audio)
    # We can assume 16kHz
    audio_length = len(audio) * (1 / 16000)
    assert fs == 16000, "Only 16000Hz input WAV files are supported for now!"

    print('Running inference.', file=sys.stderr)
    inference_start = timer()
    print(ds.stt(audio, fs))
    inference_end = timer() - inference_start
    print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)


def main():
    args = parse_args()
    i = 0
    while True:
        print("Say something! I'm listening...")
        audio = record_audio()
        if args.num_recordings <= 1:
            break
        if args.num_recordings > i:
            save_audio(audio, 'record{}.wav'.format(i))
            i += 1
            continue
        break

    print('This is what your microphone picked up...')
    play_audio(audio)

    text = []
    try:
        text.append(stt(audio, api='sphinx'))
    except sr.RequestError:
        text.append(stt(audio, api='google'))

    print("This is what I understood: {}".format(text))

    print("And this is what I sound like saying that...")
    print('Used the voice named {}'.format(tts('.  '.join(text))))


if __name__ == '__main__':
    main()
