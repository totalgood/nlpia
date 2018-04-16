#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division, absolute_import
from builtins import (bytes, dict, int, list, object, range, str,  # noqa
    ascii, chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)
from future import standard_library
standard_library.install_aliases()  # noqa: Counter, OrderedDict,

from timeit import default_timer as timer
import os
import sys
from traceback import format_exc
import argparse
# from io import BytesIO


# import pyaudio  # only works with python2.7?
import simpleaudio as sa
import wave
import speech_recognition as sr
import pyttsx3
from deepspeech.model import Model


def record_audio(source='Microphone', energy_threshold=300, pause_threshold=.9,
                 dynamic_energy_ratio=1.5, dynamic_energy_adjustment_damping=.15, **kwargs):
    """ Listten for a single utterance (concluded with a 2 sec pause) and return a recording in an Audio object

    Arguments:
        energy_threshold (int): minimum audio energy to trigger start of recording (default=300)
        dynamic_energy_adjustment_damping (float): dyn thresh adjustment slowness: 1=static energy_threshold (.15)
        dynamic_energy_ratio (float): sound energy change that triggers recording: 1=static energy_threshold (1.5)
        pause_threshold (float): non-speaking audio seconds before a phrase is considered complete (.9)
        operation_timeout (float): internal operation timeout in seconds: None=never
        self.phrase_threshold (float): minimum speaking duration in seconds to record (.3)
        self.non_speaking_duration (float): nonspeaking audio seconds to retain before+after recording (pause_threshold)
 """
    r = sr.Recognizer()
    r.energy_threshold = energy_threshold
    r.pause_threshold = pause_threshold
    r.dynamic_energy_threshold = dynamic_energy_ratio > 1 and dynamic_energy_adjustment_damping < 1
    r.dynamic_energy_ratio = dynamic_energy_ratio
    r.dynamic_energy_adjustment_damping = dynamic_energy_adjustment_damping
    r.__dict__.update(kwargs)
    with getattr(sr, 'Microphone', sr.Microphone)() as audio_source:
        audio = r.listen(source=audio_source)
    return audio


# def play_pyaudio(audio, start=0, stop=None, save=None, batch_size=1024):
#     player = pyaudio.PyAudio()
#     if isinstance(audio, str):
#         input_stream = wave.openfp(open(audio, 'rb'))
#     else:
#         input_stream = wave.openfp(BytesIO(audio.get_wav_data()))

#     output_stream = player.open(
#         format=player.get_format_from_width(input_stream.getsampwidth()),
#         channels=input_stream.getnchannels(),
#         rate=input_stream.getframerate(),
#         output=True)
#     # play stream
#     batch = input_stream.readframes(batch_size)
#     save = open(save, 'wb') if save is not None else save
#     i = 0
#     while batch and (i + 1) * batch_size >= start and (stop is None or (i - 1) * batch_size < stop):
#         if stop is not None and i * batch_size >= stop:
#             batch = batch[:(stop % batch_size)]
#         if start and i * batch_size < start:
#             batch = batch[(start % batch_size):]
#         output_stream.write(batch)
#         if save is not None:
#             save.write(batch)
#         batch = input_stream.readframes(batch_size)
#     return audio


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


def parse_args_deep():
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


def main_deepspeech(args):
    args = parse_args_deep() if args is None else args
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

    fs, audio = wave.read(args.audio)
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
    i, j = 0, 0
    for i in range(args.num_recordings):
        print("Say something! I'm listening...")
        audio = record_audio()
        base_dir = '.'
        ext = '.wav'
        filepath = os.path.join(base_dir, 'audio-{}{}'.format(i, ext))
        while os.path.exists(filepath):
            j += 1
            filepath = os.path.join(base_dir, 'audio-{}{}'.format(j + i, ext))
            print("filepath '{}'".format(filepath))
        save_audio(audio, filepath)
        print("Saved audio clip to '{}'".format(filepath))
        print("Playing audio clip ...")
        play_audio(filepath)
        print("Done.")

    if args.playpath:
        print("Playing {}[{}:{}]".format(args.playpath, args.begin, args.end))
        play_audio(args.playpath)


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
    # voice_name = tts(text)
    # print('I used the voice named {}'.format(voice_name))
    # 2018-02-01 10:10:58.435 python[3863:89560] -[OC_PythonArray length]:
    #     unrecognized selector sent to instance 0x7feeee691740
    # ^[[A^C^C^C^C^CTraceback (most recent call last):
    #   File "/Users/hobs/anaconda3/envs/nlpia/bin/record", line 10, in <module>
    #     sys.exit(main())
    #   File "/Users/hobs/src/nlpia/nlpia/scripts/record.py", line 179, in main
    #   File "/Users/hobs/src/nlpia/nlpia/scripts/record.py", line 64, in tts
    #     engine.runAndWait()
    #   File "/Users/hobs/src/nlpia/src/pyttsx3/pyttsx3/engine.py", line 188, in runAndWait
    #     self.proxy.runAndWait()
    #   File "/Users/hobs/src/nlpia/src/pyttsx3/pyttsx3/driver.py", line 204, in runAndWait
    #     self._driver.startLoop()
    #   File "/Users/hobs/src/nlpia/src/pyttsx3/pyttsx3/drivers/nsss.py", line 33, in startLoop
    #     AppHelper.runConsoleEventLoop()
    #   File "/Users/hobs/anaconda3/envs/nlpia/lib/python3.6/site-packages/PyObjCTools/AppHelper.py",
    #         line 242, in runConsoleEventLoop
    #     if not runLoop.runMode_beforeDate_(mode, nextfire):


if __name__ == '__main__':
    main()
