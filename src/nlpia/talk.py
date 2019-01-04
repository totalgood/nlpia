""" Speech recognition and generation tools based on pocketsphinx and wavenet """
import os

import numpy as np
import pyaudio as pa
from pocketsphinx import pocketsphinx as ps
# from sphinxbase import sphinxbase as sb

import logging

try:
    from nlpia.constants import DATA_PATH
except ImportError:
    DATA_PATH=os.path.abspath(os.path.curdir)

logger = logging.getLogger()

"""
An attempt at programmitically finding language models for pocketsphinx

>> from pocketsphinx import pocketsphinx
>> pocketsphinx.__file__
'/anaconda3/lib/python3.6/site-packages/pocketsphinx/pocketsphinx.py'

we want:
'/anaconda3/pkgs/pocketsphinx-python-0.1.3-py36h470a237_0/lib/python3.6/site-packages/pocketsphinx
"""
LIBDIR = os.path.join(os.environ.get('CONDA_PREFIX'),
                      '..', '..',
                      'pkgs',
                      *("pocketsphinx-python-0.1.3-py36h470a237_0/lib/python3.6/site-packages/pocketsphinx".split('/')))
MODELDIR = os.path.join(LIBDIR, 'model')

RATE = 16000
BUFSIZE = 1024
SECONDS = 5


def find_model():
    return dict(libdir=None, modeldir=None)


def get_decoder(libdir=None, modeldir=None, lang='en-us'):
    """ Create a decoder with the requested language model """
    modeldir = modeldir or (os.path.join(libdir, 'model') if libdir else MODELDIR)
    libdir = os.path.dirname(modeldir)
    config = ps.Decoder.default_config()
    config.set_string('-hmm', os.path.join(modeldir, lang))
    config.set_string('-lm', os.path.join(modeldir, lang + '.lm.bin'))
    config.set_string('-dict', os.path.join(modeldir, 'cmudict-' + lang + '.dict'))
    print(config)
    return ps.Decoder(config)


def evaluate_results(dec):
    hypothesis = dec.hyp()
    logmath = dec.get_logmath()
    report = dict(
        text=hypothesis.hypstr,
        score=hypothesis.best_score,
        confidence=logmath.exp(hypothesis.prob),
        segments=tuple((seg.word for seg in dec.seg()))
    )
    return report


def transcribe(decoder, audio_file, libdir=None):
    """ Decode streaming audio data from raw binary file on disk. """
    decoder = get_decoder()

    decoder.start_utt()
    stream = open(audio_file, 'rb')
    while True:
        buf = stream.read(1024)
        if buf:
            decoder.process_raw(buf, False, False)
        else:
            break
    decoder.end_utt()
    return evaluate_results(decoder)


def listen(seconds=SECONDS, rate=RATE, bufsize=BUFSIZE):
    dec = get_decoder()
    mic = pa.PyAudio()
    stream = mic.open(format=pa.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=bufsize)
    stream.start_stream()

    in_speech_bf = False
    dec.start_utt()

    recording = []
    results = []
    for i in range(int(seconds * rate / bufsize)):
        buf = stream.read(bufsize, exception_on_overflow=False)

        if buf:
            # recording += list(np.array(buf))
            # print(np.array(buf).std())
            dec.process_raw(buf, False, False)
            if dec.get_in_speech() != in_speech_bf:
                in_speech_bf = dec.get_in_speech()
                if not in_speech_bf:
                    dec.end_utt()
                    print('Result:', dec.hyp().hypstr)
                    results.append(evaluate_results(dec))
                    dec.start_utt()
        else:
            break
    dec.end_utt()

    # import pandas as pd
    # df = pd.DataFrame(recording)
    return results


def test(audio_file=os.path.join(DATA_PATH, 'goforward.raw'), decoder=None):
    decoder = decoder or get_decoder()
    report = transcribe(decoder, audio_file=audio_file)
    try:
        assert report['text'] == 'go forward ten meters'
    except AssertionError:            
        print(report)
        print('Decoder instance: ', str(dec))
        raise
    print('nlpia.talk.test() passed')


if __name__ == '__main__':
    reports = listen()
    print([r['text'] for r in reports])

    
