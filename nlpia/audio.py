#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import speech_recognition as sr
import pyaudio
import wave
from io import BytesIO


def record_audio(source='Microphone', energy_threshold=300, pause_threshold=.9,
        dynamic_energy_ratio=1.5, dynamic_energy_adjustment_damping=.15, **kwargs):
    """ Listten for a single utterance (concluded with a 2 sec pause) and return a recording in an Audio object

    Arguments:
        energy_threshold (int): minimum audio energy to trigger start of recording (default=300)
        dynamic_energy_adjustment_damping (float): delay dynamic energy threshold change: 1=static energy_threshold (.15)
        dynamic_energy_ratio (float): sound energy change that triggers recording: 1=static energy_threshold (1.5)
        pause_threshold (float): non-speaking audio seconds before a phrase is considered complete (.9)
        operation_timeout (float): seconds after an internal operation starts before it times out: None=neverdynamic_energy_adjustment_damping
        self.phrase_threshold (float): minimum seconds of speaking audio required before stopping recording of a phrase (.3)
        self.non_speaking_duration (float): seconds of non-speaking audio to retain on both sides of the recording (pause_threshold)
 """
    r = sr.Recognizer(
        energy_threshold=300, pause_threshold=pause_threshold,
        dynamic_energy_threshold=(dynamic_energy_ratio > 1 and dynamic_energy_adjustment_damping < 1),
        dynamic_energy_ratio=dynamic_energy_ratio, dynamic_energy_adjustment_damping=dynamic_energy_adjustment_damping,
        **kwargs)
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


def save_audio(audio, path='audio.wav'):
    data = getattr(audio, 'get_{}_data'.format(path.lower().split('.')[-1].strip()), 'get_wav_data')()
    with open(path, 'wb') as fout:
        fout.write(data)
    return path
