#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from traceback import format_exc

import speech_recognition as sr
import pyttsx3


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
