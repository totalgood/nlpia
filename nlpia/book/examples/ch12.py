#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Natural Language Processing in Action -- Chapter 12 Getting Chatty -- examples
>>> import os
>>> from nlpia.constants import DATA_PATH
>>> import aiml_bot
>>> bot.learn(os.path.join(DATA_PATH, 'greeting_step1.aiml'))
Loading /Users/hobs/src/nlpia/nlpia/data/greeting_step1.aiml...
done (0.00 seconds)

>>> bot.respond('Hello Rosa')
'Hi there!'

>>> bot.respond("hello **stupid** !!!")
'Good one, human.'
"""

import os
from nlpia.constants import DATA_PATH
import aiml_bot

bot = aiml_bot.Bot(learn=os.path.join(DATA_PATH, 'greeting_step1.aiml'))
# Loading /Users/hobs/src/nlpia/nlpia/data/greeting_step1.aiml...
# done (0.00 seconds)
# Loading /Users/hobs/src/nlpia/nlpia/data/greeting_step1.aiml...
# done (0.00 seconds)

## Step1: Good ones
bot.respond("Hello Rosa,")
# 'Hi there!'
bot.respond("hello **stupid** !!!")
# 'Good one, human.'

## Step1: Mismatches
bot.respond("Helo Rosa")
# WARNING: No match found for input: Helo Rosa
# ''
bot.respond("Hello stu-pid")
# WARNING: No match found for input: Hello stu-pid
# Out[4]: ''

