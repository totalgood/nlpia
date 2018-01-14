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

>>> bot.respond("hello **troll** !!!")
'Good one, human.'
"""

import os
from nlpia.constants import DATA_PATH
import aiml_bot


""" AIML Step 1
<category><pattern>HELLO ROSA </pattern><template>Hi Human!</template></category>
<category><pattern>HELLO TROLL </pattern><template>Good one, human.</template></category>
"""
bot = aiml_bot.Bot(learn=os.path.join(DATA_PATH, 'greeting_step1.aiml'))
# Loading /Users/hobs/src/nlpia/nlpia/data/greeting_step1.aiml...
# done (0.00 seconds)
# Loading /Users/hobs/src/nlpia/nlpia/data/greeting_step1.aiml...
# done (0.00 seconds)

""" AIML Patterns Step1: Good ones """
bot.respond("Hello Rosa,")
# 'Hi there!'
bot.respond("hello **troll** !!!")
# 'Good one, human.'

""" AIML Patterns Step1: Mismatches """
bot.respond("Helo Rosa")
# WARNING: No match found for input: Helo Rosa
# ''
bot.respond("Hello t-r-o-l-l")
# WARNING: No match found for input: Hello t-r-o-l-l
# Out[4]: ''

""" AIML Patterns Step2: Synonyms """
bot.learn(os.path.join(DATA_PATH, 'greeting_step2.aiml'))
bot.respond("Hey Rosa")
'Hi there!'
bot.respond("Hi Rosa")
'Hi there!'
bot.respond("Helo Rosa")
'Hi there!'
bot.respond("hello **troll** !!!")  # <1>
'Good one, human.'

""" AIML Patterns Step2: Mismatches """
bot.respond("Hello t-r-o-l-l")
# WARNING: No match found for input: Hello t-r-o-l-l
# Out[4]: ''

""" AIML Patterns Step3: Random Responses and Lists """
bot = aiml_bot.Bot(learn=os.path.join(DATA_PATH, 'greeting_step3.aiml'))
bot.learn(os.path.join(DATA_PATH, 'greeting_step3.aiml'))
bot.respond("Hey Rosa")
'Hello friend'
bot.respond("Hey Rosa")
'Hey you :)'
bot.respond("Hey Rosa")
'Hi Human!'
