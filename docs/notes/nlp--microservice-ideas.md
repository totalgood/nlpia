# Chatbot Microservice Ideas

+ indicates one we've started on
- is just an idea

- Book Index Builder
	input: text file or URL to text
	output: asciidoc-formated index section (words and page numbers)
- Summarizer
- Link Shrinker/Tracker (Hannes)
	input: text file or URL
	output: text file with all the URLs shrunken to bit.ly or whatever
+ Slang Translator (Cole has a start on this)
	input: word or phrase
	output: word or phrase in "plain" English
- Sarcasm Detector
- Troll Detector
+ Kindness Sentiment Analyzer (Hobson's hope)
- Joke Generator
- Quote Generator
- Pun Generator
- Poem Generator
- Haiku Generator
- IQ Test Dialog Engine
- Pyschological Profile Dialog Engine
	input: responses to questions from a human
	output: questions a psychologist would ask to administer a psych profile test
- Psychological Profile estimator (from text/tweets/posts)
- WikiDB Question Answerer
- WordNet Thesaurus
- Dictionary
- Usage Suggester (kindof like OED)
	input: word or phrase
	output: list of the most common, prevalent sentences where the word/phrase is used
- Grammar Checker
- Genre Identifier (from music lyric genres)
+ Decade Identifier (from music lyric decades)
- Sentence Segmenter
- Section/Topic Transition Detector (Change Detection)
+ Popularity Estimator (Tweet Faves from `twip`)
- Conversation Graph DB Storage
	input: line of text and meta data about context, author, datetime, geo
	output: nothing ;)
	input: query for conversations, or individual lines, or iterables (pagination) over a corpus
    output: text, or paginated text, with metadata
- Modality (Pattern sentiment analysis package)
- Polarity and Subjectivity (Pattern sentiment analysis package)
- Passive/Active Voice
- Style similarity to famous prolific authors like HemmingwayApp
	- Shakespear
	- Niethche
	- Tolstoy
	- Hemmingway
	- Mark Twain
+ Twitter Snarfer
	input: search query and other configuration parameters
	output: queriable DB of Tweets
- Sentence/phrase/quote segmenter
  - add newlines to DetectorMorse's set of possible "sentence" terminators "!?." and retrain
  - tagging sentences with any context information that is lost during segmentation. 
      - is this a heading, a bullet beneath a heading, a sentence within a paragraph, a solitary sentence disconnected from the others, is it a question, is it a question or a statement about a question, (e.g. "She said 'What's up?' " etc), is it emphatic ("She said "Help!" and I slept.")
  - tagging makes the segmenters reversible and the next stage in the pipeline has more information about context. 
- Plug up an Grant Ingersol's RTFMBot to the Manning writing process documentation hosted on Box.com
- Grant Ingersol's RTFMbot so that chatbot can talk on totalgood.slack.com


Many of these could just aggregate other open-source, open-data APIs like these and add themselves to lists like this:

[Web API Database](http://www.programmableweb.com/category/dictionary/api)
[Text Analysis APIs](https://market.mashape.com/textanalysis/textanalysis)