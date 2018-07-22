import nltk
import spacy


text = open('/Users/hobs/src/lane/manuscript/Chapter 01 -- The Language of Thought.asc').read()
sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
nltk_sents = list(sentence_detector.tokenize(text.strip()))

nlp = spacy.load('en_core_web_sm')
ch1 = nlp(text)
spacy_sents = []
for w, span in enumerate(ch1.sents):
    sent = ''.join(ch1[i].string for i in range(span.start, span.end)).strip()
    spacy_sents.append(sent)

len(nltk_sents)
len(spacy_sents)
