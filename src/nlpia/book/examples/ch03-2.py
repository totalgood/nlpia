
# coding: utf-8

# In[1]:


from nltk.tokenize import TreebankWordTokenizer

sentence = "The faster Harry got to the store, the faster Harry, the faster, would get home."
tokenizer = TreebankWordTokenizer()
token_sequence = tokenizer.tokenize(sentence.lower())
print(token_sequence)


# In[2]:


from collections import Counter
bag_of_words = Counter(token_sequence)
print(bag_of_words)


# In[3]:


word_list = bag_of_words.most_common()  # Passing an integer as an argument will give you that many from the top of the list
print(word_list)


# In[4]:


times_harry_appears = bag_of_words['harry']
total_words = len(word_list) # The number of tokens from our original source.
tf = times_harry_appears/total_words

print(tf)


# In[5]:


kite_text = "A kite is traditionally a tethered heavier-than-air craft with wing surfaces that react against the air to create lift and drag. A kite consists of wings, tethers, and anchors. Kites often have a bridle to guide the face of the kite at the correct angle so the wind can lift it. A kite's wing also may be so designed so a bridle is not needed; when kiting a sailplane for launch, the tether meets the wing at a single point. A kite may have fixed or moving anchors. Untraditionally in technical kiting, a kite consists of tether-set-coupled wing sets; even in technical kiting, though, a wing in the system is still often called the kite. The lift that sustains the kite in flight is generated when air flows around the kite's surface, producing low pressure above and high pressure below the wings. The interaction with the wind also generates horizontal drag along the direction of the wind. The resultant force vector from the lift and drag force components is opposed by the tension of one or more of the lines or tethers to which the kite is attached. The anchor point of the kite line may be static or moving (e.g., the towing of a kite by a running person, boat, free-falling anchors as in paragliders and fugitive parakites or vehicle). The same principles of fluid flow apply in liquids and kites are also used under water. A hybrid tethered craft comprising both a lighter-than-air balloon as well as a kite lifting surface is called a kytoon. Kites have a long and varied history and many different types are flown individually and at festivals worldwide. Kites may be flown for recreation, art or other practical uses. Sport kites can be flown in aerial ballet, sometimes as part of a competition. Power kites are multi-line steerable kites designed to generate large forces which can be used to power activities such as kite surfing, kite landboarding, kite fishing, kite buggying and a new trend snow kiting. Even Man-lifting kites have been made."


# In[6]:


from collections import Counter
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

# kite_text = "A kite is traditionally ..."  # Step left to user, so we aren't repeating ourselves
tokens = tokenizer.tokenize(kite_text.lower())
token_sequence = Counter(tokens)
print(token_sequence)


# In[7]:


import nltk

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

tokens = [x for x in tokens if x not in stopwords]
kite_count = Counter(tokens)
print(kite_count)


# In[8]:


document_vector = []
doc_length = len(tokens)
for key, value in kite_count.most_common():
    document_vector.append(value / doc_length)

print(document_vector)


# In[9]:


doc_0 = "The faster Harry got to the store, the faster Harry, the faster, would get home."
doc_1 = "Harry is hairy and faster than Jill."
doc_2 = "Jill is not as hairy as Harry."


# In[10]:


tokens_0 = tokenizer.tokenize(doc_0.lower())
tokens_1 = tokenizer.tokenize(doc_1.lower())
tokens_2 = tokenizer.tokenize(doc_2.lower())
lexicon = set(tokens_0 + tokens_1 + tokens_2)

print(lexicon)


# In[11]:


print(len(lexicon))


# In[12]:


from collections import OrderedDict

vector_template = OrderedDict((token, 0) for token in lexicon)
print(vector_template)


# In[13]:


import copy

document_vectors = []
for doc in [doc_0, doc_1, doc_2]:

    vec = copy.copy(vector_template)  # So we are dealing with new objects, not multiple references to the same object

    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)

    for key, value in token_counts.items():
        vec[key] = value / len(lexicon)
    document_vectors.append(vec)


# In[14]:


import math

def cosine_sim(vec1, vec2):
    """
    Since our vectors are dictionaries, lets convert them to lists for easier mathing.
    """
    vec1 = [val for val in vec1.values()]
    vec2 = [val for val in vec2.values()]
    
    dot_prod = 0
    for i, v in enumerate(vec1):
        dot_prod += v * vec2[i]
        
    mag_1 = math.sqrt(sum([x**2 for x in vec1]))
    mag_2 = math.sqrt(sum([x**2 for x in vec2]))
    
    return dot_prod / (mag_1 * mag_2)


# In[15]:


from nltk.corpus import brown
print(len(brown.words()))  # words is a builtin method of the nltk corpus object that gives a list of tokens


# In[16]:


from collections import Counter

puncs = [',', '.', '--', '-', '!', '?', ':', ';', '``', "''", '(', ')', '[', ']']
word_list = [x.lower() for x in brown.words() if x not in puncs]
token_counts = Counter(word_list)
print(token_counts.most_common(20))


# In[17]:


history_text = 'Kites were invented in China, where materials ideal for kite building were readily available: silk fabric for sail material; fine, high-tensile-strength silk for flying line; and resilient bamboo for a strong, lightweight framework. The kite has been claimed as the invention of the 5th-century BC Chinese philosophers Mozi (also Mo Di) and Lu Ban (also Gongshu Ban). By 549 AD paper kites were certainly being flown, as it was recorded that in that year a paper kite was used as a message for a rescue mission. Ancient and medieval Chinese sources describe kites being used for measuring distances, testing the wind, lifting men, signaling, and communication for military operations. The earliest known Chinese kites were flat (not bowed) and often rectangular. Later, tailless kites incorporated a stabilizing bowline. Kites were decorated with mythological motifs and legendary figures; some were fitted with strings and whistles to make musical sounds while flying. From China, kites were introduced to Cambodia, Thailand, India, Japan, Korea and the western world. After its introduction into India, the kite further evolved into the fighter kite, known as the patang in India, where thousands are flown every year on festivals such as Makar Sankranti. Kites were known throughout Polynesia, as far as New Zealand, with the assumption being that the knowledge diffused from China along with the people. Anthropomorphic kites made from cloth and wood were used in religious ceremonies to send prayers to the gods. Polynesian kite traditions are used by anthropologists get an idea of early "primitive" Asian traditions that are believed to have at one time existed in Asia.'


# In[18]:


# intro_text = "A kite is traditionally ..."  # Step left to user, as above
intro_text = kite_text.lower()
intro_tokens = tokenizer.tokenize(intro_text)
# history_text = "Kites were invented in China, ..."  # Also as above
history_text = history_text.lower()
history_tokens = tokenizer.tokenize(history_text)
intro_total = len(intro_tokens)
history_total = len(history_tokens)


# In[19]:


intro_tf = {}
history_tf = {}
intro_counts = Counter(intro_tokens)
intro_tf['kite'] = intro_counts['kite'] / intro_total
history_counts = Counter(history_tokens)
history_tf['kite'] = history_counts['kite'] / history_total
print('Term Frequency of "kite" in intro is: {}'.format(intro_tf['kite']))
print('Term Frequency of "kite" in history is: {}'.format(history_tf['kite']))


# In[20]:


intro_tf['and'] = intro_counts['and'] / intro_total
history_tf['and'] = history_counts['and'] / history_total
print('Term Frequency of "and" in intro is: {}'.format(intro_tf['and']))
print('Term Frequency of "and" in history is: {}'.format(history_tf['and']))


# In[21]:


num_docs_containing_and = 0
for doc in [intro_tokens, history_tokens]:
    if 'and' in doc:
        num_docs_containing_and += 1


# In[22]:


num_docs_containing_kite = 0
for doc in [intro_tokens, history_tokens]:
    if 'kite' in doc:
        num_docs_containing_kite += 1


# In[23]:


num_docs_containing_china = 0
for doc in [intro_tokens, history_tokens]:
    if 'china' in doc:
        num_docs_containing_china += 1


# In[24]:


intro_tf['china'] = intro_counts['china'] / intro_total
history_tf['china'] = history_counts['china'] / history_total


# In[25]:


num_docs = 2
intro_idf = {}
history_idf = {}
intro_idf['and'] = num_docs / num_docs_containing_and 
history_idf['and'] = num_docs / num_docs_containing_and 
intro_idf['kite'] = num_docs / num_docs_containing_kite 
history_idf['kite'] = num_docs / num_docs_containing_kite 
intro_idf['china'] = num_docs / num_docs_containing_china 
history_idf['china'] = num_docs / num_docs_containing_china 


# In[26]:


intro_tfidf = {}

intro_tfidf['and'] = intro_tf['and'] * intro_idf['and']
intro_tfidf['kite'] = intro_tf['kite'] * intro_idf['kite']
intro_tfidf['china'] = intro_tf['china'] * intro_idf['china']


# In[27]:


history_tfidf = {}

history_tfidf['and'] = history_tf['and'] * history_idf['and']
history_tfidf['kite'] = history_tf['kite'] * history_idf['kite']
history_tfidf['china'] = history_tf['china'] * history_idf['china']


# In[28]:


document_tfidf_vectors = []
documents = [doc_0, doc_1, doc_2]
for doc in documents:

    vec = copy.copy(vector_template)  # So we are dealing with new objects, not multiple references to the same object

    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)

    for key, value in token_counts.items():
        docs_containing_key = 0
        for _doc in documents:
          if key in _doc:
            docs_containing_key += 1
        tf = value / len(lexicon)
        if docs_containing_key:
            idf = len(documents) / docs_containing_key
        else:
            idf = 0
        vec[key] = tf * idf 
    document_tfidf_vectors.append(vec)


# In[29]:


query = "How long does it take to get to the store?"
query_vec = copy.copy(vector_template) 

query_vec = copy.copy(vector_template)  # So we are dealing with new objects, not multiple references to the same object

tokens = tokenizer.tokenize(query.lower())
token_counts = Counter(tokens)

for key, value in token_counts.items():
    docs_containing_key = 0
    for _doc in documents:
      if key in _doc.lower():
        docs_containing_key += 1
    if docs_containing_key == 0:  # We didn't find that token in the lexicon go to next key
        continue
    tf = value / len(tokens)
    idf = len(documents) / docs_containing_key 
    query_vec[key] = tf * idf 

print(cosine_sim(query_vec, document_tfidf_vectors[0]))
print(cosine_sim(query_vec, document_tfidf_vectors[1]))
print(cosine_sim(query_vec, document_tfidf_vectors[2]))


# In[30]:


from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [doc_0, doc_1, doc_2]

vectorizer = TfidfVectorizer(min_df=1)
model = vectorizer.fit_transform(corpus)

print(model.todense())  # The model becomes a sparse numpy matrix, as in a large corpus there would be mostly zeros to deal with.  todense() brings it back to a regular numpy matrix for our viewing pleasure.

