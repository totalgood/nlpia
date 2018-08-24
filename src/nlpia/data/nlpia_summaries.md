# NLPIA Summary Sentences from TextSum

v0.0.1rc results for hacky @ChattyAboutNLPIA pipeline

```asciidoc
[QUOTE,@ChattyAboutNLPIA,_About this Book_]
Language's true power is not necessarily in the words, but in the intent and emotion that formed that particular combination of words. 

This sentence was generated as @Chatty attempted to summarize the first 30 lines of chapter 8.

[QUOTE,@ChattyAboutNLPIA,_About this Book_]
Convolutional neural nets make an attempt to capture that ordering relationship by capturing localized relationships.
```

## Sentences are tagged to indicate the sentence is:

    `-` reflux
    `+` ill-formed: not reflux, but not well-formed
    `*` novel *and* well-formed

### Part 1 Summary

    - in chapter 1 , you 'll quickly begin to think of ways you can use machines that process words in your own life . 
    + in chapter 2 , we give you the tools you need to teach machines to extract words from the words in a natural language document . 
    - in chapter 4 , you 'll have the tools you need for many interesting nlp applications .

### Chapter 1 Summary

    - nlp is the first of many you 'll use to learn all the tricks needed to assemble an english language dialog engine . 
    - a computer program written with a programming language tells a machine exactly what to do . 
    - but there are no compilers or interpreters for natural languages such as english and french .

### Chapter 2 Summary

    _ zipf 's law _ and _ okapi bm25 _ . 
    + estimating the similarity of documents from a corpus using _ inverse document frequencies _ . 
    + finding relevant documents in a corpus based on keyword importance within each document .

### Chapter 3 Summary

    - punctuation marks , and numbers , but the techniques we use are easily extended to any other units of meaning contained in a sequence of characters , like ascii emoticons , mathematical symbols , and so on . 
    - retrieving tokens from a document will require some string manipulation beyond just the ` str.split -lrb- -rrb- ` method employed in chapter 1 '

### Chapter 4 Summary
    - google search had seemed more like an index , a tool that required a little skill in order to find what you were looking . 
    - but it soon got smarter and would accept more and more natural language searches . 
    * then smartphone autocomplete began dominating our lives .

Minor originality.

### Part 2 Summary

    - the neural networks of part 2 accomplish most of the tedious feature extraction work for you . 
    - the use of multilayered neural networks for machine learning is called _ deep learning _ . 
    * in part 2 , we begin to look at the patterns not just between words but between the characters within words .

Minor originality: "We also start to" -> "In part 2,"

### Chapter 10 Summary (30 lines)

    - over the course of the book , we have introduced common natural language processing tools which are key to understanding texts , or to classifying their sentiments or their content with a machine . 
    - but would n't it be nice to train a neural network to respond to a human 's questions ? 
    - or a response and the pairing of a statement in one language with its equivalent translated into another language are very closely related .

### Chapter 5 Summary

    * the network weights will be adjusted in the individual neurons based on the error through backpropagation . 
    - the effects of the next example 's learning stage are largely independent of the order of input data . 
    * convolutional neural nets make an attempt to capture that ordering relationship by capturing localized relationships .

First one needs context because it's just half of a longer sentence
Second one only deletes ", but there's another way."

### Chapter 6 Summary

    + this is the most exciting recent advancements in nlp was the `` discovery '' of word vectors . 
    - this process will help focus your word vector training on the relevant words . 
    - in the previous chapters , we ignored the nearby context of a word .

### Chapter 7 Summary

    + the first node will have on the third node -lrb- the 2nd time step thereafter -rrb- . 
    - this is , of course , important to the basic structure of the net , but it precludes the common case in human language that the tokens may be deeply interrelated but be separated greatly in the sentence .

### Chapter 8 Summary

    + the sentiment of novel text in a way that suits neural networks . 
    * language 's true power is not necessarily in the words , but in the intent and emotion that formed that particular combination of words . 
    * even the phrase `` machine-generated text '' inspires dread of a hollow , tinned voice .

First one is good, it came from: "And sometimes meaning is hidden beneath the words, in the intent and emotion that formed that particular combination of words."
Second one is a truncation of "Even the phrase 'machine-generated text' inspires dread of a hollow, tinned voice *issuing a chopped list of words.*"


### Part 3 Summary

    + three chapters we 'll also tackle the trickier problems of nlp . 
    in these last three chapters we 'll also tackle the trickier problems . 
    - and you 'll learn how to combine these techniques together to create complex behavior .

### Chapter 12 Summary

    + a hybrid chatbot architecture that combines the best ideas into one . 
    - for the first time in history we can speak to a machine in our own language , and we ca n't always tell that it is n't human . 
    - this means that machines can `` fake '' being human .

### Chapter 13 Summary (30 lines)

    - the `` humanness '' , or iq , of our dialog system seems to be limited by the data we train it with . 
    - besides ram , there 's another bottleneck in our natural language processing pipelines , the processor . 
    + as you had unlimited ram , larger corpora would take days to process with some of the more complex algorithms .

### Acknowledgements Summary (30 lines)

    + contributors came from a vibrant portland community sustained by organizations like pdx python . 
    - developers like zachary kent masterminded the architecture of ` openchat ` and extended it over two years as the book and our skills progressed . 
    - riley rustad helped create the data model used by the ` openchat ` twitter bot to promote pycon openspaces events '
