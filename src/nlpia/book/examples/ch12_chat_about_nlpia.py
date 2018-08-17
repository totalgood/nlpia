""" Text summarizer to "author" (generate) some content for the unfinished "About this Book" section.

1. Download a pretrained sequence-to-sequence 
    [text summarization model](https://github.com/totalgood/pointer-generator#looking-for-pretrained-model)
2. Parse and segment asciidoc text to extract natural language sentences with 
    [nlpia.book_parser](https://github.com/totalgood/nlpia)
3. Use the text summarization model on the first 30 or so lines of text in each asciidoc file (typically a chapter):
    [nlpia.book.examples.ch12_chat_about_nlpia](https://github.com/totalgood/nlpia/tree/master/src/nlpia/book/examples/ch12_chat_about_nlpia.py)

References:
    * [Pointer Generator](https://github.com/totalgood/pointer-generator)
    * [CNN and Daily Mail Story Datasets](https://github.com/totalgood/cnn-dailymail)
"""
