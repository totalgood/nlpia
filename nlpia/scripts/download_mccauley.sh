#!/usr/bin/env sh

# Thank you to Juilan McAuley, Mengting Wan, and Alex Yang at UCSD for providing this data!

# Modeling ambiguity, subjectivity, and diverging viewpoints in opinion question answering systems
# Mengting Wan, Julian McAuley
# International Conference on Data Mining (ICDM), 2016
# [pdf](http://cseweb.ucsd.edu/~jmcauley/pdfs/icdm16c.pdf)

# Addressing complex and subjective product-related queries with customer reviews
# Julian McAuley, Alex Yang
# World Wide Web (WWW), 2016
# [pdf](http://cseweb.ucsd.edu/~jmcauley/pdfs/www16b.pdf)

# unabridged Question and Answer Pairs (mulitple answers per product question, creating ambiguity)
wget http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Automotive.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Baby.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Beauty.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Cell_Phones_and_Accessories.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Clothing_Shoes_and_Jewelry.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Electronics.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Grocery_and_Gourmet_Food.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Health_and_Personal_Care.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Home_and_Kitchen.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Musical_Instruments.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Office_Products.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Patio_Lawn_and_Garden.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Pet_Supplies.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Sports_and_Outdoors.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Tools_and_Home_Improvement.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Toys_and_Games.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Video_Games.json.gz

# curated Question and Answer Pairs (only one answer per product question)
wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Appliances.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Arts_Crafts_and_Sewing.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Automotive.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Baby.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Beauty.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Cell_Phones_and_Accessories.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Clothing_Shoes_and_Jewelry.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Electronics.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Grocery_and_Gourmet_Food.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Health_and_Personal_Care.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Home_and_Kitchen.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Industrial_and_Scientific.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Musical_Instruments.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Office_Products.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Patio_Lawn_and_Garden.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Pet_Supplies.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Software.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Sports_and_Outdoors.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Tools_and_Home_Improvement.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Toys_and_Games.json.gz
wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Video_Games.json.gz