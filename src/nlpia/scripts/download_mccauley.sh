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

# reviews
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_10.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_10.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_10.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_10.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Home_and_Kitchen_10.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Home_and_Kitchen_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Kindle_Store_10.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Kindle_Store_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_10.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Health_and_Personal_Care_10.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Health_and_Personal_Care_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_10.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games_10.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Tools_and_Home_Improvement_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_10.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Apps_for_Android_10.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Apps_for_Android_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Office_Products_10.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Office_Products_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Pet_Supplies_10.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Pet_Supplies_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Automotive_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Grocery_and_Gourmet_Food_10.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Grocery_and_Gourmet_Food_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Patio_Lawn_and_Garden_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Baby_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Digital_Music_10.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Digital_Music_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Amazon_Instant_Video_5.json.gz
