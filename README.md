# Sentimemt_Analysis

## Env
* Run on Google Colab
* package
```
- numpy
- pandas
- string
- nltk
- re
- sklearn
- gensim
- Word2Vec
```
## Context
This dataset consists of reviews of fine foods from amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review. It also includes reviews from all other Amazon categories.

## Dataset 
* Resource: Amazon Fine Food Reviews
* link: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
* Data content:
  - Reviews.csv: Pulled from the corresponding SQLite table named Reviews in database.sqlite
  - database.sqlite: Contains the table 'Reviews'
* download : Review.csv

## Data preprocessing
* Retain column "Text", "Score"
* Remove stop words in column "Text"
* Turn word to vector by tf-idf, word2vec

## Model 
* CNN
* LSTM

## Output
* predict the column "Score"
* Evaluation: Accuracy -> 0.7861250042915344
