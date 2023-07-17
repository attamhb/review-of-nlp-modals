

import pandas as pd
import numpy as np

#https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?select=IMDB+Dataset.csv
df = pd.read_csv('./data/imdb_data.csv')

df_review = df['review']
df_sentiment = df['sentiment']

string =  ""
[string + char for char in df_review]
print(string)
print(''.join(sorted(set(df_review[0]))))

#print(df_sentiment[0])





