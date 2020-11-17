'''
This is the program which returns the list of recommended items based on the item clicked/bought by the user
'''

#!/usr/bin/env python
# coding: utf-8




from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from google.colab import files
# files.upload()

import pandas as pd
# import numpy as np

def mix_contents(row):
  return (f"{row['genre']} {row['keywords']}")

def recommendation(user_clicked):
    df = pd.read_csv("conf/recommendation_dataset.csv")
    contents = ['keywords', 'genre']
    
    for content in contents:
      df[content] = df[content].fillna('')
    
    df["mixed_contents"] = df.apply(mix_contents, axis = 1)
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df["mixed_contents"])
    
    #Finding the cosine similarity to recommend similar items
    sim_score = cosine_similarity(count_matrix)
    
    item_index = df[df.class_name == user_clicked]['index'].values[0]
    similar_items = list(enumerate(sim_score[item_index]))
    sorted_similar_items = sorted(similar_items, key = lambda x:x[1], reverse = True)
    
    recomm_items = []
    for item in sorted_similar_items:
      recomm_items.append(df[df.index == item[0]]['class_name'].values[0])

    return recomm_items

# recom = recommendation('gloves')
# for item in recom:
#     print(f"Rank 1: {item}")



    