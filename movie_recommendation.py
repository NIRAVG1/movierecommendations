"""Movie Recommendation SystemConverted from the uploaded Jupyter notebook."""import numpy as np
import pandas as pd

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

credits.head(1)

movies.head(1)

movies = movies.merge(credits,on='title')



#genres
#id
#keywords
#title
#overview
#cast
#crew
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

movies.head()

movies.dropna(inplace = True)

movies.duplicated().sum()

movies.iloc[0].genres

def convert(obj):
  L=[]
  for i in ast.literal_eval(obj):
    L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)

movies['keywords'] =movies['keywords'].apply(convert)

movies.head()

def convert3(obj):
  L=[]
  for i in ast.literal_eval(obj):
    counter =0
    if counter !=3:
      L.append (i['name'])
      counter+=1
    else:
      break

    L.append(i['name'])
    return L

movies['cast'] = movies['cast'].apply(convert3)

import ast

def fetch_director(obj):
    L = []
    if obj is None or obj == []:
        return L
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
    except:
        pass
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(lambda x:x.split())

movies.head()

movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x] if x is not None else [])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x] if x is not None else [])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x] if x is not None else [])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x] if x is not None else [])

movies.head()

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

movies.head()

new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
#new.head()

new

new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()

!pip install nltk

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')

vector = cv.fit_transform(new['tags']).toarray()

vector.shape

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)
similarity

new[new['title'] == 'The Lego Movie'].index[0]

def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)


recommend('Avatar')

import pickle
pickle.dump(new,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))

