import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

movies=pd.read_csv("../Data/tmdb_movies.csv")
credits=pd.read_csv("../Data/tmdb_credits.csv")
print(f"Movieset size:{movies.shape}")
print()
print(f"Creditset size:{credits.shape}")
print()

dataset=pd.merge(movies,credits[["movie_id","cast","crew"]] ,left_on="id",right_on="movie_id",how="inner")

soup_ingiredients=[
    "title",
    "keywords",
    "overview",
    "original_language",
    "genres",
    "cast",
    "crew"
]

dataset[soup_ingiredients]=dataset[soup_ingiredients].fillna("")

print(f"Merged dataset shape:{dataset.shape}")
print()
print("Datasetsamples")
print()
print(dataset.head(3))

def extract_names(raw_text):
    data_list=ast.literal_eval(raw_text)
    return[item["name"].lower() for item in data_list]

def extract_actors(raw_text):
    data_list=ast.literal_eval(raw_text)
    return [item["name"].lower() for item in data_list[:3]]

def extract_director(raw_text):
    datalist=ast.literal_eval(raw_text)
    for item in datalist:
        if item["job"]=="director":
          return item["name"].lower()
    return []    

dataset["keywords"]=dataset["keywords"].apply(extract_names)
dataset["genres"]=dataset["genres"].apply(extract_names)
dataset['cast']=dataset["cast"].apply(extract_actors)
dataset["crew"]=dataset['crew'].apply(extract_director)

dataset["keywords"]=dataset["keywords"].apply(lambda x:" ".join(x))
dataset["genres"]=dataset['genres'].apply(lambda x:" ".join(x))
dataset["cast"]=dataset["cast"].apply(lambda x:" ".join(x))
dataset["crew"]=dataset["crew"].apply(lambda x:" ".join(x))

dataset["soup"] = (
    dataset["title"] + " " +
    dataset["keywords"] + " " +
    dataset["overview"] + " " +
    dataset["original_language"] + " " +
    dataset["genres"] + " " +
    dataset["cast"] + " " +
    dataset["crew"]
)
print()
print("Samples")
print(dataset["soup"].iloc[0])
print()
print(dataset["soup"].iloc[5])
print()
print(dataset["soup"].iloc[10])
print()

vectorizer=TfidfVectorizer()

Tfid_matrice=vectorizer.fit_transform(dataset['soup'])
print(f"Tfid matrice size:{Tfid_matrice.shape}")
print()

similarity=cosine_similarity(Tfid_matrice)
print(f"Similarity Matrice Shape:{similarity.shape}")
print()


movie_index=pd.Series(
    dataset.index,
    index=dataset["title"]
).drop_duplicates()

def recommend(movie_name):
    index=movie_index[movie_name]

    similarity_score=list(
        enumerate(similarity[index])
    )
    similarity_score=sorted(
        similarity_score,key=lambda x: x[1],reverse=True
    )
    print(f'Similar movies like {movie_name}')
    for movie,score in similarity_score[1:8]:
        movie_name=dataset.iloc[movie]["title"]

        print(movie_name)

recommend("Spider-Man 3")        
    

