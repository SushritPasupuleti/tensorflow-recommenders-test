#%%
from typing import Dict, Text

import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
# %%
# Ratings data.
ratings = tfds.load('movie_lens/100k-ratings', split="train")
# Features of all the available movies.
movies = tfds.load('movie_lens/100k-movies', split="train")

# Select the basic features.
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"]
})
movies = movies.map(lambda x: x["movie_title"])
# %%
print(movies)