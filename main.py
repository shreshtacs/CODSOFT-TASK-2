import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Sample dataset
movies = pd.DataFrame({
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['Inception', 'Interstellar', 'The Dark Knight', 'Memento', 'Dunkirk'],
    'genre': ['Sci-Fi', 'Sci-Fi', 'Action', 'Thriller', 'War']
})

ratings = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    'movie_id': [1, 2, 2, 3, 1, 3, 4, 5, 2, 5],
    'rating': [5, 4, 5, 3, 4, 5, 2, 4, 3, 5]
})

# Content-Based Filtering
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(movies['genre'])
cosine_sim = cosine_similarity(tfidf_matrix)

def content_recommendations(movie_title):
    idx = movies.index[movies['title'] == movie_title][0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in scores[1:4]]  # Top 3 recommendations
    return movies.iloc[movie_indices]['title'].tolist()

# Collaborative Filtering using SVD
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)
model = SVD()
model.fit(trainset)

def collaborative_recommendations(user_id, num_recommendations=3):
    all_movie_ids = movies['movie_id'].unique()
    rated_movies = ratings[ratings['user_id'] == user_id]['movie_id'].tolist()
    unrated_movies = [m for m in all_movie_ids if m not in rated_movies]
    predictions = [(m, model.predict(user_id, m).est) for m in unrated_movies]
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    recommended_movie_ids = [m[0] for m in predictions[:num_recommendations]]
    return movies[movies['movie_id'].isin(recommended_movie_ids)]['title'].tolist()

# Example usage
print("Content-Based Recommendations for 'Inception':", content_recommendations('Inception'))
print("Collaborative Filtering Recommendations for User 1:", collaborative_recommendations(1))
