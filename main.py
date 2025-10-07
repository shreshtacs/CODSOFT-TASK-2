import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

#  Content-Based Filtering
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(movies['genre'])
cosine_sim = cosine_similarity(tfidf_matrix)

def content_recommendations(movie_title, top_n=3):
    
    if movie_title not in movies['title'].values:
        return f"Movie '{movie_title}' not found in the database. Available movies are: {', '.join(movies['title'].tolist())}"
    
    idx = movies.index[movies['title'] == movie_title][0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in scores[1:top_n+1]]
    return movies.iloc[movie_indices]['title'].tolist()

# Simple Collaborative Filtering 

user_item_matrix = ratings.pivot_table(index='user_id', columns='movie_id', values='rating', fill_value=0)


user_similarity = cosine_similarity(user_item_matrix)

def collaborative_recommendations(user_id, top_n=3):
    
    if user_id not in ratings['user_id'].values:
        return f"User {user_id} not found in the database. Available users are: {sorted(ratings['user_id'].unique().tolist())}"
    
  
    user_idx = user_item_matrix.index.get_loc(user_id)
    
    
    similar_users = list(enumerate(user_similarity[user_idx]))
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)
    similar_users = [i for i in similar_users if i[0] != user_idx]  
    
    user_ratings = user_item_matrix.iloc[user_idx].to_dict()
    unrated_movies = [m for m in movies['movie_id'] if user_ratings.get(m, 0) == 0]

    predictions = []
    for movie_id in unrated_movies:
    
        movie_col = user_item_matrix[movie_id]
        weighted_sum = 0
        similarity_sum = 0
        
        for similar_user_idx, similarity in similar_users[:3]: 
            if movie_col.iloc[similar_user_idx] > 0:  
                weighted_sum += similarity * movie_col.iloc[similar_user_idx]
                similarity_sum += similarity
            
        if similarity_sum > 0:
            predictions.append((movie_id, weighted_sum / similarity_sum))
    

    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    recommended_movie_ids = [m[0] for m in predictions[:top_n]]
    
    return movies[movies['movie_id'].isin(recommended_movie_ids)]['title'].tolist()


def get_recommendations():
    print("\n🎬 Movie Recommendation System 🎬")
    print("Available movies:", ", ".join(movies['title'].tolist()))
    print("Available users:", sorted(ratings['user_id'].unique().tolist()))
    
    while True:
        print("\nChoose an option:")
        print("1. Get content-based recommendations for a movie")
        print("2. Get collaborative recommendations for a user")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            movie_title = input("Enter a movie title: ")
            recommendations = content_recommendations(movie_title)
            print(f"\nRecommendations for '{movie_title}':")
            if isinstance(recommendations, str):
                print(recommendations)  # Error message
            else:
                for i, movie in enumerate(recommendations, 1):
                    print(f"{i}. {movie}")
        
        elif choice == '2':
            try:
                user_id = int(input("Enter a user ID: "))
                recommendations = collaborative_recommendations(user_id)
                print(f"\nRecommendations for User {user_id}:")
                if isinstance(recommendations, str):
                    print(recommendations)  # Error message
                else:
                    for i, movie in enumerate(recommendations, 1):
                        print(f"{i}. {movie}")
            except ValueError:
                print("Please enter a valid user ID (integer).")
        
        elif choice == '3':
            print("Thank you for using the Movie Recommendation System!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


get_recommendations()
