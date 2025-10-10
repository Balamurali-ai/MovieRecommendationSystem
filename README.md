import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# -----------------------------
st.title("🎬 Movie Recommendation System")

# Load datasets
try:
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv").drop("timestamp", axis=1)
    st.success("Datasets loaded successfully!")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# User-item matrix
user_movie_matrix = ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)

# Train SVD
svd = TruncatedSVD(n_components=20, random_state=42)
matrix_reduced = svd.fit_transform(user_movie_matrix)
predicted_ratings = np.dot(matrix_reduced, svd.components_)
predicted_df = pd.DataFrame(predicted_ratings, index=user_movie_matrix.index, columns=user_movie_matrix.columns)

# Recommender
def recommend_movies(user_id, top_n=10):
    if user_id not in predicted_df.index:
        return pd.DataFrame([["Invalid User ID", "N/A"]], columns=["title", "genres"])
    user_ratings = predicted_df.loc[user_id]
    already_rated = ratings[ratings["userId"] == user_id]["movieId"].tolist()
    user_ratings = user_ratings.drop(labels=already_rated, errors="ignore")
    top_movie_ids = user_ratings.sort_values(ascending=False).head(top_n).index
    return movies[movies["movieId"].isin(top_movie_ids)][["title", "genres"]]

# Frontend
st.header("🔍 Get Your Recommendations")
user_id = st.number_input("Enter your User ID:", min_value=1, max_value=int(ratings["userId"].max()), value=1)
top_n = st.slider("Select number of recommendations:", min_value=1, max_value=20, value=10)

if st.button("Recommend"):
    recommendations = recommend_movies(user_id, top_n)
    st.write(recommendations)
