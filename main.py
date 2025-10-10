import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.decomposition import TruncatedSVD

# --- TMDb API Configuration ---
# Securely access the API key from Streamlit's secrets management.
# This requires you to have a .streamlit/secrets.toml file.
try:
    TMDB_API_KEY = st.secrets["tmdb_api_key"]
except (KeyError, FileNotFoundError):
    TMDB_API_KEY = None

def fetch_poster(movie_title):
    """Fetches the movie poster URL from TMDb API."""
    if not TMDB_API_KEY:
        return "https://placehold.co/500x750/333333/FFFFFF?text=API+Key+Missing"

    # Properly encode the movie title for the URL query
    search_query = requests.utils.quote(movie_title)
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={search_query}"
    
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()
        
        if data.get('results'):
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return "https://image.tmdb.org/t/p/w500" + poster_path
    
    except requests.exceptions.RequestException:
        # Silently fail on API errors to not disrupt the user experience
        pass 

    # Fallback image if no poster is found
    return "https://placehold.co/500x750/333333/FFFFFF?text=No+Poster"

# --- Page Configuration and Custom CSS ---
st.set_page_config(layout="wide")

# Inject MINIMAL and cleaner CSS to fix alignment without overriding the theme.
st.markdown("""
<style>
    /* Target the container of each column to ensure they stretch vertically */
    .st-emotion-cache-16txtl3 {
        align-items: stretch;
    }
    /* A container to help with structure and hover effects */
    .movie-card-container {
        display: flex;
        flex-direction: column;
        height: 100%; /* This is crucial to make cards in a row equal height */
        border-radius: 10px;
        overflow: hidden;
        transition: transform 0.2s ease-in-out;
    }
    .movie-card-container:hover {
        transform: scale(1.02); /* A subtle lift on hover */
    }
    /* Poster image styling */
    .movie-poster {
        width: 100%;
        aspect-ratio: 2 / 3; /* Enforce a consistent aspect ratio */
        object-fit: cover;
    }
    /* Details section below the poster */
    .movie-details {
        padding: 10px 4px; /* More vertical, less horizontal padding */
        flex-grow: 1; /* Allow this section to grow and fill space */
        display: flex;
        flex-direction: column;
    }
    /* Movie title styling for consistent height */
    .movie-title {
        font-weight: bold;
        font-size: 1rem;
        /* CRITICAL: Fix height and handle overflow for perfect alignment */
        height: 3em; /* Reserve space for approx. 2 lines of text */
        line-height: 1.5em;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2; /* Limit text to 2 lines */
        -webkit-box-orient: vertical;
        margin-bottom: auto; /* Pushes title up and genres down */
    }
    /* Genres styling */
    .movie-genres {
        font-size: 0.85rem;
        color: #888; /* A subtle gray for the genres */
        padding-top: 5px;
    }
</style>
""", unsafe_allow_html=True)


st.title("🎬 Movie Recommendation System")
st.write("Built with Collaborative Filtering (SVD)")

# --- Data Loading ---
@st.cache_data
def load_data():
    """Loads and caches the movie and rating datasets."""
    try:
        movies = pd.read_csv("data/movies.csv")
        ratings = pd.read_csv("data/ratings.csv").drop("timestamp", axis=1)
        return movies, ratings
    except FileNotFoundError:
        st.error("Error: 'movies.csv' or 'ratings.csv' not found.")
        st.info("Please make sure your data files are in a folder named 'data' in the same directory as your script.")
        return None, None

movies, ratings = load_data()

if movies is None or ratings is None:
    st.stop()

# --- SVD Model Training ---
@st.cache_resource
def train_svd_model(ratings_data):
    """Trains the SVD model and caches the result."""
    st.write("Training SVD model... (This will be cached after the first run)")
    user_movie_matrix = ratings_data.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    svd = TruncatedSVD(n_components=20, random_state=42)
    matrix_reduced = svd.fit_transform(user_movie_matrix)
    predicted_ratings = np.dot(matrix_reduced, svd.components_)
    predicted_df = pd.DataFrame(predicted_ratings, index=user_movie_matrix.index, columns=user_movie_matrix.columns)
    return predicted_df

predicted_df = train_svd_model(ratings)

# --- Recommender Function ---
def recommend_movies(user_id, top_n=10):
    """Recommends movies for a given user based on predicted ratings."""
    if user_id not in predicted_df.index:
        return pd.DataFrame(columns=["movieId", "title", "genres"])

    user_ratings = predicted_df.loc[user_id]
    already_rated_movie_ids = ratings[ratings["userId"] == user_id]["movieId"].tolist()
    user_ratings = user_ratings.drop(labels=already_rated_movie_ids, errors="ignore")
    top_movie_ids = user_ratings.sort_values(ascending=False).head(top_n).index
    recommendations = movies[movies["movieId"].isin(top_movie_ids)]
    # Reorder the recommendations to match the predicted rating order
    recommendations = recommendations.set_index('movieId').loc[top_movie_ids].reset_index()
    return recommendations

# --- Streamlit Frontend ---
st.header("🔍 Get Your Movie Recommendations")

user_id = st.number_input(
    "Enter your User ID:", 
    min_value=1, 
    max_value=int(ratings["userId"].max()), 
    value=1,
    help="Enter a User ID from the dataset to get personalized recommendations."
)
top_n = st.slider(
    "Select number of recommendations:", 
    min_value=5, 
    max_value=20, 
    value=10
)

if st.button("Recommend", type="primary"):
    if not TMDB_API_KEY:
        st.error("TMDb API key is not configured. Please add it to your .streamlit/secrets.toml file.")
    else:
        with st.spinner(f'Finding top {top_n} movies for you...'):
            recommendations = recommend_movies(user_id, top_n)
            
            if not recommendations.empty:
                st.success("Here are your recommended movies!")
                
                num_columns = 5
                cols = st.columns(num_columns)
                
                for idx, row in recommendations.iterrows():
                    col = cols[idx % num_columns]
                    with col:
                        poster_url = fetch_poster(row['title'])
                        # Sanitize title to avoid breaking the HTML if it contains quotes
                        safe_title = row['title'].replace('"', '&quot;')
                        genres = row['genres'].replace('|', ', ')

                        # Render the card using HTML with our new, cleaner CSS classes
                        st.markdown(f"""
                        <div class="movie-card-container">
                            <img class="movie-poster" src="{poster_url}" alt="{safe_title} Poster">
                            <div class="movie-details">
                                <div class="movie-title">{safe_title}</div>
                                <div class="movie-genres">{genres}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning(f"Could not generate recommendations for User ID {user_id}.")

