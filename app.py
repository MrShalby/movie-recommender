import streamlit as st
import pandas as pd
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(page_title="Simple Movie Recommender", layout="centered")

# Load and prepare data
@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

    def convert(obj):
        return [i['name'] for i in ast.literal_eval(obj)]

    def convert1(obj):
        return [i['name'] for i in ast.literal_eval(obj)[:3]]

    def director(obj):
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return [i['name']]
        return []

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert1)
    movies['crew'] = movies['crew'].apply(director)
    movies['overview'] = movies['overview'].fillna('').apply(lambda x: x.split())

    for col in ['genres', 'keywords', 'cast', 'crew']:
        movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new_df = movies[['movie_id', 'title', 'tags']].copy()  # << fix warning
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

    ps = PorterStemmer()
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join([ps.stem(i) for i in x.split()]))


    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()

    similarity = cosine_similarity(vectors)

    return new_df, similarity

movies_up, similarity = load_data()

# Recommender logic
def recommend(movie):
    movie = movie.strip()
    if movie not in movies_up['title'].values:
        return []
    movie_index = movies_up[movies_up['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies_up.iloc[i[0]].title for i in movie_list]

# UI
st.title("🎬 Simple Movie Recommender")
st.markdown("Type a movie name below and get similar movie recommendations.")

movie_input = st.selectbox(
    "Search or select a movie:",
    options=movies_up['title'].sort_values().values,
    index=0,
    placeholder="Type or choose a movie..."
)


if st.button("Recommend"):
    if movie_input:
        results = recommend(movie_input)
        if results:
            st.subheader("Recommended Movies:")
            for i, name in enumerate(results, 1):
                st.write(f"{i}. {name}")
        else:
            st.warning("Movie not found. Please try exact title from dataset.")
    else:
        st.warning("Please enter a movie name.")
