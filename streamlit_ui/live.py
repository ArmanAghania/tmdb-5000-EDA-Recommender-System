import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity

# Load the data and models
df = pd.read_pickle("movies_df.pkl")
cosine_sim2 = np.load("cosine_sim2.npy")
count = load("count_vectorizer.joblib")  # If you're planning to update your dataset
indices = pd.Series(df.index, index=df['title'])

def get_recommendations(title, cosine_sim=cosine_sim2):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    movie_similarity = [i[1] for i in sim_scores]

    return pd.DataFrame(zip(df['title'].iloc[movie_indices], movie_similarity), columns=["title", "similarity"])

# Streamlit app
st.title('Movie Recommender System')

# Movie selection
selected_movie = st.selectbox('Select a movie:', df['title'].values)

if st.button('Recommend'):
    recommendations = get_recommendations(selected_movie, cosine_sim2)
    for idx, row in recommendations.iterrows():
        st.write(f"{row['title']}: Similarity - {row['similarity']:.3f}")
