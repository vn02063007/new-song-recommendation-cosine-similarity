import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Custom CSS for background image
st.markdown(
    """
<style>
    body {
        background-image: url("https://images.unsplash.com/photo-1524678606370-a47ad25cb82a?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2669&q=80");
        background-size: cover;
        background-repeat: no-repeat;
    }
</style>
""",
    unsafe_allow_html=True,
)


# 1. Import database file
data = pd.read_csv('database.csv')

# Reduce data size to 10,000 rows and reset the index
data = data.sample(n=30000, random_state=42).reset_index(drop=True)

# Combine relevant features into a single string for each song
data['combined_features'] = data[['duration_ms', 'explicit', 'danceability', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'energy', 'liveness', 'valence', 'tempo']].apply(lambda x: ' '.join(x.astype(str)), axis=1)


# Calculate the cosine similarity matrix
vectorizer = CountVectorizer()
count_matrix = vectorizer.fit_transform(data['combined_features'])
cosine_sim = cosine_similarity(count_matrix)

def get_song_index(title):
    return data[data['name'] == title].index.values[0]

def recommend_songs(title, cosine_sim=cosine_sim):
    song_index = get_song_index(title)
    input_artist = data.iloc[song_index]['artists']
    
    similarity_scores = list(enumerate(cosine_sim[song_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Filter out songs from the same artist and get the indices of the top 5 similar songs
    similar_song_indices = []
    for score in similarity_scores:
        if data.iloc[score[0]]['artists'] != input_artist:
            similar_song_indices.append(score[0])
            if len(similar_song_indices) >= 5:
                break

    # Get the details of the top 5 similar songs
    recommended_songs = data.iloc[similar_song_indices]
    return recommended_songs


# Streamlit web app
st.title("🎶 Song Recommender 🎉")

# Autocomplete input box for song title or artist name
user_input = st.selectbox(
    "Enter a song title or artist name:",
    options=[""] + list(data['name'].unique()) + list(data['artists'].unique()),
)

# Submit button
submit_button = st.button("Get Recommendations")

if submit_button:
    # Check if the input is a song title
    if user_input in data['name'].values:
        st.subheader("Top 5 similar songs:")
        recommended_songs = recommend_songs(user_input)
        for index, song in recommended_songs.iterrows():
            st.write(f"{song['name']} | {song['artists']}")

    # Check if the input is an artist name
    elif user_input in data['artists'].values:
        st.subheader("Top 5 songs by this artist:")
        top_songs = data[data['artists'] == user_input].sort_values(by='popularity', ascending=False).head(5)
        for index, song in top_songs.iterrows():
            st.write(f"{song['name']} | {song['artists']}")

    else:
        st.error("Song or artist not found in the database. Here are 5 random song recommendations:")
        random_songs = data.sample(n=5, random_state=42)
        for index, song in random_songs.iterrows():
            st.write(f"{song['name']} | {song['artists']}")
