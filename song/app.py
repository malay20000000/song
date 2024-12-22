import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import time

# Load the dataset
@st.cache_data
def load_data():
    genres_df = pd.read_csv("genres_v2.csv")
")
    return genres_df

# Preprocess the dataset
@st.cache_data
def preprocess_data(genres_df):
    audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                      'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    
    genres_df_cleaned = genres_df.dropna(subset=audio_features)
    genres_df_cleaned = genres_df_cleaned.drop_duplicates(subset='song_name', keep='first')
    
    scaler = MinMaxScaler()
    genres_df_cleaned[audio_features] = scaler.fit_transform(genres_df_cleaned[audio_features])
    
    pca = PCA(n_components=0.95)
    reduced_features = pca.fit_transform(genres_df_cleaned[audio_features])
    
    return genres_df_cleaned, reduced_features

# Train the NearestNeighbors model
@st.cache_resource
def train_model(reduced_features):
    knn_model = NearestNeighbors(metric='cosine', algorithm='auto')
    knn_model.fit(reduced_features)
    return knn_model

# Define the recommendation function
def recommend_songs(song_name, genres_df_cleaned, reduced_features, knn_model, top_n=5):
    song_name_to_idx = {name: idx for idx, name in enumerate(genres_df_cleaned['song_name'])}
    
    if song_name not in song_name_to_idx:
        return f"Song '{song_name}' not found in the dataset."
    
    song_idx = song_name_to_idx[song_name]
    distances, indices = knn_model.kneighbors([reduced_features[song_idx]], n_neighbors=top_n+1)
    recommended_indices = [i for i in indices[0] if i != song_idx]
    similar_songs = genres_df_cleaned.iloc[recommended_indices[:top_n]]
    return similar_songs

# Main Streamlit app
def main():
    # Set page configuration
    st.set_page_config(
        page_title="🎶 Song Recommender",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for animations and style
    st.markdown(
        """
        <style>
        body {
            font-family: 'Helvetica', sans-serif;
            background: linear-gradient(to bottom right, #1f4037, #99f2c8);
            color: white;
        }
        .stSelectbox>div>div>select {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        .stButton>button {
            background-color: #f64f59 !important;
            border-radius: 8px !important;
            color: white !important;
            font-size: 18px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #12c2e9 !important;
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        .fade-in {
            animation: fadeIn 2s ease-in-out;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header animation
    st.markdown(
        """
        <div class="fade-in" style="text-align: center;">
            <h1 style="font-size: 3em; margin-bottom: -10px;">🎧 Song Recommendation System 🎧</h1>
            <p style="font-size: 1.2em;">Discover your next favorite song effortlessly!</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load and preprocess data
    genres_df = load_data()
    genres_df_cleaned, reduced_features = preprocess_data(genres_df)
    knn_model = train_model(reduced_features)

    # Song selection dropdown
    song_name = st.selectbox(
        "Select a song from the list:",
        options=genres_df_cleaned['song_name'].unique(),
    )

    # Recommendation button
    if st.button("🎵 Get Recommendations"):
        with st.spinner("Fetching recommendations..."):
            time.sleep(1.5)  # Simulate loading

        recommendations = recommend_songs(song_name, genres_df_cleaned, reduced_features, knn_model)
        
        st.success(f"Here are the recommendations for '{song_name}':")
        
        # Display recommendations with animations
        for idx, row in recommendations.iterrows():
            st.markdown(
                f"""
                <div class="fade-in" style="margin-bottom: 20px;">
                    <h3>🎶 {row['song_name']}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Add song image if available
            if 'image_url' in row and row['image_url']:
                st.image(row['image_url'], width=200, caption=row['song_name'])

    # Footer
    st.markdown(
        """
        <div style="text-align: center; margin-top: 50px; font-size: 1em;">
            <p>Made with ❤️ </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
