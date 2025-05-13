# 🎬 Movie Recommendation System

![Movie Recommendation System Banner](https://images.unsplash.com/photo-1542204165-65bf26472b9b?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&h=400&q=80)

## Table of Contents
- [Project Overview](#project-overview)
- [Demo](#demo)
- [Features](#features)
- [Technical Details](#technical-details)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Implementation Walkthrough](#implementation-walkthrough)
- [How the Recommendation Works](#how-the-recommendation-works)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Future Improvements](#future-improvements)
- [Resources](#resources)
- [Credits](#credits)

## Project Overview

This Movie Recommendation System is a content-based filtering application that suggests movies similar to a selected movie based on its content features. Unlike collaborative filtering systems that rely on user ratings and behaviors, this system analyzes movie metadata such as genres, keywords, cast, crew, and plot overview to find movies with similar content characteristics.

The system processes and combines various textual features from movies into tags, vectorizes these tags, and uses cosine similarity to identify movies with similar content profiles. This approach allows for personalized movie recommendations without requiring user interaction history.

## Demo

The application is built using Streamlit, providing an intuitive and interactive user interface where users can:
- Select a movie from a dropdown menu
- Click the "Recommend" button to get 5 similar movie recommendations
- View the recommended movies along with their posters

## Features

- **Content-Based Filtering**: Recommends movies based on similarity of content rather than user preferences
- **Interactive UI**: Easy-to-use Streamlit interface for selecting movies and viewing recommendations
- **Visual Recommendations**: Displays movie posters alongside titles for better user experience
- **Fast Response**: Pre-computed similarity matrix enables quick recommendation generation
- **API Integration**: Fetches movie posters from TMDB API to enhance visual appeal

## Technical Details

### Technologies Used

- **Python 3.x**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **NLTK**: Natural language processing for text preprocessing
- **Scikit-learn**: Machine learning tools for vectorization and similarity calculation
- **Streamlit**: Web application framework for the user interface
- **Requests**: HTTP library for API calls
- **Pickle**: For serializing Python objects

### Libraries and Packages

- pandas, numpy: Data handling
- nltk (PorterStemmer): Text preprocessing and stemming
- scikit-learn (CountVectorizer, cosine_similarity): Feature extraction and similarity calculation
- streamlit: Frontend UI development
- requests: API integration for movie posters
- pickle: Model persistence

## Dataset

The project utilizes the TMDB 5000 Movie Dataset which contains:

- **tmdb_5000_movies.csv**: Movie metadata including titles, genres, keywords, overview, etc.
- **tmdb_5000_credits.csv**: Cast and crew information for the movies

Key features used from the dataset:
- Movie titles and IDs
- Genres
- Keywords
- Plot overviews
- Cast information (top 3 actors)
- Director information

## Project Architecture

The project follows a simple yet effective architecture:

```
             ┌───────────────┐
             │    Dataset    │
             │ TMDB 5000     │
             └───────┬───────┘
                     │
                     ▼
┌──────────────────────────────────────┐
│       Data Processing Pipeline       │
│                                      │
│  ┌─────────────┐    ┌─────────────┐  │
│  │ Text        │    │ Feature     │  │
│  │ Processing  │───►│ Engineering │  │
│  └─────────────┘    └─────────────┘  │
│                                      │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│        Similarity Calculation        │
│                                      │
│  ┌─────────────┐    ┌─────────────┐  │
│  │ Vectorize   │    │ Cosine      │  │
│  │ Tags        │───►│ Similarity  │  │
│  └─────────────┘    └─────────────┘  │
│                                      │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│        Streamlit Application         │
│                                      │
│  ┌─────────────┐    ┌─────────────┐  │
│  │ User        │    │ Display     │  │
│  │ Input       │───►│ Results     │  │
│  └─────────────┘    └─────────────┘  │
│                                      │
└──────────────────────────────────────┘
```

## Implementation Walkthrough

### 1. Data Preprocessing and Feature Engineering

The system processes the raw movie data through several steps:

- **Data Loading**: Reading and merging movie and credits datasets
- **Feature Selection**: Extracting relevant features (movie_id, title, overview, genres, keywords, cast, crew)
- **Data Cleaning**: Handling missing values and duplicates
- **Format Conversion**: Converting JSON strings to Python lists for genres, keywords, cast, and crew
- **Feature Extraction**: 
  - Extracting the top 3 actors from cast
  - Extracting the director's name from crew
- **Tag Creation**: Combining overview, genres, keywords, cast, and crew into a single "tags" feature
- **Text Processing**:
  - Tokenizing text (splitting into words)
  - Converting to lowercase
  - Stemming (reducing words to their root form)
- **Vectorization**: Converting processed tags into numerical vectors using CountVectorizer with stop word removal

```python
# Sample code for tag creation and processing
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df['tags'] = new_df['tags'].apply(stem)
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
```

### 2. Similarity Calculation

The system computes similarity between movies using:

- **Count Vectorization**: Converting text data into a sparse matrix of token counts
- **Cosine Similarity**: Calculating the cosine of the angle between movie vectors to measure similarity

```python
# Vectorization and similarity calculation
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)
```

### 3. Recommendation Function

The core recommendation function works by:
- Finding the index of the selected movie
- Retrieving the similarity scores of all movies with the selected movie
- Sorting the movies based on similarity scores
- Returning the top 5 most similar movies (excluding the selected movie itself)

```python
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    
    recommended_movies = []
    recommended_movies_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        # Fetch poster from API
        recommended_movies_posters.append(fetch_poster(movie_id))
    return recommended_movies, recommended_movies_posters
```

### 4. Web Application

The Streamlit web application:
- Loads the preprocessed movie data and similarity matrix
- Provides a dropdown to select a movie
- Implements a recommendation button
- Displays the recommended movies with their posters in a grid layout
- Fetches movie posters from TMDB API

## How the Recommendation Works

Let's understand how the content-based recommendation system works with an example:

1. **Movie Selection**: User selects "Avatar" from the dropdown
2. **Feature Analysis**: System looks at Avatar's processed tags, which include:
   - Genres: Action, Adventure, Fantasy, Science Fiction
   - Keywords: culture clash, future, space war, space colony, society, space travel, etc.
   - Cast: Sam Worthington, Zoe Saldana, Sigourney Weaver
   - Director: James Cameron
   - Overview: Text about a paraplegic marine on Pandora

3. **Similarity Calculation**: System compares these features with all other movies using vector similarity
4. **Results**: Returns top 5 movies with the highest similarity scores to Avatar

For example, when "Avatar" is selected, the system recommends:
- Aliens vs Predator: Requiem
- Aliens
- Falcon Rising
- Independence Day
- Titan A.E.

These recommendations share content similarities with Avatar, such as science fiction themes, alien encounters, or action/adventure elements.

## Installation & Setup

### Prerequisites
- Python 3.x
- Pip package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Arshnoor-Singh-Sohi/Movie-Recommendation-System.git
   cd Movie-Recommendation-System
   ```

2. **Create and activate a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and prepare the dataset**
   - The processed data files are included in the repository as `movies_dict.pkl` and `similarity.pkl`
   - If you want to process the raw data yourself, download the TMDB 5000 dataset and run the Jupyter notebook

5. **Get TMDB API Key**
   - Create an account on [The Movie Database](https://www.themoviedb.org/)
   - Generate an API key from your account settings
   - Replace the API key in `app.py` (if needed)

6. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

7. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`

## Usage Guide

1. **Select a Movie**: Use the dropdown menu to choose a movie you enjoy
2. **Get Recommendations**: Click the "Recommend" button
3. **Explore Results**: Browse the 5 recommended movies shown with their posters
4. **Try Different Movies**: Select different movies to discover various recommendations

## Future Improvements

- **Hybrid Recommendation System**: Combine content-based filtering with collaborative filtering for better recommendations
- **User Profiles**: Allow users to rate movies and create personalized recommendation profiles
- **More Movie Information**: Add additional details like release year, runtime, and ratings
- **Larger Dataset**: Expand to include more recent movies and a larger catalog
- **Advanced NLP**: Implement more sophisticated text processing techniques like TF-IDF or word embeddings
- **Filtering Options**: Add filters for genres, release years, or ratings
- **User Feedback Loop**: Incorporate user feedback on recommendations to improve the system

## Resources

- [TMDB API Documentation](https://developers.themoviedb.org/3/getting-started/introduction)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Content-Based Filtering](https://developers.google.com/machine-learning/recommendation/content-based/basics)
- [Cosine Similarity Explained](https://www.machinelearningplus.com/nlp/cosine-similarity/)

## Credits

- **Dataset**: The Movie Database (TMDB) 5000 Movie Dataset
- **API**: The Movie Database (TMDB) API for movie posters
- **Built With**: Python, Pandas, NLTK, Scikit-learn, Streamlit

---

Built with ❤️ by [Arshnoor Singh Sohi](https://github.com/Arshnoor-Singh-Sohi)
