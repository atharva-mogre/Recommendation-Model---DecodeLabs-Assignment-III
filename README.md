# Content-Based Movie Recommendation System

This project is a streamlined, content-based recommendation system designed to suggest movies based on user preferences. Built with Python, Pandas, and Scikit-learn, the script takes a natural language input from the user—such as preferred genres or themes—and calculates the most relevant movie suggestions from an internal dataset. It is an excellent demonstration of how text vectorization and similarity metrics can be used to build effective recommendation engines.

At the core of this system is the CountVectorizer, which transforms the textual features of both the movies and the user's query into a numerical matrix. By applying Cosine Similarity to these vectors, the algorithm can mathematically determine how closely a movie's features match the user's desired keywords. The system then ranks these similarity scores to present the top three best-matching movies, providing a seamless and personalized discovery experience.

## Key Features
- **Text Vectorization**: Utilizes Scikit-learn's CountVectorizer to seamlessly convert genres and themes into actionable numerical data.
- **Cosine Similarity Matching**: Accurately computes the mathematical distance between user queries and available movies to find the best possible matches.
- **Interactive User Prompt**: Allows users to input their own custom keywords (e.g., "action sci-fi") directly into the terminal to receive dynamic recommendations.
- **Pre-loaded Dataset**: Includes a built-in selection of popular movies and their associated tags, allowing the system to run out of the box without external databases.

## Getting Started
To test the recommendation model, make sure you have Python, `pandas`, and `scikit-learn` installed in your environment. Run the script directly from your terminal:
`python Recommendation_Model.py`
