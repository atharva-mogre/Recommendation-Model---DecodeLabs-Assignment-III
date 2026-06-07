# Tech Stack Recommender System

This project is a streamlined, content-based recommendation system designed to match a user's skills and career goals to specific tech job roles. Built with Python, Pandas, and Scikit-learn, the script interactively ingests a minimum of three core skills from the user and calculates the most relevant career paths from a dataset (`raw_skills.csv`). It is an excellent demonstration of how advanced text vectorization and similarity metrics can be used to build effective recommendation engines.

At the core of this system is the TfidfVectorizer, which transforms the textual features of both the job roles and the user's profile into a numerical matrix. By applying TF-IDF weighting, it mathematically penalizes generic words and rewards highly specific terms. Cosine Similarity is then applied to these vectors to determine the mathematical alignment between the user's profile and the industry roles. The system truncates these similarity scores to present the top three best-matching career paths, preventing choice overload.

## Key Features
- **Advanced Text Vectorization**: Utilizes Scikit-learn's TfidfVectorizer to seamlessly convert skill profiles into numerically weighted vectors.
- **Cosine Similarity Matching**: Accurately computes the angular distance between the user's profile and available job roles to find the closest semantic matches.
- **Interactive Ingestion Pipeline**: Requires users to input at least three core skills directly into the terminal to bootstrap their profile and ensure data density.
- **Dynamic Dataset**: Reads from an external `raw_skills.csv` file, treating job roles as the "items" in the recommendation logic to output objective, ranked suggestions.

## Getting Started
To test the recommendation model, make sure you have Python, `pandas`, and `scikit-learn` installed in your environment, and ensure `raw_skills.csv` is present in the directory. Run the script directly from your terminal:
`python Recommendation_Model.py`
