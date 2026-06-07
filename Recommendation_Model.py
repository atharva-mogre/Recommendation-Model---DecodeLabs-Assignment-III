import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

data = {
    'Item': ['The Matrix', 'Inception', 'Interstellar', 'Toy Story', 'Finding Nemo', 'The Dark Knight', 'Avengers', 'The Conjuring', 'It', 'The Notebook', 'Titanic'],
    'Features': ['action sci-fi', 'action sci-fi thriller', 'sci-fi drama', 'animation comedy', 'animation family', 'action crime drama', 'action sci-fi superhero', 'horror thriller', 'horror', 'romance drama', 'romance drama']
}

df = pd.DataFrame(data)

user_interest = input("Enter keywords for your interest (e.g., action sci-fi): ")

cv = CountVectorizer()
count_matrix = cv.fit_transform(df['Features'].tolist() + [user_interest])

cosine_sim = cosine_similarity(count_matrix[-1], count_matrix[:-1])

scores = list(enumerate(cosine_sim[0]))
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

print("\nRecommended Items:")
found = False
for i in range(3):
    index = sorted_scores[i][0]
    if sorted_scores[i][1] > 0:
        print(f"- {df['Item'][index]} (Score: {sorted_scores[i][1]:.2f})")
        found = True

if not found:
    print("- No matching items found for your preferences. Try different keywords!")
