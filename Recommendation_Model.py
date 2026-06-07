import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

print("Welcome to the Tech Stack Recommender!")
print("Please enter a minimum of three skills or interests to bootstrap your profile.")
skill_1 = input("Skill 1 (e.g., Python): ").strip()
skill_2 = input("Skill 2 (e.g., Cloud Computing): ").strip()
skill_3 = input("Skill 3 (e.g., Automation): ").strip()

user_profile = f"{skill_1} {skill_2} {skill_3}"

try:
    df = pd.read_csv("raw_skills.csv")
except FileNotFoundError:
    print("Error: raw_skills.csv not found.")
    exit(1)

tfidf = TfidfVectorizer()
all_features = df['Skills'].tolist() + [user_profile]
tfidf_matrix = tfidf.fit_transform(all_features)

user_vector = tfidf_matrix[-1]
job_vectors = tfidf_matrix[:-1]
cosine_sim = cosine_similarity(user_vector, job_vectors)

scores = list(enumerate(cosine_sim[0]))
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

print("\nTop 3 Recommended Career Paths:")
found = False
for i in range(min(3, len(sorted_scores))):
    index = sorted_scores[i][0]
    score = sorted_scores[i][1]
    
    if score > 0:
        print(f"- {df['Job_Role'][index]} (Match Score: {score:.2f})")
        found = True

if not found:
    print("- No matching career paths found for your skills. Keep learning!")
