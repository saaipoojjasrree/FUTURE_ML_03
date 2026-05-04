import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample resumes
resumes = [
    "Python developer with experience in machine learning and data analysis",
    "Web developer skilled in HTML CSS JavaScript",
    "Data scientist with Python machine learning and deep learning experience",
    "Software engineer with Java and backend development"
]

# Job description
job_description = "Looking for a data scientist with machine learning and python skills"

# Convert text to vectors
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(resumes + [job_description])

# Compute similarity
similarity = cosine_similarity(vectors[-1], vectors[:-1])

# Rank resumes
scores = similarity[0]

# Display results
for i, score in enumerate(scores):
    print(f"Resume {i+1} Match Score: {score:.2f}")

# Best candidate
best_match = scores.argmax()
print("\nBest Resume is Resume", best_match + 1)