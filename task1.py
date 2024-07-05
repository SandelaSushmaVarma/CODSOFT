# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Example dataset (replace with your dataset)
data = {
    'plot_summary': [
        "In a world where dinosaurs roam freely, a group of humans must find a way to survive.",
        "A young wizard battles against dark forces to save the world from impending doom.",
        "A romantic comedy about two strangers who meet on a train and fall in love against all odds.",
        "A team of explorers discover a clue to the origins of mankind on Earth, leading them on a thrilling journey.",
        "A group of friends embark on an epic quest to defeat an ancient evil threatening their land."
    ],
    'genre': ['Action', 'Fantasy', 'Romance', 'Sci-Fi', 'Adventure']
}

# Create a DataFrame from the example data
df = pd.DataFrame(data)

# Separate features (plot summaries) and target (genre labels)
X = df['plot_summary']
y = df['genre']

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Consider top 1000 features

# Fit and transform the training data
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Initialize Logistic Regression classifier
clf = LogisticRegression(max_iter=1000)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Example of predicting genres for new plot summaries
new_plot_summaries = [
    "A group of scientists discover a portal to another dimension.",
    "A chef opens a new restaurant and finds love along the way."
]

# Transform new plot summaries using TF-IDF vectorizer
new_plot_summaries_tfidf = tfidf_vectorizer.transform(new_plot_summaries)

# Predict genres for new plot summaries
predicted_genres = clf.predict(new_plot_summaries_tfidf)
print("\nPredicted genres for new plot summaries:")
for summary, genre in zip(new_plot_summaries, predicted_genres):
    print(f"Plot Summary: {summary} --> Predicted Genre: {genre}")
