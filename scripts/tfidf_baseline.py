"""Lab 3A starter: TF-IDF + LinearSVC baseline."""

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score


def main():
    # Load the supplied feedback dataset
    df = pd.read_csv("data/raw/bayan_feedback.csv")

    # Use the supplied train and test splits
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    # Text features and topic labels
    X_train = train_df["text"]
    y_train = train_df["topic"]

    X_test = test_df["text"]
    y_test = test_df["topic"]

    # Convert text into TF-IDF features
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train the LinearSVC classifier
    model = LinearSVC()
    model.fit(X_train_tfidf, y_train)

    # Make predictions
    predictions = model.predict(X_test_tfidf)

    # Evaluate
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print("TF-IDF + LinearSVC Baseline")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()