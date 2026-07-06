# Spam and Ham Detection using Naive Bayes
import io
import os
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:

            filepath = os.path.join(root, filename)

            with io.open(filepath, 'r', encoding='latin1', errors='ignore') as f:
                message = f.read()

            yield filepath, message


# Create a DataFrame from a directory

def dataFrameFromDirectory(path, classification):

    rows = []
    index = []

    for filename, message in readFiles(path):

        rows.append({
            'message': message,
            'class': classification
        })

        index.append(filename)

    return DataFrame(rows, index=index)


# Create an empty DataFrame

data = DataFrame({'message': [], 'class': []})


# Load Spam Emails

data = pd.concat([
    data,
    dataFrameFromDirectory(
        r"C:\Users\kesha\OneDrive\Documents\spamdetector\emails\spam",
        "spam"
    )
])


# Load Ham Emails

data = pd.concat([
    data,
    dataFrameFromDirectory(
        r"C:\Users\kesha\OneDrive\Documents\spamdetector\emails\ham",
        "ham"
    )
])


# Display Dataset

print("Dataset Loaded Successfully!\n")
print(data.head())
print("\nDataset Shape :", data.shape)
print("\nClass Distribution")
print(data["class"].value_counts())

# Convert Text into Numerical Features

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

# Train Naive Bayes Classifier

classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)
print("\nModel Trained Successfully!")

# Test Examples

examples = [
    'Free Viagra now!',
    'Hi Drake, how about we go to the party?',
    'Congratulations! You won ₹50,000.',
    'Claim your free gift now.',
    'Hi Keshav, are we meeting tomorrow?'
]

example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
print("\nPredictions:\n")
for message, prediction in zip(examples, predictions):

    print("-" * 50)

    print("Message:")
    print(message)

    print("\nPrediction:")
    print(prediction)

print("-" * 50)