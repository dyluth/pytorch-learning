import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import numpy as np
import json
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Data Preparation
class TweetDataset(Dataset):
    def __init__(self, tweets, labels, vocab, max_len):
        self.tweets = tweets
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        label = self.labels[idx]
        numericalized_tweet = [self.vocab[token] if token in self.vocab else self.vocab['<UNK>'] for token in tweet]
        padded_tweet = numericalized_tweet[:self.max_len] + [self.vocab['<PAD>']] * (self.max_len - len(numericalized_tweet))
        return torch.tensor(padded_tweet, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def build_vocab(tweets, min_freq=2):
    counter = Counter()
    for tweet in tweets:
        counter.update(tweet)
    filtered_tokens = [token for token, freq in counter.items() if freq >= min_freq]
    vocab = {token: idx + 2 for idx, token in enumerate(filtered_tokens)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

def tokenize(text):
    return text.split()

# Load JSON data
with open('/Users/cam/go/src/github.com/dyluth/votes/classifier/approvedResponses.json', 'r') as f:
    data = json.load(f)
    df = pd.DataFrame(data)

    # Modify ApprovedResponse values using pandas
    df['ApprovedResponse'] = df['ApprovedResponse'].str.split('the policy: ', n=1, expand=True)[1]

    # Handle NaN values
    df = df.dropna(subset=['ApprovedResponse'])

    # Convert the DataFrame back to a list of dictionaries if needed
    data = df.to_dict(orient='records')

tweets = [item['TweetMsg'] for item in data]
labels = [item['ApprovedResponse'] for item in data]

tweets_tokenized = [tokenize(tweet) for tweet in tweets]
labels_factorized = pd.factorize(labels)[0]

# Identify and handle singleton classes
label_counts = pd.Series(labels_factorized).value_counts()
singleton_labels = label_counts[label_counts == 1].index.tolist()

if singleton_labels:
    print(f"Warning: Singleton classes found: {singleton_labels}")
    # Remove rows with singleton labels
    indices_to_keep = ~pd.Series(labels_factorized).isin(singleton_labels)
    tweets_tokenized = [tweet for i, tweet in enumerate(tweets_tokenized) if indices_to_keep[i]]
    labels_factorized = labels_factorized[indices_to_keep]
    print(f"Removed {len(singleton_labels)} singleton classes.")

# Re-factorize after removing singleton classes
unique_labels = [labels[i] for i, keep in enumerate(indices_to_keep) if keep]
labels_factorized = pd.factorize(unique_labels)[0]

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tweets_flat = [" ".join(tweet) for tweet in tweets_tokenized]
tweets_tfidf = tfidf_vectorizer.fit_transform(tweets_flat).toarray()

# Oversampling using SMOTE
smote = SMOTE(random_state=42, k_neighbors=1)
tweets_resampled, labels_resampled = smote.fit_resample(tweets_tfidf, labels_factorized)

# Convert back to tokenized tweets (for vocabulary building)
tweets_tokenized_resampled = tfidf_vectorizer.inverse_transform(tweets_resampled)

# Build vocabulary from the resampled dataset
vocab = build_vocab(tweets_tokenized_resampled)
max_len = max(len(tweet) for tweet in tweets_tokenized_resampled)

# Split data (adjust test_size as needed)
X_train, X_test, y_train, y_test = train_test_split(tweets_tokenized_resampled, labels_resampled, test_size=0.2, random_state=42, stratify=labels_resampled)

# Create datasets and dataloaders
train_dataset = TweetDataset(X_train, y_train, vocab, max_len)
test_dataset = TweetDataset(X_test, y_test, vocab, max_len)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 2. Model Definition (Improved LSTM)
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_time_step = lstm_out[:, -1, :]
        x = self.fc(last_time_step)
        return x

vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 128
output_dim = len(np.unique(labels_resampled))

model = TextClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)

# 3. Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 50

for epoch in range(epochs):
    for tweets, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(tweets)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 4. Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for tweets, labels in test_loader:
        outputs = model(tweets)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.tolist())
        all_labels.extend(labels.tolist())

accuracy = accuracy_score(all_labels, all_preds)
print(f'Accuracy: {accuracy}')
print(classification_report(all_labels, all_preds))

# Print the list of original classifications
print("\nOriginal Classifications:")
unique_labels_list = np.unique(unique_labels)
for i, label in enumerate(unique_labels_list):
    print(f"{i}: {label}")