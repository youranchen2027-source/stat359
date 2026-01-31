import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 512  # change it to fit your memory constraints, e.g., 256, 128 if you run out of memory
EPOCHS = 5
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, skipgram_df):
        self.centers = torch.LongTensor(skipgram_df['center'].values)
        self.contexts = torch.LongTensor(skipgram_df['context'].values)

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        return self.centers[idx], self.contexts[idx]

# Simple Skip-gram Module
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)

        nn.init.xavier_uniform_(self.input_embeddings.weight)
        nn.init.xavier_uniform_(self.output_embeddings.weight)

    def forward(self, center, context):
        center_emb = self.input_embeddings(center)      # (B, D)
        context_emb = self.output_embeddings(context)   # (B, D)
        score = torch.sum(center_emb * context_emb, dim=1)
        return score

    def get_embeddings(self):
        return self.input_embeddings.weight.detach().cpu().numpy()
    


# Load processed data
with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)

skipgram_df = data['skipgram_df']
word2idx = data['word2idx']
idx2word = data['idx2word']
counter = data['counter']

vocab_size = len(word2idx)

# Precompute negative sampling distribution below
word_counts = torch.zeros(vocab_size)
for word, idx in word2idx.items():
    word_counts[idx] = counter.get(word, 0)

neg_sampling_dist = word_counts.pow(0.75)
neg_sampling_dist /= neg_sampling_dist.sum()

# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")



# Dataset and DataLoader
dataset = SkipGramDataset(skipgram_df)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)


# Model, Loss, Optimizer
model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def make_targets(size, value, device):
    return torch.full((size,), value, device=device)

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0.0
    model.train()

    for center, context in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        center = center.to(device)
        context = context.to(device)
        batch_size = center.size(0)

        optimizer.zero_grad()

        # ----- Positive samples -----
        pos_scores = model(center, context)
        pos_labels = make_targets(batch_size, 1.0, device)
        pos_loss = criterion(pos_scores, pos_labels)

        # ----- Negative samples -----
        neg_context = torch.multinomial(
            neg_sampling_dist,
            batch_size * NEGATIVE_SAMPLES,
            replacement=True
        ).to(device)

        neg_center = center.repeat_interleave(NEGATIVE_SAMPLES)

        neg_scores = model(neg_center, neg_context)
        neg_labels = make_targets(batch_size * NEGATIVE_SAMPLES, 0.0, device)
        neg_loss = criterion(neg_scores, neg_labels)

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")



# Save embeddings and mappings
embeddings = model.get_embeddings()
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': data['word2idx'], 'idx2word': data['idx2word']}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
