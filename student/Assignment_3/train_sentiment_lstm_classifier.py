import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import gensim
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

# ======================== Config ========================
MAX_SEQ_LEN = 32
EMBED_DIM = 300
BATCH_SIZE = 32
NUM_EPOCHS = 30
LR = 1e-3
SEED = 42
DATASET_NAME = 'financial_phrasebank'
SUBSET_NAME = 'sentences_50agree'
OUTPUT_DIR = 'outputs'

# ======================== Reproducibility ========================
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ======================== Device ========================
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {DEVICE}')

# ======================== Load Dataset ========================
import datasets
dataset = datasets.load_dataset(DATASET_NAME, SUBSET_NAME)
data = pd.DataFrame(dataset['train'])
data['text_label'] = data['label'].apply(lambda x: 'positive' if x==2 else 'neutral' if x==1 else 'negative')

# ======================== Load FastText Embeddings ========================
fasttext_path = 'cc.en.300.vec.gz'
print('Loading FastText embeddings...')
ft_model = gensim.models.KeyedVectors.load_word2vec_format(fasttext_path)
print('FastText embeddings loaded.')

# ======================== Helper Functions ========================
def tokenize_text(text):
    return word_tokenize(text.lower())

def sentence_to_vector(sentence, ft_model, max_len=MAX_SEQ_LEN, embed_dim=EMBED_DIM):
    tokens = tokenize_text(sentence)
    vecs = []
    for t in tokens[:max_len]:
        if t in ft_model:
            vecs.append(ft_model[t])
        else:
            vecs.append(np.zeros(embed_dim))
    while len(vecs) < max_len:
        vecs.append(np.zeros(embed_dim))
    return np.array(vecs, dtype=np.float32)

# ======================== Precompute Sentence Vectors ========================
print('Encoding sentences into FastText vectors...')
X_vectors = np.stack([sentence_to_vector(s, ft_model) for s in tqdm(data['sentence'])])
y = data['label'].values
print(f'X_vectors shape: {X_vectors.shape}, y shape: {y.shape}')

# ======================== Stratified Split ========================
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_vectors, y, test_size=0.15, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=SEED
)
print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

# ======================== PyTorch Dataset ========================
class FastTextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = FastTextDataset(X_train, y_train)
val_dataset = FastTextDataset(X_val, y_val)
test_dataset = FastTextDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======================== LSTM Model ========================
class LSTMSentimentClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

hidden_dim = 128
num_layers = 2
num_classes = len(np.unique(y))
model = LSTMSentimentClassifier(EMBED_DIM, hidden_dim, num_layers, num_classes).to(DEVICE)

# ======================== Loss, Optimizer, Scheduler ========================
counts = [604, 2879, 1363]
class_weights = 1. / torch.tensor(counts, dtype=torch.float)
class_weights /= class_weights.sum()
class_weights = class_weights.to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# ======================== Training Loop ========================
os.makedirs(OUTPUT_DIR, exist_ok=True)
best_val_f1 = 0.0
train_loss_history, val_loss_history = [], []
train_f1_history, val_f1_history = [], []
train_acc_history, val_acc_history = [], []

for epoch in range(NUM_EPOCHS):
    print(f'\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===')
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for X_batch, y_batch in tqdm(train_loader, leave=False):
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    train_loss = running_loss / len(train_loader.dataset)
    train_f1 = f1_score(all_labels, all_preds, average='macro')
    train_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    train_loss_history.append(train_loss)
    train_f1_history.append(train_f1)
    train_acc_history.append(train_acc)
    print(f'Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Train Acc: {train_acc:.4f}')

    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    val_loss /= len(val_loader.dataset)
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    val_loss_history.append(val_loss)
    val_f1_history.append(val_f1)
    val_acc_history.append(val_acc)
    print(f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}')
    scheduler.step(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'outputs/best_lstm_model.pth')
        print(f'>>> Saved new best model (Val F1: {best_val_f1:.4f})')

plt.figure(figsize=(12, 15))
plt.subplot(3, 1, 1)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(train_f1_history, label='Train F1')
plt.plot(val_f1_history, label='Val F1')
plt.title('F1 Macro Score Curve')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)
plt.subplot(3, 1, 3)
plt.plot(train_acc_history, label='Train Acc')
plt.plot(val_acc_history, label='Val Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/lstm_f1_learning_curves.png')
plt.show()
print("Learning curves saved as 'outputs/lstm_f1_learning_curves.png'.")


plt.figure(figsize=(8, 6))
plt.plot(train_acc_history, label='Train Acc')
plt.plot(val_acc_history, label='Val Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/lstm_accuracy_learning_curve.png')
plt.show()
print("Accuracy curve saved as 'outputs/lstm_accuracy_learning_curve.png'.")

model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_lstm_model.pth')))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        logits = model(X_batch)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

test_acc = (np.array(all_preds) == np.array(all_labels)).mean()
test_f1_macro = f1_score(all_labels, all_preds, average='macro')
test_f1_weighted = f1_score(all_labels, all_preds, average='weighted')
print('\n' + '='*50)
print(f'Final Test Accuracy: {test_acc:.4f}')
print(f'Test F1 Macro: {test_f1_macro:.4f}')
print(f'Test F1 Weighted: {test_f1_weighted:.4f}')
print('='*50 + '\n')

class_names = ['Negative (0)', 'Neutral (1)', 'Positive (2)']
print('Classification Report:')
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.savefig('outputs/lstm_confusion_matrix.png')
plt.show()
print("Confusion matrix saved as 'outputs/lstm_confusion_matrix.png'.")

print('\nPer-class F1 Scores:')
for i, name in enumerate(class_names):
    class_f1 = f1_score(all_labels, all_preds, labels=[i], average='macro')
    print(f'{name}: {class_f1:.4f}')

print('\nScript Complete.')