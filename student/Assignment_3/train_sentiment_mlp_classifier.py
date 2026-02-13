import os
import numpy as np
import pandas as pd
import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gensim.downloader as api

# ===================== Settings =====================
SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 50  # train for >=30 epochs
LEARNING_RATE = 1e-3
MAX_LEN = 300  # FastText embedding size
EARLY_STOPPING_PATIENCE = 5
OUTPUT_DIR = 'outputs'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== Reproducibility =====================
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ===================== Load Data =====================
dataset = datasets.load_dataset('financial_phrasebank', 'sentences_50agree', trust_remote_code=True)
print("Dataset loaded. Example:", dataset['train'][0])

print("\n========== Preparing DataFrame ==========")
data = pd.DataFrame(dataset['train'])
data['text_label'] = data['label'].apply(lambda x: 'positive' if x == 2 else 'neutral' if x == 1 else 'negative')
print(f"DataFrame shape: {data.shape}")

# ===================== Load FastText =====================
print("\n========== Loading FastText ==========")
ft_model = api.load('fasttext-wiki-news-subwords-300')  # 300-dim embeddings

# ===================== Sentence Embedding =====================
def sentence_to_vec(sentence, model, dim=300):
    tokens = sentence.lower().split()
    vecs = [model[word] for word in tokens if word in model]
    if len(vecs) == 0:
        return np.zeros(dim)
    else:
        return np.mean(vecs, axis=0)

print("\n========== Computing Sentence Embeddings ==========")
data['embedding'] = data['sentence'].apply(lambda x: sentence_to_vec(x, ft_model))

X = np.stack(data['embedding'].values)
y = data['label'].values

# ===================== Train/Val/Test Split =====================
print("\n========== Splitting Data ==========")
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=SEED
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ===================== PyTorch Dataset =====================
class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = EmbeddingDataset(X_train, y_train)
val_dataset = EmbeddingDataset(X_val, y_val)
test_dataset = EmbeddingDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===================== MLP Model =====================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=256, num_classes=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_classes)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLPClassifier().to(device)

# ===================== Loss & Optimizer =====================
class_counts = np.bincount(y_train)
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ===================== Training Loop =====================
best_val_f1 = 0.0
patience_counter = 0

train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []
train_f1_history = []
val_f1_history = []

print("\n========== Starting Training ==========")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    epoch_train_loss = running_loss / len(train_loader.dataset)
    epoch_train_acc = accuracy_score(all_labels, all_preds)
    epoch_train_f1 = f1_score(all_labels, all_preds, average='macro')

    # Validation
    model.eval()
    val_loss = 0.0
    val_preds_all = []
    val_labels_all = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            val_preds_all.extend(preds.cpu().numpy())
            val_labels_all.extend(y_batch.cpu().numpy())
    epoch_val_loss = val_loss / len(val_loader.dataset)
    epoch_val_acc = accuracy_score(val_labels_all, val_preds_all)
    epoch_val_f1 = f1_score(val_labels_all, val_preds_all, average='macro')

    # Track metrics
    train_loss_history.append(epoch_train_loss)
    val_loss_history.append(epoch_val_loss)
    train_acc_history.append(epoch_train_acc)
    val_acc_history.append(epoch_val_acc)
    train_f1_history.append(epoch_train_f1)
    val_f1_history.append(epoch_val_f1)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f} | Train F1: {epoch_train_f1:.4f}, Val F1: {epoch_val_f1:.4f}")

    # Early stopping and save best model
    if epoch_val_f1 > best_val_f1:
        best_val_f1 = epoch_val_f1
        torch.save(model.state_dict(), 'outputs/best_mlp_model.pth')
        patience_counter = 0
        print(f"--> Saved new best model (Val F1: {best_val_f1:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE and epoch >= 30:
            print("Early stopping triggered.")
            break

# ===================== Plot Learning Curves =====================
plt.figure(figsize=(12, 15))
plt.subplot(3,1,1)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(train_f1_history, label='Train F1')
plt.plot(val_f1_history, label='Val F1')
plt.title('F1 Macro Score Curve')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(train_acc_history, label='Train Acc')
plt.plot(val_acc_history, label='Val Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/mlp_f1_learning_curves.png')
plt.show()
print("Learning curves saved as 'outputs/mlp_f1_learning_curves.png'.")

plt.figure(figsize=(8, 6))
plt.plot(train_acc_history, label='Train Acc')
plt.plot(val_acc_history, label='Val Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/mlp_accuracy_learning_curve.png')
plt.show()
print("Accuracy curve saved as 'outputs/mlp_accuracy_learning_curve.png'.")


# ===================== Test Evaluation =====================
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_mlp_model.pth')))
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

test_acc = accuracy_score(all_labels, all_preds)
test_f1_macro = f1_score(all_labels, all_preds, average='macro')
test_f1_weighted = f1_score(all_labels, all_preds, average='weighted')

print('\n' + '='*50)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1 Macro: {test_f1_macro:.4f}")
print(f"Test F1 Weighted: {test_f1_weighted:.4f}")
print('='*50 + '\n')

class_names = ['Negative (0)', 'Neutral (1)', 'Positive (2)']
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.savefig('outputs/mlp_confusion_matrix.png')
plt.show()
print("Confusion matrix saved as 'outputs/mlp_confusion_matrix.png'.")
print("\nPer-class F1 Scores:")
for i, name in enumerate(class_names):
    class_f1 = f1_score(all_labels, all_preds, labels=[i], average='macro')
    print(f"{name}: {class_f1:.4f}")
