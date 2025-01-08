# Import necessary libraries
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AlbertForSequenceClassification, AlbertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.tensorboard import SummaryWriter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
import numpy as np
import time
import os
import logging

# Set up logging
logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Load the training dataset
train_df = pd.read_csv('train.csv')
train_df['passage_length'] = train_df['passage'].apply(lambda x: len(x.split()))

# Create directory for saving plots
if not os.path.exists('plots'):
    os.makedirs('plots')

# Plot distribution of passage lengths
plt.hist(train_df['passage_length'], bins=50, color='blue', alpha=0.7)
plt.xlabel('Passage Length (words)')
plt.ylabel('Frequency')
plt.title('Distribution of Passage Lengths')
plt.savefig('plots/passage_length_distribution.png')

# Plot class distribution
sns.countplot(x='y', data=train_df)
plt.xlabel('Bias Class')
plt.ylabel('Frequency')
plt.title('Class Distribution in Training Data')
plt.savefig('plots/class_distribution.png')

# Handle class imbalance using SMOTE with TF-IDF
X = train_df['passage']
y = train_df['y']

# Convert text data to numerical format using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Handle class imbalance with SMOTE or skip if data is too small
use_smote = len(y.unique()) > 1 and y.value_counts().min() > 1

if use_smote:
    # Apply SMOTE to balance the classes
    smote = SMOTE(random_state=42, k_neighbors=1)  # Reduce the number of neighbors to avoid the error
    X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)
    
    # Reconstructing the resampled dataframe using inverse transformation
    passages_resampled = tfidf.inverse_transform(X_resampled)
    train_df_resampled = pd.DataFrame({
        'passage': [' '.join(row) for row in passages_resampled], 
        'y': y_resampled
    })
else:
    print("Skipping SMOTE for small dataset.")
    train_df_resampled = train_df.copy()

# Define a custom Dataset class for PyTorch
class GenderBiasDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, is_test=False):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        passage = str(self.dataframe.iloc[idx]['passage'])
        inputs = self.tokenizer(
            passage,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        item = {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten()
        }
        if not self.is_test:
            target = self.dataframe.iloc[idx]['y']
            item['labels'] = torch.tensor(target, dtype=torch.long)
        return item

# Initialize ALBERT tokenizer
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
max_length = 128

# Define the FocalLoss class
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        alpha = self.alpha.to(targets.device)  # Ensure alpha is on the same device as targets
        log_prob = self.cross_entropy(logits, targets)
        prob = torch.exp(-log_prob)
        focal_loss = alpha[targets] * (1 - prob) ** self.gamma * log_prob
        return focal_loss.mean()

# Stratified K-Fold Cross-Validation (K=min(5, minimum number of samples per class))
num_splits = min(5, y.value_counts().min()) 
skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)

# Cross-Validation Loop
for fold, (train_index, val_index) in enumerate(skf.split(train_df_resampled['passage'], train_df_resampled['y'])):
    print(f"Fold {fold + 1}")
    train_df_fold = train_df_resampled.iloc[train_index]
    val_df_fold = train_df_resampled.iloc[val_index]

    # Create train and validation datasets
    train_dataset = GenderBiasDataset(train_df_fold, tokenizer, max_length)
    val_dataset = GenderBiasDataset(val_df_fold, tokenizer, max_length)

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Calculate class weights for handling class imbalance
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_resampled), y=y_resampled)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize ALBERT model
    model = AlbertForSequenceClassification.from_pretrained(
    'textattack/albert-base-v2-imdb', 
    num_labels=4, 
    ignore_mismatched_sizes=True
)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Define the optimizer and the learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    total_steps = len(train_dataloader) * 3  # Assuming 3 epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Define FocalLoss with class weights
    loss_fn = FocalLoss(alpha=class_weights)

    # Training function with validation and early stopping
    def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn, device, num_epochs=3, patience=2):
        best_val_loss = float('inf')
        patience_counter = 0
        writer = SummaryWriter(f'runs/gender_bias_classification_fold_{fold + 1}')

        for epoch in range(num_epochs):
            start_time = time.time()
            model.train()
            total_loss = 0
            true_labels, pred_labels = [], []

            # Training loop
            for batch in train_dataloader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels)
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                true = labels.detach().cpu().numpy()
                true_labels.extend(true)
                pred_labels.extend(preds)

            avg_train_loss = total_loss / len(train_dataloader)
            train_f1 = f1_score(true_labels, pred_labels, average='weighted')

            # Validation loop
            model.eval()
            val_loss = 0
            true_labels, pred_labels = [], []

            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = loss_fn(logits, labels)
                    val_loss += loss.item()

                    preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                    true = labels.detach().cpu().numpy()
                    true_labels.extend(true)
                    pred_labels.extend(preds)

            avg_val_loss = val_loss / len(val_dataloader)
            val_f1 = f1_score(true_labels, pred_labels, average='weighted')
            epoch_time = time.time() - start_time

            # Log metrics to TensorBoard and print them
            writer.add_scalar('Loss/Training', avg_train_loss, epoch)
            writer.add_scalar('F1/Training', train_f1, epoch)
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
            writer.add_scalar('F1/Validation', val_f1, epoch)

            print(f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}")
            print(f"Training Loss: {avg_train_loss:.4f}, Training F1 Score: {train_f1:.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}, Validation F1 Score: {val_f1:.4f}, Time: {epoch_time:.2f}s")
            logging.info(f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation F1 Score: {val_f1:.4f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                model.save_pretrained(f'best_gender_bias_classifier_fold_{fold + 1}')
                tokenizer.save_pretrained(f'best_gender_bias_classifier_fold_{fold + 1}')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    logging.info("Early stopping triggered")
                    break

        writer.close()

        # Reload the best model for this fold
        model = AlbertForSequenceClassification.from_pretrained(f'best_gender_bias_classifier_fold_{fold + 1}', num_labels=4)
        model = model.to(device)

        # Post-training evaluation
        model.eval()
        true_labels, pred_labels = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                true = labels.detach().cpu().numpy()
                true_labels.extend(true)
                pred_labels.extend(preds)

                # Confusion Matrix
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Male', 'Female', 'Non-binary', 'Neutral'], yticklabels=['Male', 'Female', 'Non-binary', 'Neutral'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for Fold {fold + 1}')
        plt.savefig(f'plots/confusion_matrix_fold_{fold + 1}.png')

        # Classification Report
        report = classification_report(true_labels, pred_labels, target_names=['Male', 'Female', 'Non-binary', 'Neutral'])
        print(report)
        logging.info(f"Classification Report for Fold {fold + 1}:\n{report}")

    # Train the model with validation and early stopping
    train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn, device)

# Load test dataset and evaluate the quantized model
test_df = pd.read_csv('test.csv')
test_dataset = GenderBiasDataset(test_df, tokenizer, max_length, is_test=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the best model for evaluation
# Here I use the best model from the last fold, you could consider averaging predictions from each fold if needed
best_model_path = f'best_gender_bias_classifier_fold_{fold + 1}'
model = AlbertForSequenceClassification.from_pretrained(best_model_path, num_labels=4)
model = model.to(device)

# Apply quantization to reduce model size
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Generate predictions for submission using the quantized model
quantized_model.eval()
predictions = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = quantized_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        predictions.extend(preds)

# Calculate the number of parameters in the quantized model
parameters = sum(p.numel() for p in quantized_model.parameters() if p.requires_grad)

# Prepare the submission file
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'y_pred': predictions,
    'parameters': [parameters] * len(predictions)  
})

submission_df.to_csv('quantized_submission.csv', index=False)
print("Submission file 'quantized_submission.csv' created successfully.")