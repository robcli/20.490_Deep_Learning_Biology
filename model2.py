import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel

import pandas as pd
import utils.parser as parser
from sklearn.metrics import accuracy_score, recall_score, precision_score

import os
import random
from typing import List

print("libraries loaded", flush=True)


class NTTokenizer():
    def __init__(self, max_contig_len: int=512):
        self.mp = {"A":0, "T":1, "C":2, "G":3, "PAD":4}
        self.r_mp = {0:"A", 1:"T", 2:"C", 3:"G", 4:""}
        self.max_contig_len = max_contig_len
        
    def convert(self, seq: str):
        seq_arr = [self.mp[nt] for nt in seq]
        while len(seq_arr) < self.max_contig_len:
            seq_arr.append(self.mp["PAD"])
        return seq_arr
    
    def revert(self, arr: List[int]):
        return "".join([self.r_mp[i] for i in arr])

class MicrobiomeDataset(Dataset): 
    def __init__(self, fasta_files, labels, k: int=6, size: int=None, trim_to: int=None):
        self.fasta_files = fasta_files
        self.labels = labels
        self.label_tokenizer = {"nonIBD":0, "CD":1, "UC": 1}
        self.seq_tokenizer = NTTokenizer()
        self.k = k
        self.vocab = parser.build_vocab(k=k)
        self.sequences = []
        
        for i, file in enumerate(fasta_files):
            if size is not None and i >= size:
                break

            data = parser.read_tokenized_file(file)
            if trim_to is not None:
                samples = random.sample(data, k=trim_to)
                data = [self.seq_tokenizer.convert(sample) for sample in samples]
                
            self.sequences.append((data, 
                                  self.label_tokenizer[labels[i]]))
            
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        tokenized_sequences, label = self.sequences[idx]
        return torch.tensor(tokenized_sequences), torch.tensor(label)
    
    
class MultiheadAttentionPooling(nn.Module):
    """
    Pooling mechanism that uses multihead attention to create a weighted average of sequence embeddings.
    """
    def __init__(self, d_model: int=768, num_heads: int=12):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Query vector to attend to the sequence
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Multi-head attention for pooling
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, microbiome_embeddings: torch.Tensor):
        """
        Apply attention pooling to a sequence of embeddings.
        Args:
            microbiome_embeddings: Tensor of shape (variable_length, d_model)  
        Returns:
            Tensor of shape (batch, d_model) containing pooled representation
        """
        microbiome_embeddings = microbiome_embeddings.unsqueeze(0)  # (1, variable_length, d_model)
        batch_size, var_len, _ = microbiome_embeddings.shape
        
        # Expand query to match batch size
        query = self.query.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        
        # Apply multi-head attention to get weighted sum
        pooled_output, _ = self.multihead_attn(
            query=query,                      # (batch, 1, d_model)
            key=microbiome_embeddings,          # (batch, variable_length, d_model)
            value=microbiome_embeddings,        # (batch, variable_length, d_model)
        )
        
        # Remove the 2nd dimension (batch, 1, d_model)
        pooled_output = pooled_output.squeeze(1)  # (batch, d_model)
        return pooled_output

class MeanPooling(nn.Module):
    """
    Pooling mechanism that uses mean pooling to create an average of sequence embeddings.
    """
    def __init__(self):
        super().__init__()

    def forward(self, microbiome_embeddings: torch.Tensor, padding_token: int=0):
        """
        Apply mean pooling to a sequence of embeddings.
        Args:
            microbiome_embeddings: Tensor of shape (variable_length, d_model)  
        Returns:
            Tensor of shape (batch, d_model) containing mean pooled representation
        """
        masked = (microbiome_embeddings == padding_token).float()  # Shape: (variable_length, d_model)
        summed_embeddings = torch.sum(microbiome_embeddings, dim=0)
        token_count = torch.sum(masked, dim=0).clamp(min=1.0)
        pooled_output = summed_embeddings / token_count  # Shape: (d_model)
 
        return pooled_output.unsqueeze(0)  # (batch, d_model)

class MicrobiomeAttentionPooler(nn.Module):
    """
    Module that applies attention pooling to each microbiome.
    """
    def __init__(self, pooling: str="mean", d_model: int=768, n_heads: int=12):
        super().__init__()
        self.d_model = d_model
        self.pooling = pooling

        if pooling == "mean":
            self.microbiome_pooler = MeanPooling()
        else:
            self.microbiome_pooler = MultiheadAttentionPooling(d_model=d_model, num_heads=n_heads)
            self.projection = nn.Linear(d_model, d_model)
            self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, batch_embeddings):
        """
        Pool a batch of microbiome embeddings at the microbiome levels.        
        Args:
            batch_embeddings: List of lists of sequence embeddings
                Each sequence embedding is a tensor of shape (d_model)
        Returns:
            List of microbiome-level embeddings, each of shape (d_model)
        """
        batch_pooled = []
        
        for microbiome_embeddings in batch_embeddings:
            # microbiome_embeddings have shape: (variable_length, d_model)

            # Pool all tokens directly to get microbiome representation
            microbiome_pooled_embedding = self.microbiome_pooler(microbiome_embeddings)  # (d_model)
            
            if self.pooling != "mean":
                # Apply projection and normalization
                microbiome_pooled_embedding = self.projection(microbiome_pooled_embedding)
                microbiome_pooled_embedding = self.layer_norm(microbiome_pooled_embedding)
            
            batch_pooled.append(microbiome_pooled_embedding)
        
        return torch.cat(batch_pooled)  # Shape: (batch_size, d_model)
    
class MicrobiomeLM(nn.Module):
    def __init__(self, 
                 num_outputs: int=2, 
                 d_model: int=768, 
                 n_heads: int=12, 
                 n_layers: int=12, 
                 dropout: float=0.1, 
                 pooling: str="mean",
                 freeze_dnabert: bool=True):
        super().__init__()

        self.nt_vocab = NTTokenizer()
        self.tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", 
                                                  trust_remote_code=True, 
                                                  force_download=False, 
                                                  revision= "2efd650282ec5d5ab377c787c76ea56b723f6b7c", 
                                                  resume_download=None)
        self.DNABERT = AutoModel.from_pretrained("zhihan1996/DNABERT-S", 
                                          trust_remote_code=True, 
                                          force_download=False, revision="2efd650282ec5d5ab377c787c76ea56b723f6b7c", 
                                          resume_download=None)
        
        if freeze_dnabert:
            for param in self.DNABERT.parameters():
                param.requires_grad = False

        self.pooler = MicrobiomeAttentionPooler(pooling=pooling, d_model=d_model, n_heads=n_heads)
        self.FC = nn.Linear(d_model, d_model) 
        self.activation_fn = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, num_outputs)
        
    def forward(self, batch_sequences):
        batch_embeddings = [] # Shape: (batch, variable_length, seq_len, d_model)

        batch_size = batch_sequences.shape[0]
        for i in range(batch_size):
            microbiome = batch_sequences[i]  # Shape: (variable_length, seq_len)
            sequence_embeddings = self._process_microbiome(microbiome)
            batch_embeddings.append(sequence_embeddings) 
        
        x = self.pooler(batch_embeddings)
        x = self.FC(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        logits = self.output_layer(x)
        return logits

    def _process_microbiome(self, microbiome):
        """
        Process a single microbiome containing multiple sequences and return list of embeddings per sequence.
        """
        sequences = [self.nt_vocab.revert(seq) for seq in microbiome.tolist()]
        tokenized_sequences = self.tokenizer(sequences, return_tensors = 'pt', padding=True)["input_ids"].to(device=microbiome.device)
        hidden_states = self.DNABERT(tokenized_sequences)[0]
        embedding_mean = torch.mean(hidden_states, dim=1)

        return embedding_mean # Shape: (variable_length, d_model)
    
import time
def train_model(model, dataset, save_as=None, val_dataset=None, epochs: int=10, batch_size: int=1, lr=2.5e-5, device='cuda', collate_fn=None):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn) if val_dataset is not None else None

    for epoch in range(epochs):
        running_loss = 0
        preds = []
        ground = []
        start_time = time.perf_counter()
        
        model.train()
        for inputs, labels in train_dataloader:            
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            preds.extend(logits.argmax(dim=1).cpu().detach())
            ground.extend(labels.cpu().detach())

            #nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += loss.item()

        train_accuracy, train_recall, train_precision = compute_metrics(ground, preds)
            
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {(running_loss/len(train_dataloader)):.4f}, Time: {time.perf_counter()-start_time:.2f}")
        print(f"   Train Accuracy: {train_accuracy:.4f}, Train Recall: {train_recall:.4f}, Train Precision: {train_precision:.4f}")
        
        if val_dataloader:
            model.eval()
            
            running_loss = 0
            preds = []
            ground = []
            min_loss = float("inf")
            for inputs, labels in val_dataloader:            
                inputs, labels = inputs.to(device), labels.to(device) 
                
                logits = model(inputs)
                loss = loss_fn(logits, labels)
                preds.extend(logits.argmax(dim=1).cpu().detach())
                ground.extend(labels.cpu().detach())
                
                running_loss += loss.item()
                
            val_accuracy, val_recall, val_precision = compute_metrics(ground, preds)
            
            print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {(running_loss/len(val_dataloader)):.4f}", flush=True)
            print(f"   Validation Accuracy: {val_accuracy:.4f}, Validation Recall: {val_recall:.4f}, Validation Precision: {val_precision:.4f}\n", flush=True)
            
            if save_as and running_loss < min_loss:
                min_loss = running_loss
                torch.save(model.state_dict(), save_as)
            
        
def evaluate_model(model, dataset, batch_size: int=1, device='cuda', collate_fn=None):
    model = model.to(device)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    loss_fn = nn.CrossEntropyLoss()
    preds, ground = [], []
    running_loss = 0
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device) 
            
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            
            running_loss += loss.item()
            preds.extend(logits.argmax(dim=1).cpu().detach())
            ground.extend(labels.cpu().detach())
            
        train_accuracy, train_recall, train_precision = compute_metrics(ground, preds)
        
        print(f"Test Loss: {(running_loss/len(test_dataloader)):.2f}", flush=True)
        print(f"   Test Accuracy: {train_accuracy:.4f}, Test Recall: {train_recall:.4f}, Test Precision: {train_precision:.4f}", flush=True)
        
def roc_auc(model, dataset, batch_size: int=1, device='cuda', collate_fn=None):
    from sklearn.metrics import roc_curve, auc
    
    model = model.to(device)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    preds, ground = [], []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device) 
            
            logits = model(inputs).softmax(dim=1)
            
            preds.extend(logits.cpu())
            ground.extend(labels.cpu())
       
    preds = torch.stack(preds).numpy()
    ground = torch.stack(ground).numpy()
    
    fpr_micro, tpr_micro, _ = roc_curve(ground, preds[:,1])
    auc_micro = auc(fpr_micro, tpr_micro)
    print(f"false positive rate:{fpr_micro}")
    print(f"true positive rate:{tpr_micro}")
    print(f"auc:{auc_micro}")
            
def compute_metrics(y_true, y_pred, average: str="micro"):
    """
    Compute per-base accuracy, sensitivity (recall), and PPV (precision).
    Args:
        y_true (list): Ground truth labels
        y_pred (list): Predicted labels
        average (str): Method for multiclass averaging (macro, micro)
    Returns:
        accuracy, recall, precision (all floats)
    """
    y_true = torch.stack(y_true).numpy().flatten().tolist()
    y_pred = torch.stack(y_pred).numpy().flatten().tolist()
    print(y_true, flush=True)
    print(y_pred, flush=True)

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average=average)
    precision = precision_score(y_true, y_pred, average=average)
    return accuracy, recall, precision
    
metadata = pd.read_csv("SJBae/sampled_metadata_ACBI_balanced.csv")

# print(sum(p.numel() for p in MicrobiomeLM(pooling="MHA").parameters() if p.requires_grad))

# sampled_metadata = pd.concat([
#           metadata[metadata["diagnosis"] == "UC"][:4], 
#           metadata[metadata["diagnosis"] == "CD"][:4], 
#           metadata[metadata["diagnosis"] == "nonIBD"][:8]]
#          ).reset_index()

tokenized_files_dir = "/pool001/robcli/hmp2_shortened_balanced"
tokenized_files = [os.path.join(tokenized_files_dir, f+".pkl") for f in metadata["External ID"]]
labels = metadata["diagnosis"]

# idx = random.sample(range(258), k=26)
# tokenized_files = [tokenized_files[i] for i in idx]
# labels = [labels[i] for i in idx]

full_dataset = MicrobiomeDataset(tokenized_files, labels, trim_to = 512)

train_size = 206
val_size = 26
test_size = 26

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

model = MicrobiomeLM(pooling="mean", dropout=0)
device = "cuda" if torch.cuda.is_available() else "cpu"

train_model(model, 
            train_dataset,
            val_dataset=val_dataset,
            epochs=50,
            batch_size=2,
            device=device,
            save_as="saved_freezed_mean2_512.pth")
            

evaluate_model(model, 
              test_dataset, 
              batch_size=2, 
              device=device)

roc_auc(model, test_dataset)

# model = MicrobiomeLM(pooling="mean", dropout=0)
# model.load_state_dict(torch.load("saved_freezed_mean_512.pth"))

    
