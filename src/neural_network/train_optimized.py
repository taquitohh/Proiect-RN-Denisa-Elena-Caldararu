"""
Script de antrenare OPTIMIZAT pentru a preveni overfitting.
============================================================

Modificări anti-overfitting:
1. Dropout crescut la 0.4
2. Weight decay (L2 regularization) 
3. Arhitectură mai simplă [64, 32]
4. Early stopping patience=5
5. Batch size mai mare (64) pentru gradient mai stabil
"""

import os
import sys
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score

# Path setup
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class SimpleNeuralNetwork(nn.Module):
    """Rețea neuronală compatibilă cu clasa NeuralNetwork din model.py."""
    
    def __init__(self, input_size, hidden_layers, output_size, dropout=0.3):
        super(SimpleNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())  # Fără BatchNorm - compatibil cu model.py
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def load_data():
    """Încarcă datele."""
    data_dir = PROJECT_ROOT / "data"
    
    X_train = np.load(data_dir / "train" / "X_train.npy")
    y_train = np.load(data_dir / "train" / "y_train.npy")
    X_val = np.load(data_dir / "validation" / "X_val.npy")
    y_val = np.load(data_dir / "validation" / "y_val.npy")
    X_test = np.load(data_dir / "test" / "X_test.npy")
    y_test = np.load(data_dir / "test" / "y_test.npy")
    
    print(f"✅ Date încărcate: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=64):
    """Creează DataLoaders."""
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Antrenare o epocă."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()
    
    return total_loss / len(train_loader), correct / total


def validate(model, val_loader, criterion, device):
    """Validare model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
    
    return total_loss / len(val_loader), correct / total


def evaluate_test(model, X_test, y_test, device):
    """Evaluare pe test set."""
    model.eval()
    X_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predictions = outputs.max(1)
        predictions = predictions.cpu().numpy()
    
    accuracy = np.mean(predictions == y_test)
    f1 = f1_score(y_test, predictions, average='macro', zero_division=0)
    
    return accuracy, f1, predictions


def main():
    print("=" * 60)
    print("🧠 Antrenare OPTIMIZATĂ (Anti-Overfitting)")
    print("=" * 60)
    
    # Config echilibrat: anti-overfitting + F1 bun
    config = {
        'hidden_layers': [128, 64],      # Capacitate rezonabilă
        'dropout': 0.3,                  # Dropout moderat
        'learning_rate': 0.001,
        'weight_decay': 1e-4,            # L2 regularization ușoară
        'batch_size': 32,                # Batch original
        'epochs': 150,
        'patience': 10,                  # Mai multă răbdare
        'min_delta': 0.001
    }
    
    print(f"\n📋 Configurație echilibrată (anti-overfitting + F1):")
    print(f"   Hidden layers: {config['hidden_layers']}")
    print(f"   Dropout: {config['dropout']}")
    print(f"   Weight decay: {config['weight_decay']} (L2 regularization)")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Early stopping patience: {config['patience']}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Load preprocessing params
    with open(PROJECT_ROOT / "config" / "preprocessing_params.pkl", 'rb') as f:
        params = pickle.load(f)
    
    input_size = X_train.shape[1]
    output_size = params['num_classes']
    
    print(f"\n📊 Model: {input_size} → {config['hidden_layers']} → {output_size}")
    
    # Create model
    model = SimpleNeuralNetwork(
        input_size=input_size,
        hidden_layers=config['hidden_layers'],
        output_size=output_size,
        dropout=config['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parametri: {total_params:,} (redus pentru anti-overfitting)")
    
    # Loss, optimizer cu weight decay
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']  # L2 regularization
    )
    
    # DataLoaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, config['batch_size']
    )
    
    # Training loop
    print(f"\n{'='*60}")
    print("🚀 Începere antrenare...")
    print(f"{'='*60}")
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    start_time = datetime.now()
    
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Check overfitting
        gap = train_acc - val_acc
        overfitting_indicator = "⚠️" if gap > 0.1 else "✓"
        
        if (epoch + 1) % 5 == 0 or epoch < 5:
            print(f"Epoca {epoch+1:3d} | Train: {train_loss:.4f} ({train_acc*100:.1f}%) | "
                  f"Val: {val_loss:.4f} ({val_acc*100:.1f}%) | Gap: {gap*100:.1f}% {overfitting_indicator}")
        
        # Early stopping
        if val_loss < best_val_loss - config['min_delta']:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\n⏹️ Early stopping la epoca {epoch + 1} (patience={config['patience']})")
                break
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"✅ Restaurat modelul cu best val_loss={best_val_loss:.4f}")
    
    # Evaluate on test
    print(f"\n{'='*60}")
    print("📈 Evaluare pe test set...")
    print(f"{'='*60}")
    
    test_acc, test_f1, predictions = evaluate_test(model, X_test, y_test, device)
    
    # Final metrics
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    gap = final_train_acc - final_val_acc
    
    print(f"\n📊 REZULTATE FINALE:")
    print(f"   Train Accuracy:  {final_train_acc*100:.2f}%")
    print(f"   Val Accuracy:    {final_val_acc*100:.2f}%")
    print(f"   Test Accuracy:   {test_acc*100:.2f}%")
    print(f"   Test F1 (macro): {test_f1:.4f}")
    print(f"   Gap Train-Val:   {gap*100:.2f}%")
    
    # Overfitting check
    print(f"\n🔍 Verificare Overfitting:")
    if gap < 0.05:
        print(f"   ✅ EXCELENT! Gap < 5% - Nu există overfitting")
        status = "GOOD_FIT"
    elif gap < 0.10:
        print(f"   ✅ BUN! Gap < 10% - Overfitting minim")
        status = "SLIGHT_OVERFIT"
    else:
        print(f"   ⚠️ ATENȚIE! Gap > 10% - Overfitting detectat")
        status = "OVERFIT"
    
    # Check objectives
    print(f"\n🎯 Verificare obiective Etapa 5:")
    acc_ok = test_acc >= 0.65
    f1_ok = test_f1 >= 0.60
    print(f"   Accuracy ≥ 65%: {'✅ DA' if acc_ok else '❌ NU'} ({test_acc*100:.2f}%)")
    print(f"   F1 ≥ 0.60:      {'✅ DA' if f1_ok else '❌ NU'} ({test_f1:.4f})")
    
    # Save model
    models_dir = PROJECT_ROOT / "models"
    model_path = models_dir / "trained_model.pt"
    
    # Save in format compatibil cu inference.py
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'input_size': input_size,
        'output_size': output_size,
        'metrics': {
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'train_val_gap': gap,
            'status': status
        }
    }, model_path)
    print(f"\n💾 Model salvat: {model_path}")
    
    # Save history
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    
    history_df = pd.DataFrame(history)
    history_df['epoch'] = range(1, len(history['train_loss']) + 1)
    history_df = history_df[['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']]
    history_df.to_csv(results_dir / "training_history.csv", index=False)
    
    # Save metrics
    metrics = {
        'test_loss': float(history['val_loss'][-1]),
        'accuracy': float(test_acc),
        'f1_macro': float(test_f1),
        'train_val_gap': float(gap),
        'overfitting_status': status,
        'epochs_completed': len(history['train_loss']),
        'training_time_seconds': training_time,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(results_dir / "test_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_title('Loss (Train vs Validation)', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].axhline(y=0.65, color='g', linestyle='--', label='Target 65%', alpha=0.7)
    axes[1].fill_between(epochs, history['train_acc'], history['val_acc'], 
                         alpha=0.2, color='orange', label=f'Gap ({gap*100:.1f}%)')
    axes[1].set_title('Accuracy (Train vs Validation)', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=150)
    plt.close()
    print(f"📊 Grafic salvat: {results_dir / 'training_curves.png'}")
    
    print(f"\n{'='*60}")
    print("✅ Antrenare completă!")
    print(f"{'='*60}")
    
    return metrics


if __name__ == "__main__":
    main()
