"""
Script pentru salvarea modelului neantrenat (cu weights random).
Necesar pentru prerequisitele Etapa 5.
"""

import os
import sys
import torch

# Adaugă path pentru import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_network.model import NeuralNetwork


def save_untrained_model():
    """Creează și salvează un model neantrenat."""
    
    # Configurare model pentru clasificare intenții
    # Input: embedding text (vom folosi 100 features pentru vocabular simplu)
    # Hidden: 3 straturi [128, 64, 32]
    # Output: număr de intenții unice (vom estima 109 din datele generate)
    
    INPUT_SIZE = 100  # Dimensiune vocabular/features
    HIDDEN_LAYERS = [128, 64, 32]
    OUTPUT_SIZE = 109  # Număr de intenții unice
    
    print("=" * 60)
    print("SALVARE MODEL NEANTRENAT")
    print("=" * 60)
    
    # Creare model
    model = NeuralNetwork(
        input_size=INPUT_SIZE,
        hidden_layers=HIDDEN_LAYERS,
        output_size=OUTPUT_SIZE,
        activation='relu',
        output_activation='softmax',
        dropout=0.2
    )
    
    print(f"\n📐 Arhitectură model:")
    print(f"   Input:  {INPUT_SIZE} features")
    print(f"   Hidden: {HIDDEN_LAYERS}")
    print(f"   Output: {OUTPUT_SIZE} clase (intenții)")
    print(f"   Activare: ReLU + Softmax")
    print(f"   Dropout: 0.2")
    
    # Număr parametri
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Parametri:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    # Salvare model
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'untrained_model.pt')
    
    # Salvăm doar state_dict și metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_size': INPUT_SIZE,
            'hidden_layers': HIDDEN_LAYERS,
            'output_size': OUTPUT_SIZE,
            'activation': 'relu',
            'output_activation': 'softmax',
            'dropout': 0.2
        },
        'trained': False,
        'version': '1.0'
    }, model_path)
    
    print(f"\n✅ Model neantrenat salvat în: {model_path}")
    print(f"   Dimensiune fișier: {os.path.getsize(model_path) / 1024:.2f} KB")
    
    # Verificare
    print("\n🔍 Verificare încărcare...")
    checkpoint = torch.load(model_path, weights_only=False)
    print(f"   Config: {checkpoint['config']}")
    print(f"   Trained: {checkpoint['trained']}")
    print("\n✅ Verificare completă!")
    
    return model_path


if __name__ == "__main__":
    save_untrained_model()
