"""
Script pentru split-ul datelor în train/validation/test.
Split stratificat 70/15/15 cu random_state=42 pentru reproducibilitate.
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# Configurare
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GENERATED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'generated', 'training_data.json')
TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'train')
VAL_DIR = os.path.join(BASE_DIR, 'data', 'validation')
TEST_DIR = os.path.join(BASE_DIR, 'data', 'test')
CONFIG_DIR = os.path.join(BASE_DIR, 'config')


def load_data():
    """Încarcă datele generate."""
    print(f"📂 Încărcare date din: {GENERATED_DATA_PATH}")
    
    with open(GENERATED_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data['samples']
    print(f"   Total samples: {len(samples)}")
    
    return samples


def create_vocabulary(samples):
    """Creează vocabularul din toate textele."""
    print("\n📝 Creare vocabular...")
    
    all_words = []
    for sample in samples:
        text = sample['text'].lower()
        words = text.split()
        all_words.extend(words)
    
    # Numără frecvența cuvintelor
    word_counts = Counter(all_words)
    
    # Creează vocabular (cuvinte cu frecvență >= 2 sau toate dacă sunt puține)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for word, count in word_counts.most_common():
        if word not in vocab:
            vocab[word] = idx
            idx += 1
    
    print(f"   Vocabular: {len(vocab)} cuvinte unice")
    
    return vocab


def create_intent_mapping(samples):
    """Creează mapping intenție -> index."""
    print("\n🎯 Creare mapping intenții...")
    
    intents = sorted(set(sample['intent'] for sample in samples))
    intent_to_idx = {intent: idx for idx, intent in enumerate(intents)}
    idx_to_intent = {idx: intent for intent, idx in intent_to_idx.items()}
    
    print(f"   Intenții unice: {len(intents)}")
    
    return intent_to_idx, idx_to_intent


def text_to_features(text, vocab, max_len=20):
    """Convertește text în vector de features (bag of words simplificat)."""
    words = text.lower().split()
    
    # Bag of words cu dimensiune fixă
    features = np.zeros(len(vocab), dtype=np.float32)
    
    for word in words:
        if word in vocab:
            features[vocab[word]] = 1.0
        else:
            features[vocab['<UNK>']] = 1.0
    
    return features


def prepare_features(samples, vocab, intent_to_idx):
    """Pregătește features și labels pentru toate samples."""
    print("\n⚙️ Pregătire features...")
    
    X = []
    y = []
    
    for sample in samples:
        # Features (bag of words)
        features = text_to_features(sample['text'], vocab)
        X.append(features)
        
        # Label (intent index)
        y.append(intent_to_idx[sample['intent']])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    
    return X, y


def split_data(X, y, samples):
    """Split stratificat în train/val/test."""
    print(f"\n✂️ Split date: {TRAIN_RATIO*100:.0f}% train / {VAL_RATIO*100:.0f}% val / {TEST_RATIO*100:.0f}% test")
    
    # Verificăm distribuția claselor
    class_counts = Counter(y)
    rare_classes = [cls for cls, count in class_counts.items() if count < 3]
    
    if rare_classes:
        print(f"   ⚠️ {len(rare_classes)} clase cu <3 samples - folosim split fără stratificare pentru acestea")
    
    # Încercăm split stratificat, dacă nu merge folosim split normal
    try:
        # Prima split: train vs (val + test)
        X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
            X, y, np.arange(len(samples)),
            test_size=(VAL_RATIO + TEST_RATIO),
            random_state=RANDOM_STATE,
            stratify=y
        )
        
        # A doua split: val vs test
        X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
            X_temp, y_temp, idx_temp,
            test_size=0.5,
            random_state=RANDOM_STATE,
            stratify=y_temp
        )
    except ValueError as e:
        print(f"   ⚠️ Stratificare imposibilă, folosim split random: {str(e)[:50]}...")
        
        # Split fără stratificare
        X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
            X, y, np.arange(len(samples)),
            test_size=(VAL_RATIO + TEST_RATIO),
            random_state=RANDOM_STATE
        )
        
        X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
            X_temp, y_temp, idx_temp,
            test_size=0.5,
            random_state=RANDOM_STATE
        )
    
    # Extrage samples corespunzătoare
    samples_train = [samples[i] for i in idx_train]
    samples_val = [samples[i] for i in idx_val]
    samples_test = [samples[i] for i in idx_test]
    
    print(f"   Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    return (X_train, y_train, samples_train), (X_val, y_val, samples_val), (X_test, y_test, samples_test)


def save_split(data, samples, directory, name):
    """Salvează un split în directorul specificat."""
    X, y = data[0], data[1]
    samples_list = data[2]
    
    os.makedirs(directory, exist_ok=True)
    
    # Salvare numpy arrays
    np.save(os.path.join(directory, f'X_{name}.npy'), X)
    np.save(os.path.join(directory, f'y_{name}.npy'), y)
    
    # Salvare CSV pentru vizualizare
    df = pd.DataFrame({
        'text': [s['text'] for s in samples_list],
        'intent': [s['intent'] for s in samples_list],
        'label': y
    })
    df.to_csv(os.path.join(directory, f'{name}_data.csv'), index=False, encoding='utf-8')
    
    print(f"   ✅ {name}: {len(X)} samples salvate în {directory}")


def save_preprocessing_params(vocab, intent_to_idx, idx_to_intent):
    """Salvează parametrii de preprocesare."""
    print("\n💾 Salvare parametri preprocesare...")
    
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    params = {
        'vocab': vocab,
        'vocab_size': len(vocab),
        'intent_to_idx': intent_to_idx,
        'idx_to_intent': idx_to_intent,
        'num_classes': len(intent_to_idx),
        'random_state': RANDOM_STATE,
        'split_ratios': {
            'train': TRAIN_RATIO,
            'validation': VAL_RATIO,
            'test': TEST_RATIO
        }
    }
    
    # Salvare pickle
    pkl_path = os.path.join(CONFIG_DIR, 'preprocessing_params.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(params, f)
    print(f"   ✅ Parametri salvați în: {pkl_path}")
    
    # Salvare JSON pentru referință
    json_params = {
        'vocab_size': len(vocab),
        'num_classes': len(intent_to_idx),
        'random_state': RANDOM_STATE,
        'split_ratios': params['split_ratios'],
        'intents': list(intent_to_idx.keys())
    }
    json_path = os.path.join(CONFIG_DIR, 'preprocessing_params.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_params, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Referință JSON: {json_path}")
    
    return params


def main():
    """Pipeline principal de split date."""
    print("=" * 60)
    print("SPLIT DATE PENTRU ANTRENARE")
    print("=" * 60)
    
    # 1. Încărcare date
    samples = load_data()
    
    # 2. Creare vocabular și mapping intenții
    vocab = create_vocabulary(samples)
    intent_to_idx, idx_to_intent = create_intent_mapping(samples)
    
    # 3. Pregătire features
    X, y = prepare_features(samples, vocab, intent_to_idx)
    
    # 4. Split date
    train_data, val_data, test_data = split_data(X, y, samples)
    
    # 5. Salvare splits
    print("\n📁 Salvare splits...")
    save_split(train_data, samples, TRAIN_DIR, 'train')
    save_split(val_data, samples, VAL_DIR, 'val')
    save_split(test_data, samples, TEST_DIR, 'test')
    
    # 6. Salvare parametri preprocesare
    params = save_preprocessing_params(vocab, intent_to_idx, idx_to_intent)
    
    # Rezumat final
    print("\n" + "=" * 60)
    print("✅ SPLIT COMPLET!")
    print("=" * 60)
    print(f"\n📊 Statistici finale:")
    print(f"   Vocabular:    {params['vocab_size']} cuvinte")
    print(f"   Clase:        {params['num_classes']} intenții")
    print(f"   Train:        {len(train_data[0])} samples")
    print(f"   Validation:   {len(val_data[0])} samples")
    print(f"   Test:         {len(test_data[0])} samples")
    print(f"\n📂 Fișiere create:")
    print(f"   {TRAIN_DIR}/")
    print(f"   {VAL_DIR}/")
    print(f"   {TEST_DIR}/")
    print(f"   {CONFIG_DIR}/preprocessing_params.pkl")
    

if __name__ == "__main__":
    main()
