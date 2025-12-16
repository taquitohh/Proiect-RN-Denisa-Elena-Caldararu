"""
Modul de inferență pentru clasificarea intențiilor Text-to-Blender.
====================================================================

Acest modul încarcă modelul antrenat și oferă funcții pentru
clasificarea textului în intenții Blender.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

# Path-uri relative la proiect
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"


class BlenderIntentClassifier:
    """
    Clasificator de intenții Blender bazat pe rețea neuronală antrenată.
    
    Încarcă modelul antrenat și parametrii de preprocesare,
    apoi oferă metode pentru clasificarea textului.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Inițializare clasificator.
        
        Args:
            model_path: Calea către modelul antrenat. 
                       Default: models/trained_model.pt
        """
        self.model = None
        self.vocab = None
        self.intent_to_idx = None
        self.idx_to_intent = None
        self.is_loaded = False
        
        # Încărcare automată dacă există model
        if model_path is None:
            model_path = str(MODELS_DIR / "trained_model.pt")
        
        if os.path.exists(model_path):
            self.load(model_path)
    
    def load(self, model_path: str) -> bool:
        """
        Încarcă modelul și parametrii de preprocesare.
        
        Returns:
            True dacă încărcarea a reușit, False altfel
        """
        try:
            # 1. Încărcare parametri preprocesare
            params_path = CONFIG_DIR / "preprocessing_params.pkl"
            if not params_path.exists():
                print(f"⚠️ Lipsește fișierul: {params_path}")
                return False
            
            with open(params_path, 'rb') as f:
                params = pickle.load(f)
            
            self.vocab = params['vocab']
            self.intent_to_idx = params['intent_to_idx']
            self.idx_to_intent = params['idx_to_intent']
            self.vocab_size = params['vocab_size']
            self.num_classes = params['num_classes']
            
            # 2. Creare word_to_idx
            self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
            
            # 3. Încărcare model
            if not os.path.exists(model_path):
                print(f"⚠️ Lipsește modelul: {model_path}")
                return False
            
            # Import lazy pentru a evita circular imports
            from neural_network.model import NeuralNetwork
            
            # Creare model cu aceeași arhitectură ca în train_optimized.py
            self.model = NeuralNetwork(
                input_size=self.vocab_size,
                hidden_layers=[128, 64],  # Actualizat conform train_optimized.py
                output_size=self.num_classes,
                activation='relu',
                output_activation='softmax',
                dropout=0.3  # Actualizat conform train_optimized.py
            )
            
            # Încărcare weights
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.is_loaded = True
            print(f"✅ Model încărcat: {self.num_classes} intenții, {self.vocab_size} vocabular")
            return True
            
        except Exception as e:
            print(f"❌ Eroare la încărcarea modelului: {e}")
            return False
    
    def text_to_bow(self, text: str) -> np.ndarray:
        """
        Convertește text în vector Bag-of-Words.
        
        Args:
            text: Textul de convertit
            
        Returns:
            Vector numpy de dimensiune vocab_size
        """
        # Tokenizare simplă
        tokens = text.lower().replace(',', ' ').replace('.', ' ').split()
        
        # Creare vector BoW
        bow = np.zeros(self.vocab_size, dtype=np.float32)
        
        for token in tokens:
            if token in self.word_to_idx:
                idx = self.word_to_idx[token]
                bow[idx] = 1.0  # Binary BoW
            # else: cuvânt necunoscut - ignorat (sau ar putea folosi <UNK>)
        
        return bow
    
    def predict(self, text: str, top_k: int = 3) -> Dict:
        """
        Clasifică textul și returnează intenția prezisă.
        
        Args:
            text: Textul de clasificat
            top_k: Câte predicții top să returneze
            
        Returns:
            Dict cu:
                - intent: intenția prezisă
                - confidence: scorul de încredere
                - top_k: lista top-k predicții cu scoruri
        """
        if not self.is_loaded:
            return {
                "intent": "create_cube",
                "confidence": 0.0,
                "top_k": [],
                "error": "Modelul nu este încărcat"
            }
        
        try:
            # 1. Convertire text în BoW
            bow = self.text_to_bow(text)
            
            # 2. Conversie în tensor
            X = torch.FloatTensor(bow).unsqueeze(0)  # Shape: (1, vocab_size)
            
            # 3. Predicție
            with torch.no_grad():
                outputs = self.model(X)
                probabilities = torch.softmax(outputs, dim=1).numpy()[0]
            
            # 4. Extragere rezultate
            predicted_idx = int(np.argmax(probabilities))
            predicted_intent = self.idx_to_intent.get(predicted_idx, "unknown")
            confidence = float(probabilities[predicted_idx])
            
            # 5. Top-k predicții
            top_indices = np.argsort(probabilities)[::-1][:top_k]
            top_predictions = [
                {
                    "intent": self.idx_to_intent.get(int(idx), "unknown"),
                    "confidence": float(probabilities[idx])
                }
                for idx in top_indices
            ]
            
            return {
                "intent": predicted_intent,
                "confidence": confidence,
                "top_k": top_predictions,
                "method": "neural_network"
            }
            
        except Exception as e:
            return {
                "intent": "create_cube",
                "confidence": 0.0,
                "top_k": [],
                "error": str(e)
            }
    
    def get_available_intents(self) -> List[str]:
        """Returnează lista de intenții disponibile."""
        if self.intent_to_idx:
            return list(self.intent_to_idx.keys())
        return []


# Singleton instance pentru utilizare globală
_classifier_instance: Optional[BlenderIntentClassifier] = None


def get_classifier() -> BlenderIntentClassifier:
    """
    Returnează instanța singleton a clasificatorului.
    
    Returns:
        BlenderIntentClassifier încărcat și gata de utilizare
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = BlenderIntentClassifier()
    return _classifier_instance


def classify_intent(text: str, top_k: int = 3) -> Dict:
    """
    Funcție helper pentru clasificarea rapidă a textului.
    
    Args:
        text: Textul de clasificat
        top_k: Câte predicții top să returneze
        
    Returns:
        Dict cu rezultatul clasificării
    """
    classifier = get_classifier()
    return classifier.predict(text, top_k)
