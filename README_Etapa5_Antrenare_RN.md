# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Popa Denis-Alexandru  
**Link Repository GitHub:** https://github.com/taquitohh/Proiect_RN  
**Data predării:** 16 Decembrie 2024

---

## 🎯 Rezultate Antrenare - REZUMAT

| **Metrică** | **Valoare** | **Target** | **Status** |
|-------------|-------------|------------|------------|
| **Accuracy** | 75.64% | ≥65% | ✅ **ATINS** |
| **F1 Score (macro)** | 0.6032 | ≥0.60 | ✅ **ATINS** |
| **F1 Score (weighted)** | 0.7311 | - | ✅ |
| **Top-3 Accuracy** | 82.91% | - | 🎉 Bonus |
| **Top-5 Accuracy** | 85.47% | - | 🎉 Bonus |

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Obiectiv principal:** Antrenarea efectivă a modelului RN definit în Etapa 4, evaluarea performanței și integrarea în aplicația completă.

**Pornire obligatorie:** Arhitectura completă și funcțională din Etapa 4:
- State Machine definit și justificat
- Cele 3 module funcționale (Data Logging, RN, UI)
- Minimum 40% date originale în dataset

---

## PREREQUISITE – Verificare Etapa 4 (OBLIGATORIU)

**Înainte de a începe Etapa 5, verificați că aveți din Etapa 4:**

- [x] **State Machine** definit și documentat în `docs/diagrams/state_machine.png`
- [x] **Contribuție ≥40% date originale** în `data/raw/` - 100% date generate de noi (1,560 samples)
- [x] **Modul 1 (Data Logging)** funcțional - `src/data_acquisition/data_loader.py`
- [x] **Modul 2 (RN)** cu arhitectură definită dar NEANTRENATĂ (`models/untrained_model.pt`)
- [x] **Modul 3 (UI/Web Service)** funcțional - Frontend React + Backend Flask
- [x] **Tabelul "Nevoie → Soluție → Modul"** complet în README Etapa 4

** Dacă oricare din punctele de mai sus lipsește → reveniți la Etapa 4 înainte de a continua.**

---

## Pregătire Date pentru Antrenare 

### Dacă ați adăugat date noi în Etapa 4 (contribuția de 40%):

**TREBUIE să refaceți preprocesarea pe dataset-ul COMBINAT:**

Exemplu:
```bash
# 1. Combinare date vechi (Etapa 3) + noi (Etapa 4)
python src/preprocessing/combine_datasets.py

# 2. Refacere preprocesare COMPLETĂ
python src/preprocessing/data_cleaner.py
python src/preprocessing/feature_engineering.py
python src/preprocessing/data_splitter.py --stratify --random_state 42

# Verificare finală:
# data/train/ → trebuie să conțină date vechi + noi
# data/validation/ → trebuie să conțină date vechi + noi
# data/test/ → trebuie să conțină date vechi + noi
```

** ATENȚIE - Folosiți ACEIAȘI parametri de preprocesare:**
- Același `scaler` salvat în `config/preprocessing_params.pkl`
- Aceiași proporții split: 70% train / 15% validation / 15% test
- Același `random_state=42` pentru reproducibilitate

**Verificare rapidă:**
```python
import pandas as pd
train = pd.read_csv('data/train/X_train.csv')
print(f"Train samples: {len(train)}")  # Trebuie să includă date noi
```

---

##  Cerințe Structurate pe 3 Niveluri

### Nivel 1 – Obligatoriu pentru Toți (70% din punctaj)

Completați **TOATE** punctele următoare:

1. **Antrenare model** definit în Etapa 4 pe setul final de date (≥40% originale)
2. **Minimum 10 epoci**, batch size 8–32
3. **Împărțire stratificată** train/validation/test: 70% / 15% / 15%
4. **Tabel justificare hiperparametri** (vezi secțiunea de mai jos - OBLIGATORIU)
5. **Metrici calculate pe test set:**
   - **Acuratețe ≥ 65%**
   - **F1-score (macro) ≥ 0.60**
6. **Salvare model antrenat** în `models/trained_model.h5` (Keras/TensorFlow) sau `.pt` (PyTorch) sau `.lvmodel` (LabVIEW)
7. **Integrare în UI din Etapa 4:**
   - UI trebuie să încarce modelul ANTRENAT (nu dummy)
   - Inferență REALĂ demonstrată
   - Screenshot în `docs/screenshots/inference_real.png`

#### Tabel Hiperparametri și Justificări (OBLIGATORIU - Nivel 1)

Completați tabelul cu hiperparametrii folosiți și **justificați fiecare alegere**:

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
|--------------------|-------------------|-----------------|
| Learning rate | 0.001 | Valoare standard pentru Adam optimizer, asigură convergență stabilă pentru clasificare multi-class |
| Batch size | 32 | Cu 1,092 train samples → 34 iterații/epocă. Compromis optim memorie/stabilitate gradient |
| Number of epochs | 100 (max) | Cu early stopping patience=10; a rulat efectiv 36 epoci |
| Optimizer | Adam | Adaptive learning rate, performant pentru rețele feed-forward cu 3 straturi hidden |
| Loss function | CrossEntropyLoss | Standard pentru clasificare multi-class cu 109 clase (intenții Blender) |
| Activation functions | ReLU (hidden), Softmax (output) | ReLU evită vanishing gradient, Softmax pentru probabilități clase |
| Hidden layers | [128, 64, 32] | Piramidă descrescătoare pentru compresie progresivă a features |
| Dropout | 0.2 | Regularizare pentru prevenire overfitting la dataset mic |
| Early stopping | patience=10 | Oprește antrenarea după 10 epoci fără îmbunătățire val_loss |

**Justificare detaliată batch size:**
```
Am ales batch_size=32 pentru că avem N=1,092 train samples → 1,092/32 ≈ 34 iterații/epocă.
Aceasta oferă un echilibru între:
- Stabilitate gradient (batch prea mic → zgomot mare în gradient)
- Memorie CPU (nu avem GPU, deci memory constraints reduse)
- Timp antrenare (batch 32 asigură convergență în 36 epoci pentru 109 clase)
- Early stopping a oprit antrenarea înainte de overfitting sever
```

**Statistici Antrenare:**
| Parametru | Valoare |
|-----------|--------|
| Epoci rulate | 36 (din 100 max) |
| Timp total antrenare | 3.17 secunde |
| Device | CPU |
| Train samples | 1,092 |
| Validation samples | 234 |
| Test samples | 234 |
| Număr clase | 109 intenții unice |
| Vocabular | 523 cuvinte unice |
| Parametri model | 81,005 |

**Resurse învățare rapidă:**
- Împărțire date: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html (video 3 min: https://youtu.be/1NjLMWSGosI?si=KL8Qv2SJ1d_mFZfr)  
- Antrenare simplă Keras: https://keras.io/examples/vision/mnist_convnet/ (secțiunea „Training”)  
- Antrenare simplă PyTorch: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-an-image-classifier (video 2 min: https://youtu.be/ORMx45xqWkA?si=FXyQEhh0DU8VnuVJ)  
- F1-score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html (video 4 min: https://youtu.be/ZQlEcyNV6wc?si=VMCl8aGfhCfp5Egi)


---

### Nivel 2 – Recomandat (85-90% din punctaj)

Includeți **TOATE** cerințele Nivel 1 + următoarele:

1. **Early Stopping** - oprirea antrenării dacă `val_loss` nu scade în 5 epoci consecutive
2. **Learning Rate Scheduler** - `ReduceLROnPlateau` sau `StepLR`
3. **Augmentări relevante domeniu:**
   - Vibrații motor: zgomot gaussian calibrat, jitter temporal
   - Imagini industriale: slight perspective, lighting variation (nu rotații simple!)
   - Serii temporale: time warping, magnitude warping
4. **Grafic loss și val_loss** în funcție de epoci salvat în `docs/loss_curve.png`
5. **Analiză erori context industrial** (vezi secțiunea dedicată mai jos - OBLIGATORIU Nivel 2)

**Indicatori țintă Nivel 2:**
- **Acuratețe ≥ 75%**
- **F1-score (macro) ≥ 0.70**

**Resurse învățare (aplicații industriale):**
- Albumentations: https://albumentations.ai/docs/examples/   
- Early Stopping + ReduceLROnPlateau în Keras: https://keras.io/api/callbacks/   
- Scheduler în PyTorch: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate 

---

### Nivel 3 – Bonus (până la 100%)

**Punctaj bonus per activitate:**

| **Activitate** |  **Livrabil** |
|----------------|--------------|
| Comparare 2+ arhitecturi diferite | Tabel comparativ + justificare alegere finală în README |
| Export ONNX/TFLite + benchmark latență | Fișier `models/final_model.onnx` + demonstrație <50ms |
| Confusion Matrix + analiză 5 exemple greșite | `docs/confusion_matrix.png` + analiză în README |

**Resurse bonus:**
- Export ONNX din PyTorch: [PyTorch ONNX Tutorial](https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)
- TensorFlow Lite converter: [TFLite Conversion Guide](https://www.tensorflow.org/lite/convert)
- Confusion Matrix analiză: [Scikit-learn Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

---

## Verificare Consistență cu State Machine (Etapa 4)

Antrenarea și inferența trebuie să respecte fluxul din State Machine-ul vostru definit în Etapa 4.

**Exemplu pentru monitorizare vibrații lagăr:**

| **Stare din Etapa 4** | **Implementare în Etapa 5** |
|-----------------------|-----------------------------|
| `ACQUIRE_DATA` | Citire batch date din `data/train/` pentru antrenare |
| `PREPROCESS` | Aplicare scaler salvat din `config/preprocessing_params.pkl` |
| `RN_INFERENCE` | Forward pass cu model ANTRENAT (nu weights random) |
| `THRESHOLD_CHECK` | Clasificare Normal/Uzură pe baza output RN antrenat |
| `ALERT` | Trigger în UI bazat pe predicție modelului real |

**În `src/app/main.py` (UI actualizat):**

Verificați că **TOATE stările** din State Machine sunt implementate cu modelul antrenat:

```python
# ÎNAINTE (Etapa 4 - model dummy):
model = keras.models.load_model('models/untrained_model.h5')  # weights random
prediction = model.predict(input_scaled)  # output aproape aleator

# ACUM (Etapa 5 - model antrenat):
model = keras.models.load_model('models/trained_model.h5')  # weights antrenate
prediction = model.predict(input_scaled)  # predicție REALĂ și corectă
```

---

## Analiză Erori în Context Industrial (OBLIGATORIU Nivel 2)

**Nu e suficient să raportați doar acuratețea globală.** Analizați performanța în contextul aplicației voastre industriale:

### 1. Pe ce clase greșește cel mai mult modelul?

**Exemplu robotică (predicție traiectorii):**
```
Confusion Matrix arată că modelul confundă 'viraj stânga' cu 'viraj dreapta' în 18% din cazuri.
Cauză posibilă: Features-urile IMU (gyro_z) sunt simetrice pentru viraje în direcții opuse.
```

**Analiza pentru proiectul Text-to-Blender:**
```
Analiză din error_analysis.csv arată 57 erori din 234 samples (24.36%).

Top 5 confuzii:
1. move_object → rotate_object: 2 erori
   Cauză: Ambele sunt operații de transformare cu sintaxă similară ("mută", "rotește")

2. add_modifier_cloth → apply_material_fabric: 1 eroare
   Cauză: Semantică similară ("cloth" vs "fabric")

3. add_modifier_ocean → edit_bridge: 1 eroare
   Cauză: Vocabular limitat pentru comenzi rare

4. add_modifier_mirror → apply_material_brown: 1 eroare
   Cauză: Clasificare incorectă pentru clase cu puține samples

5. add_modifier_screw → add_modifier_subsurf: 1 eroare
   Cauză: Ambele sunt modifiers, structură comandă similară
```

### 2. Ce caracteristici ale datelor cauzează erori?

**Exemplu vibrații motor:**
```
Modelul eșuează când zgomotul de fond depășește 40% din amplitudinea semnalului util.
În mediul industrial, acest nivel de zgomot apare când mai multe motoare funcționează simultan.
```

**Analiza pentru proiectul Text-to-Blender:**
```
Modelul are dificultăți când:
- Clase cu <5 samples în train au accuracy sub 50%
- Comenzile scurte (1-2 cuvinte) sunt mai ambigue
- Sinonime românești ("mișcă" vs "deplasează") confundă modelul
- Comenzile cu context lipsă ("fă un cub" vs "creează un cub basic") sunt ambigue
- Clasele minoritare (modifiers rari) au recall scăzut
```

### 3. Ce implicații are pentru aplicația industrială?

**Exemplu detectare defecte sudură:**
```
FALSE NEGATIVES (defect nedetectat): CRITIC → risc rupere sudură în exploatare
FALSE POSITIVES (alarmă falsă): ACCEPTABIL → piesa este re-inspectată manual

Prioritate: Minimizare false negatives chiar dacă cresc false positives.
Soluție: Ajustare threshold clasificare de la 0.5 → 0.3 pentru clasa 'defect'.
```

**Analiza pentru proiectul Text-to-Blender:**
```
FALSE NEGATIVES (comandă nerecunoscută):
- Impact: Utilizatorul trebuie să reformuleze comanda
- Severitate: MEDIE - utilizatorul poate reîncerca cu alt text

FALSE POSITIVES (comandă incorect clasificată):
- Impact: Se generează cod Blender incorect
- Severitate: JOASĂ - utilizatorul poate vizualiza rezultatul și anula (Ctrl+Z)

Prioritate: Minimizare confuzii între comenzi destructive (delete_all) și 
comenzi constructive (create_*). Modelul actual NU confundă aceste categorii critice.

Top-3 accuracy de 82.91% arată că în 83% din cazuri, intenția corectă
este în primele 3 predicții - util pentru sistem de sugestii.
```

### 4. Ce măsuri corective propuneți?

**Exemplu clasificare imagini piese:**
```
Măsuri corective:
1. Colectare 500+ imagini adiționale pentru clasa minoritară 'zgârietură ușoară'
2. Implementare filtrare Gaussian blur pentru reducere zgomot cameră industrială
3. Augmentare perspective pentru simulare unghiuri camera variabile (±15°)
4. Re-antrenare cu class weights: [1.0, 2.5, 1.2] pentru echilibrare
```

**Măsuri propuse pentru Text-to-Blender:**
```
Măsuri corective implementabile:
1. AUGMENTARE DATE: Generare 50+ variante suplimentare pentru clasele minoritare
   (modifiers rari, comenzi complexe)

2. SINONIME: Extindere vocabular cu sinonime românești:
   - mută/mișcă/deplasează/translatează
   - rotește/întoarce/pivotează
   - creează/fă/generează/adaugă

3. N-GRAMS: Adăugare bigrams pentru context mai bun:
   - "cub mare" vs "cub basic" vs "cub roșu"

4. CLASS WEIGHTS: Aplicare weights inverse proporționale cu frecvența clasei
   pentru a penaliza mai mult erorile pe clase minoritare

5. ENSEMBLE: Combinare cu sistem bazat pe reguli pentru comenzi simple și
   frecvente (create_cube, delete_all) - fallback rapid
```

---

## Structura Repository-ului la Finalul Etapei 5

**Clarificare organizare:** Vom folosi **README-uri separate** pentru fiecare etapă în folderul `docs/`:

```
proiect-rn-[prenume-nume]/
├── README.md                           # Overview general proiect (actualizat)
├── etapa3_analiza_date.md         # Din Etapa 3
├── etapa4_arhitectura_sia.md      # Din Etapa 4
├── etapa5_antrenare_model.md      # ← ACEST FIȘIER (completat)
│
├── docs/
│   ├── state_machine.png              # Din Etapa 4
│   ├── loss_curve.png                 # NOU - Grafic antrenare
│   ├── confusion_matrix.png           # (opțional - Nivel 3)
│   └── screenshots/
│       ├── inference_real.png         # NOU - OBLIGATORIU
│       └── ui_demo.png                # Din Etapa 4
│
├── data/                               # Din Etapa 3-4 (NESCHIMBAT)
│   ├── raw/
│   ├── generated/                     # Contribuția voastră 40%
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── src/
│   ├── data_acquisition/              # Din Etapa 4
│   ├── preprocessing/                 # Din Etapa 3
│   │   └── combine_datasets.py        # NOU (dacă ați adăugat date în Etapa 4)
│   ├── neural_network/
│   │   ├── model.py                   # Din Etapa 4
│   │   ├── train.py                   # NOU - Script antrenare
│   │   └── evaluate.py                # NOU - Script evaluare
│   └── app/
│       └── main.py                    # ACTUALIZAT - încarcă model antrenat
│
├── models/
│   ├── untrained_model.h5             # Din Etapa 4
│   ├── trained_model.h5               # NOU - OBLIGATORIU
│   └── final_model.onnx               # (opțional - Nivel 3 bonus)
│
├── results/                            # NOU - Folder rezultate antrenare
│   ├── training_history.csv           # OBLIGATORIU - toate epoch-urile
│   ├── test_metrics.json              # Metrici finale pe test set
│   └── hyperparameters.yaml           # Hiperparametri folosiți
│
├── config/
│   └── preprocessing_params.pkl       # Din Etapa 3 (NESCHIMBAT)
│
├── requirements.txt                    # Actualizat
└── .gitignore
```

**Diferențe față de Etapa 4:**
- Adăugat `docs/etapa5_antrenare_model.md` (acest fișier)
- Adăugat `docs/loss_curve.png` (Nivel 2)
- Adăugat `models/trained_model.h5` - OBLIGATORIU
- Adăugat `results/` cu history și metrici
- Adăugat `src/neural_network/train.py` și `evaluate.py`
- Actualizat `src/app/main.py` să încarce model antrenat

---

## Instrucțiuni de Rulare (Actualizate față de Etapa 4)

### 1. Setup mediu (dacă nu ați făcut deja)

```bash
pip install -r requirements.txt
```

### 2. Pregătire date (DACĂ ați adăugat date noi în Etapa 4)

```bash
# Combinare + reprocesare dataset complet
python src/preprocessing/combine_datasets.py
python src/preprocessing/data_cleaner.py
python src/preprocessing/feature_engineering.py
python src/preprocessing/data_splitter.py --stratify --random_state 42
```

### 3. Antrenare model

```bash
python src/neural_network/train.py --epochs 50 --batch_size 32 --early_stopping

# Output așteptat:
# Epoch 1/50 - loss: 0.8234 - accuracy: 0.6521 - val_loss: 0.7891 - val_accuracy: 0.6823
# ...
# Epoch 23/50 - loss: 0.3456 - accuracy: 0.8234 - val_loss: 0.4123 - val_accuracy: 0.7956
# Early stopping triggered at epoch 23
# ✓ Model saved to models/trained_model.h5
```

### 4. Evaluare pe test set

```bash
python src/neural_network/evaluate.py --model models/trained_model.h5

# Output așteptat:
# Test Accuracy: 0.7823
# Test F1-score (macro): 0.7456
# ✓ Metrics saved to results/test_metrics.json
# ✓ Confusion matrix saved to docs/confusion_matrix.png
```

### 5. Lansare UI cu model antrenat

```bash
streamlit run src/app/main.py

# SAU pentru LabVIEW:
# Deschideți WebVI și rulați main.vi
```

**Testare în UI:**
1. Introduceți date de test (manual sau upload fișier)
2. Verificați că predicția este DIFERITĂ de Etapa 4 (când era random)
3. Verificați că confidence scores au sens (ex: 85% pentru clasa corectă)
4. Faceți screenshot → salvați în `docs/screenshots/inference_real.png`

---

## Checklist Final – Bifați Totul Înainte de Predare

### Prerequisite Etapa 4 (verificare)
- [x] State Machine există și e documentat în `docs/diagrams/state_machine.png`
- [x] Contribuție ≥40% date originale verificabilă - 100% date generate (1,560 samples)
- [x] Cele 3 module din Etapa 4 funcționale

### Preprocesare și Date
- [x] Dataset 100% original preprocesat cu `data_splitter.py`
- [x] Split train/val/test: 70/15/15% → 1,092 / 234 / 234 samples
- [x] Parametri preprocesare salvați în `config/preprocessing_params.pkl`

### Antrenare Model - Nivel 1 (OBLIGATORIU)
- [x] Model antrenat de la ZERO (nu fine-tuning pe model pre-antrenat)
- [x] Minimum 10 epoci rulate → 36 epoci (verificabil în `results/training_history.csv`)
- [x] Tabel hiperparametri + justificări completat în acest README
- [x] Metrici calculate pe test set: **Accuracy 75.64%** ≥65% ✅, **F1 0.6032** ≥0.60 ✅
- [x] Model salvat în `models/trained_model.pt`
- [x] `results/training_history.csv` există cu toate 36 epoch-urile

### Integrare UI și Demonstrație - Nivel 1 (OBLIGATORIU)
- [ ] Model ANTRENAT încărcat în UI din Etapa 4 (TODO: integrare API)
- [ ] UI face inferență REALĂ cu predicții corecte
- [ ] Screenshot inferență reală în `docs/screenshots/inference_real.png`
- [ ] Verificat: predicțiile sunt diferite față de Etapa 4

### Documentație Nivel 2 (dacă aplicabil)
- [x] Early stopping implementat și documentat în cod (patience=10)
- [ ] Learning rate scheduler folosit (ReduceLROnPlateau / StepLR)
- [ ] Augmentări relevante domeniu aplicate
- [x] Grafic loss/val_loss salvat în `results/training_curves.png`
- [x] Analiză erori în context industrial completată (4 întrebări răspunse)
- [x] Metrici Nivel 2: **Accuracy 75.64%** ≥75% ✅, **F1 0.6032** (sub 0.70)

### Documentație Nivel 3 Bonus (dacă aplicabil)
- [ ] Comparație 2+ arhitecturi (tabel comparativ + justificare)
- [ ] Export ONNX/TFLite + benchmark latență (<50ms demonstrat)
- [x] Confusion matrix salvată în `results/confusion_matrix.png`
- [x] Analiză erori (57 erori analizate în `results/error_analysis.csv`)

### Verificări Tehnice
- [x] `requirements.txt` actualizat cu toate bibliotecile
- [x] Toate path-urile RELATIVE (nu absolute)
- [x] Cod nou comentat în limba română (train.py, evaluate.py)
- [x] `git log` arată commit-uri incrementale (multiple commits Etapa 5)
- [x] Verificare anti-plagiat: model creat de la zero, date 100% originale

### Verificare State Machine (Etapa 4)
- [x] Fluxul de inferență respectă stările din State Machine
- [x] Toate stările critice definite (INPUT → PREPROCESS → INFERENCE → OUTPUT)
- [ ] UI reflectă State Machine-ul pentru utilizatorul final (TODO)

### Pre-Predare
- [x] `docs/etapa5_antrenare_model.md` creat cu rezultatele
- [x] Structură repository conformă: `results/`, `models/` populate
- [x] Commit: `"Etapa 5 completă – Accuracy=75.64%, F1=0.6032"`
- [x] Tag: `git tag -a v0.5-model-trained` ✅
- [x] Push: `git push origin main --tags` ✅
- [x] Repository public pe GitHub

---

## Livrabile Obligatorii (Nivel 1)

Asigurați-vă că următoarele fișiere există și sunt completate:

1. **`docs/etapa5_antrenare_model.md`** (acest fișier) cu:
   - Tabel hiperparametri + justificări (complet)
   - Metrici test set raportate (accuracy, F1)
   - (Nivel 2) Analiză erori context industrial (4 paragrafe)

2. **`models/trained_model.h5`** (sau `.pt`, `.lvmodel`) - model antrenat funcțional

3. **`results/training_history.csv`** - toate epoch-urile salvate

4. **`results/test_metrics.json`** - metrici finale:

Exemplu:
```json
{
  "test_accuracy": 0.7823,
  "test_f1_macro": 0.7456,
  "test_precision_macro": 0.7612,
  "test_recall_macro": 0.7321
}
```

5. **`docs/screenshots/inference_real.png`** - demonstrație UI cu model antrenat

6. **(Nivel 2)** `docs/loss_curve.png` - grafic loss vs val_loss

7. **(Nivel 3)** `docs/confusion_matrix.png` + analiză în README

---

## Predare și Contact

**Predarea se face prin:**
1. Commit pe GitHub: `"Etapa 5 completă – Accuracy=X.XX, F1=X.XX"`
2. Tag: `git tag -a v0.5-model-trained -m "Etapa 5 - Model antrenat"`
3. Push: `git push origin main --tags`

---

**Mult succes! Această etapă demonstrează că Sistemul vostru cu Inteligență Artificială (SIA) funcționează în condiții reale!**