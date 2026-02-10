# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Caldararu Denisa Elena  
**Link Repository GitHub:** https://github.com/taquitohh/Proiect_RN  
**Data predării:** 10.01.2026

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Obiectiv principal:** Antrenarea efectivă a modelului RN definit în Etapa 4, evaluarea performanței și integrarea în aplicația completă.

**Pornire obligatorie:** Arhitectura completă și funcțională din Etapa 4:
- State Machine definit și justificat
- Cele 3 module funcționale (Data Logging, RN, UI)
- Minimum 40% date originale în dataset

---

### 5.1 Pregătirea antrenării modelului RN

În această etapă a fost pregătit pipeline-ul complet pentru antrenarea
modelului de Rețea Neuronală. Datele utilizate sunt complet preprocesate,
împărțite stratificat în seturi de antrenare, validare și test, conform
bunelor practici pentru evitarea scurgerii de informație.

Problema este formulată ca o clasificare multi-clasă supravegheată. Pentru
chair sunt 4 clase, iar pentru table/cabinet/stove sunt 3 clase, respectiv
pentru fridge 2 clase, folosind un model de tip MLP (Multilayer Perceptron)
implementat în TensorFlow/Keras.

Au fost definiți hiperparametrii inițiali (baseline), care vor fi folosiți
pentru prima antrenare și vor constitui punctul de referință pentru
optimizările ulterioare din Etapa 6.

---

## PREREQUISITE – Verificare Etapa 4 (OBLIGATORIU)

**Înainte de a începe Etapa 5, verificați că aveți din Etapa 4:**

- [x] **State Machine** definit și documentat în `docs/state_machine.*`
- [x] **Contribuție ≥40% date originale** în `data/generated/` (verificabil)
- [x] **Modul 1 (Data Logging)** funcțional - produce CSV-uri
- [x] **Modul 2 (RN)** cu arhitectură definită dar NEANTRENATĂ (`models/untrained_model.h5`)
- [x] **Modul 3 (UI/Web Service)** funcțional cu model dummy
- [x] **Tabelul "Nevoie → Soluție → Modul"** complet în README Etapa 4

** Dacă oricare din punctele de mai sus lipsește → reveniți la Etapa 4 înainte de a continua.**

---

## Pregătire Date pentru Antrenare 

### Dacă ați adăugat date noi în Etapa 4 (contribuția de 40%):

**TREBUIE să refaceți preprocesarea pe dataset-ul COMBINAT:**

Exemplu:
```bash
# 1. Curățare date
python src/preprocessing/chair_data_cleaner.py

# 2. Scalare StandardScaler
python src/preprocessing/chair_feature_scaler.py

# 3. Împărțire stratificată
python src/preprocessing/chair_data_splitter.py

# Verificare finală:
# data/chairs/train/ → trebuie să conțină date vechi + noi
# data/chairs/validation/ → trebuie să conțină date vechi + noi
# data/chairs/test/ → trebuie să conțină date vechi + noi
```

** ATENȚIE - Folosiți ACEIAȘI parametri de preprocesare:**
- Același `scaler` salvat în `config/chair_scaler.pkl`
- Aceiași proporții split: 70% train / 15% validation / 15% test
- Același `random_state=42` pentru reproducibilitate

**Verificare rapidă:**
```python
import pandas as pd
train = pd.read_csv('data/chairs/train/X_train.csv')
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
6. **Salvare model antrenat** în `models/chair_model.h5` (Keras/TensorFlow) sau `.pt` (PyTorch) sau `.lvmodel` (LabVIEW)
7. **Integrare în UI din Etapa 4:**
   - UI trebuie să încarce modelul ANTRENAT (nu dummy)
   - Inferență REALĂ demonstrată
   - Screenshot în `docs/screenshots/inference_real.png`

#### Tabel Hiperparametri și Justificări (OBLIGATORIU - Nivel 1)

Completați tabelul cu hiperparametrii folosiți și **justificați fiecare alegere**:

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
|--------------------|-------------------|-----------------|
| Learning rate | 0.001 | Valoare standard pentru Adam optimizer, asigură convergență stabilă |
| Batch size | 32 | Compromis memorie/stabilitate pentru setul de date tabular |
| Number of epochs | 50 | Early stopping (patience=5) oprește când `val_loss` nu scade |
| Optimizer | Adam | Adaptive learning rate, stabil pentru MLP cu 2 straturi ascunse |
| Loss function | Sparse Categorical Crossentropy | Clasificare multi-class cu 4 clase pentru chair |
| Activation functions | ReLU (hidden), Softmax (output) | ReLU pentru non-linearitate, Softmax pentru probabilități clase |

**Justificare detaliată batch size:**
```
Am ales batch_size=32 pentru că avem N=15,000 samples → 15,000/32 ≈ 469 iterații/epocă.
Aceasta oferă un echilibru între:
- Stabilitate gradient (batch prea mic → zgomot mare în gradient)
- Memorie (batch prea mare → consum ridicat)
- Timp antrenare (batch 32 permite rulare rapida la 10 epoci)
```

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

Implementare in proiect:
- Early stopping pe `val_loss` cu `patience=5` si `restore_best_weights=True`.
- Scheduler `ReduceLROnPlateau` (factor 0.5, `patience=3`).
- Augmentare tabulara usoara: zgomot gaussian pe feature-uri continue (2% din std).

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
| Export ONNX/TFLite + benchmark latență | Fișier `models/chair_model.onnx` + demonstrație <50ms |
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
| `ACQUIRE_DATA` | Citire batch date din `data/chairs/train/` pentru antrenare |
| `PREPROCESS` | Aplicare scaler salvat din `config/chair_scaler.pkl` |
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
model = keras.models.load_model('models/chair_model.h5')  # weights antrenate
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

**Completați pentru proiectul vostru:**
```
Confusion Matrix indică cele mai multe confuzii între Bar Chair și Simple Chair.
Un al doilea tip de confuzie este între Chair with Backrest și Bar Chair.
În exemplele greșite, înălțimea șezutului este ridicată, dar backrest_height rămâne modest,
ceea ce face Bar Chair să fie perceput ca Simple Chair în unele cazuri.
```

### 2. Ce caracteristici ale datelor cauzează erori?

**Exemplu vibrații motor:**
```
Modelul eșuează când zgomotul de fond depășește 40% din amplitudinea semnalului util.
În mediul industrial, acest nivel de zgomot apare când mai multe motoare funcționează simultan.
```

**Completați pentru proiectul vostru:**
```
Erorile apar când seat_height este mare, dar backrest_height este mic (backrest scurt),
respectiv când leg_count este atipic (3 sau 5), ceea ce reduce separabilitatea între clase.
```

### 3. Ce implicații are pentru aplicația industrială?

**Exemplu detectare defecte sudură:**
```
FALSE NEGATIVES (defect nedetectat): CRITIC → risc rupere sudură în exploatare
FALSE POSITIVES (alarmă falsă): ACCEPTABIL → piesa este re-inspectată manual

Prioritate: Minimizare false negatives chiar dacă cresc false positives.
Soluție: Ajustare threshold clasificare de la 0.5 → 0.3 pentru clasa 'defect'.
```

**Completați pentru proiectul vostru:**
```
Impactul este limitat la selecția greșită a tipului de scaun în UI, fără consecințe critice.
În aplicație, prioritatea rămâne demonstrarea corectitudinii pipeline-ului end-to-end.
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

**Completați pentru proiectul vostru:**
```
1. Creșterea separării în generator pentru seat_height și backrest_height la Bar Chair.
2. Adăugarea de feature-uri derivate (ex: raport backrest_height / seat_height).
3. Creșterea numărului de exemple pentru clasele cu leg_count atipic (3 sau 5).
4. Ajustare class weights pentru a penaliza mai mult confuziile Bar Chair -> Simple Chair.
```

---

## Structura Repository-ului la Finalul Etapei 5

Structura reala din proiect este:

```
Proiect_RN/
├── README – Etapa 3 -Analiza si Pregatirea Setului de Date pentru Retele Neuronale.md
├── README_Etapa4_Arhitectura_SIA_03.12.2025.md
├── README_Etapa5_Antrenare_RN.md
├── README_Etape6_Analiza_Performantei_Optimizare_Concluzii.md
├── ORDERINE_RULARE.txt
├── docs/
│   ├── state_machine.png
│   ├── confusion_matrix.png
│   ├── loss_curve.png
│   └── screenshots/
├── data/
│   ├── README.md
│   ├── cabinets/
│   ├── chairs/
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   ├── fridges/
│   ├── generated/
│   ├── processed/
│   ├── raw/
│   ├── stoves/
│   └── tables/
├── src/
│   ├── data_acquisition/
│   ├── preprocessing/
│   ├── neural_network/
│   └── app/
├── models/
│   ├── untrained_model.h5
│   ├── chair_model.h5
│   ├── chair_model.onnx
│   ├── table_model.h5
│   ├── cabinet_model.h5
│   ├── fridge_model.h5
│   └── stove_model.h5
├── results/
│   ├── chair_training_history.csv
│   ├── chair_test_metrics.json
│   ├── hyperparameters.yaml
│   ├── table_training_history.csv
│   ├── table_training_metrics.json
│   ├── cabinet_training_history.csv
│   ├── cabinet_training_metrics.json
│   ├── fridge_training_history.csv
│   ├── fridge_training_metrics.json
│   ├── stove_training_history.csv
│   └── stove_training_metrics.json
├── config/
│   ├── chair_scaler.pkl
│   ├── table_scaler.pkl
│   ├── cabinet_scaler.pkl
│   ├── fridge_scaler.pkl
│   └── stove_scaler.pkl
├── requirements.txt
└── .gitignore
```


**Diferențe față de Etapa 4:**
- Adăugat `docs/etapa5_antrenare_model.md` (acest fișier)
- Adăugat `docs/loss_curve.png` (Nivel 2)
- Adăugat `models/chair_model.h5` - OBLIGATORIU
- Adăugat `models/chair_model.onnx` - Export ONNX (Nivel 3)
- Adăugat `results/` cu history și metrici
- Adăugat `src/neural_network/train_chair.py`, `evaluate.py`, `compare_architectures.py`, `export_onnx.py`
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
python src/preprocessing/chair_data_cleaner.py
python src/preprocessing/chair_feature_scaler.py
python src/preprocessing/chair_data_splitter.py
```

### 3. Antrenare model

```bash
python src/neural_network/train_chair.py --epochs 50 --batch_size 32 --early_stopping

# Output așteptat:
# Epoch 1/50 - loss: 0.8234 - accuracy: 0.6521 - val_loss: 0.7891 - val_accuracy: 0.6823
# ...
# Epoch 23/50 - loss: 0.3456 - accuracy: 0.8234 - val_loss: 0.4123 - val_accuracy: 0.7956
# Early stopping triggered at epoch 23
# ✓ Model saved to models/chair_model.h5
```

### 4. Evaluare pe test set

```bash
python src/neural_network/evaluate.py

# Output așteptat:
# Test Accuracy: 0.9911
# Test F1-score (macro): 0.9915
# ✓ Metrics saved to results/chair_test_metrics.json
# ✓ Confusion matrix salvată în docs/confusion_matrix.png
```

### 5. Lansare UI cu model antrenat

```bash
python src/app/main.py

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
- [x] State Machine există și e documentat în `docs/state_machine.*`
- [x] Contribuție ≥40% date originale verificabilă în `data/generated/`
- [x] Cele 3 module din Etapa 4 funcționale

### Preprocesare și Date
- [x] Dataset combinat (vechi + nou) preprocesat (dacă ați adăugat date)
- [x] Split train/val/test: 70/15/15% (verificat dimensiuni fișiere)
- [x] Scaler din Etapa 3 folosit consistent (`config/chair_scaler.pkl`)

### Antrenare Model - Nivel 1 (OBLIGATORIU)
- [x] Model antrenat de la ZERO (nu fine-tuning pe model pre-antrenat)
- [x] Minimum 10 epoci rulate (verificabil în `results/chair_training_history.csv`)
- [x] Tabel hiperparametri + justificări completat în acest README
- [x] Metrici calculate pe test set: **Accuracy ≥65%**, **F1 ≥0.60**
- [x] Model salvat în `models/chair_model.h5` (sau .pt, .lvmodel)
- [x] `results/chair_training_history.csv` există cu toate epoch-urile

### Integrare UI și Demonstrație - Nivel 1 (OBLIGATORIU)
- [x] Model ANTRENAT încărcat în UI din Etapa 4 (nu model dummy)
- [x] UI face inferență REALĂ cu predicții corecte
- [x] Screenshot inferență reală în `docs/screenshots/inference_real.png`
- [x] Verificat: predicțiile sunt diferite față de Etapa 4 (când erau random)

### Documentație Nivel 2 (dacă aplicabil)
- [x] Early stopping implementat și documentat în cod
- [x] Learning rate scheduler folosit (ReduceLROnPlateau / StepLR)
- [x] Augmentări relevante domeniu aplicate (NU rotații simple!)
- [x] Grafic loss/val_loss salvat în `docs/loss_curve.png`
- [x] Analiză erori în context industrial completată (4 întrebări răspunse)
- [x] Metrici Nivel 2: **Accuracy ≥75%**, **F1 ≥0.70**

### Documentație Nivel 3 Bonus (dacă aplicabil)
- [x] Confusion matrix generată în `docs/confusion_matrix.png`
- [x] Comparație 2+ arhitecturi (tabel comparativ + justificare)
- [x] Export ONNX/TFLite + benchmark latență (<50ms demonstrat)
- [x] Confusion matrix + analiză 5 exemple greșite cu implicații

### Verificări Tehnice
- [x] `requirements.txt` actualizat cu toate bibliotecile noi
- [x] Toate path-urile RELATIVE (nu absolute: `/Users/...` )
- [x] Cod nou comentat în limba română sau engleză (minimum 15%)
- [x] `git log` arată commit-uri incrementale (NU 1 commit gigantic)
- [x] Verificare anti-plagiat: toate punctele 1-5 respectate

### Verificare State Machine (Etapa 4)
- [x] Fluxul de inferență respectă stările din State Machine
- [x] Toate stările critice (PREPROCESS, INFERENCE, ALERT) folosesc model antrenat
- [x] UI reflectă State Machine-ul pentru utilizatorul final

### Pre-Predare
- [x] `docs/etapa5_antrenare_model.md` completat cu TOATE secțiunile
- [x] Structură repository conformă: `docs/`, `results/`, `models/` actualizate
- [x] Commit: `"Etapa 5 completă – Accuracy=0.9911, F1=0.9915"`
- [x] Tag: `git tag -a v0.5-model-trained -m "Etapa 5 - Model antrenat"`
- [x] Push: `git push origin main --tags`
- [x] Repository accesibil (public sau privat cu acces profesori)

---

## Livrabile Obligatorii (Nivel 1)

Asigurați-vă că următoarele fișiere există și sunt completate:

1. **`docs/etapa5_antrenare_model.md`** (acest fișier) cu:
   - Tabel hiperparametri + justificări (complet)
   - Metrici test set raportate (accuracy, F1)
   - (Nivel 2) Analiză erori context industrial (4 paragrafe)

2. **`models/chair_model.h5`** (sau `.pt`, `.lvmodel`) - model antrenat funcțional

3. **`results/chair_training_history.csv`** - toate epoch-urile salvate

4. **`results/chair_test_metrics.json`** - metrici finale:

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
1. Commit pe GitHub: `"Etapa 5 completă – Accuracy=0.9911, F1=0.9915"`
2. Tag: `git tag -a v0.5-model-trained -m "Etapa 5 - Model antrenat"`
3. Push: `git push origin main --tags`

---

**Mult succes! Această etapă demonstrează că Sistemul vostru cu Inteligență Artificială (SIA) funcționează în condiții reale!**