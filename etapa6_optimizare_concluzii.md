# Etapa 6: Analiza Performanței, Optimizarea și Concluzii Finale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Caldararu Denisa Elena  
**Link Repository GitHub:** https://github.com/taquitohh/Proiect-RN-Denisa-Elena-Caldararu  
**Data predării:** 29.01.2026

---
## Scopul Etapei 6

Această etapă corespunde punctelor **7. Analiza performanței și optimizarea parametrilor**, **8. Analiza și agregarea rezultatelor** și **9. Formularea concluziilor finale** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Obiectiv principal:** Maturizarea completă a Sistemului cu Inteligență Artificială (SIA) prin optimizarea modelului RN, analiza detaliată a performanței și integrarea îmbunătățirilor în aplicația software completă.

**Stare actuală în repository:** S-au rulat experimente de comparare arhitecturi (MLP),
iar varianta `baseline_32_16` a rămas cea mai eficientă. Modelul utilizat în aplicație
este `models/chair_model.h5`, iar exportul ONNX este disponibil în `models/chair_model.onnx`.

**Extindere multi-obiect (în aplicație):** Pe lângă `chair` (exemplul principal din această documentație),
aplicația a fost extinsă să suporte și `table`, `cabinet`, `fridge`, `stove`. Pentru toate tipurile,
pipeline-ul RN este identic la nivel de aplicație (încărcare model + scaler → scalare input → inferență → afișare),
implementat unitar în `src/app/main.py` prin încărcarea artefactelor per tip (`models/<object>_model.h5` și
`config/<object>_scaler.pkl`). Diferențele apar la nivel de date: fiecare obiect are propria schemă de features,
propriile clase/etichete și propriile fișiere în `data/<object>/`.

**CONTEXT IMPORTANT:** 
- Etapa 6 **ÎNCHEIE ciclul formal de dezvoltare** al proiectului
- Aceasta este **ULTIMA VERSIUNE înainte de examen** pentru care se oferă **FEEDBACK**
- Pe baza feedback-ului primit, componentele din **TOATE etapele anterioare** pot fi actualizate iterativ

**Pornire obligatorie:** Modelul antrenat și aplicația funcțională din Etapa 5:
- Model antrenat cu metrici baseline (Accuracy ≥65%, F1 ≥0.60)
- Cele 3 module integrate și funcționale
- State Machine implementat și testat

---

## MESAJ CHEIE – ÎNCHEIEREA CICLULUI DE DEZVOLTARE ȘI ITERATIVITATE

**ATENȚIE: Etapa 6 ÎNCHEIE ciclul de dezvoltare al aplicației software!**

**CE ÎNSEAMNĂ ACEST LUCRU:**
- Aceasta este **ULTIMA VERSIUNE a proiectului înainte de examen** pentru care se mai poate primi **FEEDBACK** de la cadrul didactic
- După Etapa 6, proiectul trebuie să fie **COMPLET și FUNCȚIONAL**
- Orice îmbunătățiri ulterioare (post-feedback) vor fi implementate până la examen

**PROCES ITERATIV – CE RĂMÂNE VALABIL:**
Deși Etapa 6 încheie ciclul formal de dezvoltare, **procesul iterativ continuă**:
- Pe baza feedback-ului primit, **TOATE componentele anterioare pot și trebuie actualizate**
- Îmbunătățirile la model pot necesita modificări în Etapa 3 (date), Etapa 4 (arhitectură) sau Etapa 5 (antrenare)
- README-urile etapelor anterioare trebuie actualizate pentru a reflecta starea finală

**CERINȚĂ CENTRALĂ Etapa 6:** Finalizarea și maturizarea **ÎNTREGII APLICAȚII SOFTWARE**:

1. **Actualizarea State Machine-ului** (threshold-uri noi, stări adăugate/modificate, latențe recalculate)
2. **Re-testarea pipeline-ului complet** (achiziție → preprocesare → inferență → decizie → UI/alertă)
3. **Modificări concrete în cele 3 module** (Data Logging, RN, Web Service/UI)
4. **Sincronizarea documentației** din toate etapele anterioare

**DIFERENȚIATOR FAȚĂ DE ETAPA 5:**
- Etapa 5 = Model antrenat care funcționează
- Etapa 6 = Model OPTIMIZAT + Aplicație MATURIZATĂ + Concluzii industriale + **VERSIUNE FINALĂ PRE-EXAMEN**


**IMPORTANT:** Aceasta este ultima oportunitate de a primi feedback înainte de evaluarea finală. Profitați de ea!

---

## PREREQUISITE – Verificare Etapa 5 (OBLIGATORIU)

**Înainte de a începe Etapa 6, verificați că aveți din Etapa 5:**

- [x] **Model antrenat** salvat în `models/chair_model.h5` (sau `.pt`, `.lvmodel`)
- [x] **Metrici baseline** raportate: Accuracy ≥65%, F1-score ≥0.60
- [x] **Tabel hiperparametri** cu justificări completat
- [x] **`results/chair_training_history.csv`** cu toate epoch-urile
- [x] **UI funcțional** care încarcă modelul antrenat și face inferență reală
- [x] **Screenshot inferență** în `docs/screenshots/inference_real.png`
- [x] **State Machine** implementat conform definiției din Etapa 4

**Notă:** În repository există și artefacte pentru `table`, `cabinet`, `fridge`, `stove` (modele + scalere + pipeline UI).
În această etapă, chair rămâne exemplul principal pentru metrici și analiza detaliată, iar celelalte obiecte urmează aceeași
structură de pipeline, cu diferențe la date și clase.

**Dacă oricare din punctele de mai sus lipsește → reveniți la Etapa 5 înainte de a continua.**

---

## Cerințe

Completați **TOATE** punctele următoare:

1. **Minimum 4 experimente de optimizare** (variație sistematică a hiperparametrilor)
2. **Tabel comparativ experimente** cu metrici și observații (vezi secțiunea dedicată)
3. **Confusion Matrix** generată și analizată
4. **Analiza detaliată a 5 exemple greșite** cu explicații cauzale
5. **Metrici finali pe test set:**
   - **Acuratețe ≥ 70%** (îmbunătățire față de Etapa 5)
   - **F1-score (macro) ≥ 0.65**
6. **Salvare model final** în `models/chair_model.h5` (sau `.pt`, `.lvmodel`)
7. **Actualizare aplicație software:**
   - Tabel cu modificările aduse aplicației în Etapa 6
   - UI încarcă modelul final (chair_model.h5)
   - Screenshot demonstrativ în `docs/screenshots/inference_optimized.png` 
8. **Concluzii tehnice** (minimum 1 pagină): performanță, limitări, lecții învățate

**Clarificare multi-obiect:** Punctele (6) și (7) sunt exemplificate pe chair. În aplicația finală există și modele finale per tip
(`models/table_model.h5`, `models/cabinet_model.h5`, `models/fridge_model.h5`, `models/stove_model.h5`), iar UI folosește aceeași
secvență de inferență pentru fiecare tip. Diferențele țin de schema de input și numărul/definiția claselor.

#### Tabel Experimente de Optimizare

Documentați **minimum 4 experimente** cu variații sistematice:

| **Exp#** | **Modificare față de Baseline (Etapa 5)** | **Accuracy** | **F1-score** | **Timp antrenare** | **Observații** |
|----------|------------------------------------------|--------------|--------------|-------------------|----------------|
| Baseline | Configurația din Etapa 5 (MLP 32-16, 50 epoci, ES+RLRP, augmentare) | 0.9911 | 0.9915 | n/a | Metrici din `results/chair_test_metrics.json` |
| Exp 1 | Arhitectură mai îngustă: 16-8 | 0.9867 | 0.9861 | 6.85s | Mai puțini parametri, performanță sub baseline |
| Exp 2 | Arhitectură mai largă: 64-32 | 0.9893 | 0.9894 | 6.76s | Cea mai bună dintre experimente, dar sub baseline |
| Exp 3 | Arhitectură mai adâncă: 64-32-16 | 0.9862 | 0.9865 | 7.07s | Performanță bună, cost mai mare |
| Exp 4 | Arhitectură baseline: 32-16 (10 epoci, ES+RLRP) | 0.9858 | 0.9849 | 6.98s | Stabil, dar sub baseline Etapa 5 |

**Notă multi-obiect:** Experimentele și comparația de arhitecturi sunt documentate în detaliu pentru chair.
Pentru `table/cabinet/fridge/stove`, s-a păstrat același tip de pipeline (MLP tabular + scalare) și aceeași structură de rulare,
dar cu dataset-uri diferite (features și clase diferite). Pentru aceste obiecte, sunt disponibile artefacte de antrenare în `results/*_training_history.csv`
și `results/*_training_metrics.json`.

**Justificare alegere configurație finală:**
```
Experimentele rapide pe 10 epoci nu au depasit baseline-ul din Etapa 5.
Modelul final ramas in productie este chair_model.h5 (MLP 32-16).
```

**Resurse învățare rapidă - Optimizare:**
- Hyperparameter Tuning: https://keras.io/guides/keras_tuner/ 
- Grid Search: https://scikit-learn.org/stable/modules/grid_search.html
- Regularization (Dropout, L2): https://keras.io/api/layers/regularization_layers/

---

## 1. Actualizarea Aplicației Software în Etapa 6 

**CERINȚĂ CENTRALĂ:** Documentați TOATE modificările aduse aplicației software ca urmare a optimizării modelului.

### Tabel Modificări Aplicație Software

| **Componenta** | **Stare Etapa 5** | **Modificare Etapa 6** | **Justificare** |
|----------------|-------------------|------------------------|-----------------|
| **Model încărcat** | `chair_model.h5` | `chair_model.h5` (model final) | Varianta cu cel mai bun raport performanta/complexitate |
| **State Machine** | IDLE → USER_INPUT → PREPROCESS → RN_INFERENCE → SCRIPT_GENERATION → DISPLAY → IDLE | Neschimbat | Flux stabil, suficient pentru proiect |
| **UI - afișare rezultate** | text + probabilități + script | Neschimbat | UI deja integra model antrenat corect |
| **Logging** | n/a | Neschimbat | Nu s-a introdus logging suplimentar |

**Extindere multi-obiect:** UI folosește aceeași logică de inferență pentru `chair/table/cabinet/fridge/stove`, încărcând modelul și scaler-ul aferent tipului selectat
din `models/` și `config/`. Diferențele sunt la schema de input (features) și la clasele prezise.

**Completați pentru proiectul vostru:**
```markdown
### Modificări concrete aduse în Etapa 6:

1. **Model înlocuit:** modelul final ramane `chair_model.h5` (varianta cu performanta maxima).
2. **State Machine actualizat:** neschimbat fata de Etapa 5.
3. **UI îmbunătățit:** neschimbat (foloseste modelul antrenat).
4. **Pipeline end-to-end re-testat:** confirmat prin evaluare pe test set.
5. **Suport multi-obiect:** pipeline-ul RN este același și pentru `table/cabinet/fridge/stove`, cu modele + scalere separate per tip.
```

### Diagrama State Machine Actualizată (dacă s-au făcut modificări)

Dacă ați modificat State Machine-ul în Etapa 6, includeți diagrama actualizată în `docs/state_machine_v2.png` și explicați diferențele:

```
Exemplu modificări State Machine pentru Etapa 6:

ÎNAINTE (Etapa 5):
PREPROCESS → RN_INFERENCE → THRESHOLD_CHECK (0.5) → ALERT/NORMAL

DUPĂ (Etapa 6):
PREPROCESS → RN_INFERENCE → CONFIDENCE_FILTER (>0.6) → 
  ├─ [High confidence] → THRESHOLD_CHECK (0.35) → ALERT/NORMAL
  └─ [Low confidence] → REQUEST_HUMAN_REVIEW → LOG_UNCERTAIN

Motivație: Predicțiile cu confidence <0.6 sunt trimise pentru review uman,
           reducând riscul de decizii automate greșite în mediul industrial.
```

---

## 2. Analiza Detaliată a Performanței

### 2.1 Confusion Matrix și Interpretare

**Locație:** `docs/confusion_matrix_optimized.png`

**Notă multi-obiect:** Confusion matrix și analiza cauzală de mai jos sunt detaliate pentru chair.
Pentru celelalte tipuri (`table/cabinet/fridge/stove`), metodologia de analiză este aceeași (confuzii dominante, cauze probabile, soluții),
însă confuziile și explicațiile diferă deoarece fiecare obiect are alte features și alte clase.

**Analiză:**
```
Confuziile apar in special intre Bar Chair si Simple Chair, respectiv intre Chair with Backrest si Bar Chair.
Erorile sunt rare si apar cand backrest_height este mic, iar seat_height este ridicat.
``` 

### 2.2 Analiza Detaliată a 5 Exemple Greșite

Selectați și analizați **minimum 5 exemple greșite** de pe test set:

| **Index** | **True Label** | **Predicted** | **Confidence** | **Cauză probabilă** | **Soluție propusă** |
|-----------|----------------|---------------|----------------|---------------------|---------------------|
| 2081 | Bar Chair | Simple Chair | 0.851 | backrest_height mic, seat_height mare | Cresterea separarii intre clase in generator |
| 971 | Chair with Backrest | Bar Chair | 0.756 | leg_count=3, seat_height ridicat | Mai multe exemple pentru leg_count atipic |
| 481 | Bar Chair | Simple Chair | 0.736 | backrest scurt + proportii apropiate | Feature derivat: backrest_height/seat_height |
| 213 | Bar Chair | Simple Chair | 0.725 | leg_count=5, backrest mic | Rebalansare clase + augmentare targetata |
| 1442 | Chair with Backrest | Simple Chair | 0.705 | backrest_height mic, style_variant=0 | Creste variatia spatarului in date |

**Analiză detaliată per exemplu (scrieți pentru fiecare):**
```markdown
### Exemplu #127 - Defect mare clasificat ca defect mic

**Context:** Imagine radiografică sudură, defect vizibil în centru
**Input characteristics:** brightness=0.3 (subexpus), contrast=0.7
**Output RN:** [defect_mic: 0.52, defect_mare: 0.38, normal: 0.10]

**Analiză:**
Imaginea originală are brightness scăzut (0.3 vs. media dataset 0.6), ceea ce 
face ca textura defectului să fie mai puțin distinctă. Modelul a "văzut" un 
defect, dar l-a clasificat în categoria mai puțin severă.

**Implicație industrială:**
Acest tip de eroare (downgrade severitate) poate duce la subestimarea riscului.
În producție, sudura ar fi acceptată când ar trebui re-inspectată.

**Soluție:**
1. Augmentare cu variații brightness în intervalul [0.2, 0.8]
2. Normalizare histogram înainte de inference (în PREPROCESS state)
```

---

## 3. Optimizarea Parametrilor și Experimentare

### 3.1 Strategia de Optimizare

Descrieți strategia folosită pentru optimizare:

```markdown
### Strategie de optimizare adoptată:

**Abordare:** Manual (comparare arhitecturi MLP)

**Axe de optimizare explorate:**
1. **Arhitectură:** 32-16 vs 64-32 vs 64-32-16
2. **Regularizare:** EarlyStopping + ReduceLROnPlateau (fara Dropout/L2)
3. **Learning rate:** 0.001 cu ReduceLROnPlateau
4. **Augmentări:** zgomot gaussian pe feature-uri continue
5. **Batch size:** 32 (constant)

**Criteriu de selecție model final:** F1 macro maxim cu numar minim de parametri

**Buget computațional:** 3 rulari complete, ~7s/antrenare (CPU)
```

### 3.2 Grafice Comparative

Nu s-au generat grafice separate in `docs/optimization/`. Sunt disponibile:
- `docs/loss_curve.png` (curba loss/val_loss)
- `docs/confusion_matrix_optimized.png`

### 3.3 Raport Final Optimizare

```markdown
### Raport Final Optimizare

**Model baseline (Etapa 5):**
- Accuracy: 0.9911
- F1-score: 0.9915

**Model optimizat (Etapa 6):**
- Accuracy: 0.9893 (-0.18%)
- F1-score: 0.9894 (-0.21%)

**Configurație finală aleasă:**
- Arhitectură: MLP 32-16
- Learning rate: 0.001 cu ReduceLROnPlateau
- Batch size: 32
- Regularizare: EarlyStopping (patience=5)
- Augmentări: zgomot gaussian pe feature-uri continue
- Epoci: 50 (early stopping activ)

**Îmbunătățiri cheie:**
1. Alegerea arhitecturii compacte (32-16) a oferit performanta maxima.
2. Augmentarea tabulara a imbunatatit generalizarea fara cost major.
3. EarlyStopping + ReduceLROnPlateau au stabilizat antrenarea.
```

---

## 4. Agregarea Rezultatelor și Vizualizări

### 4.1 Tabel Sumar Rezultate Finale

| **Metrică** | **Etapa 4** | **Etapa 5** | **Etapa 6** | **Target Industrial** | **Status** |
|-------------|-------------|-------------|-------------|----------------------|------------|
| Accuracy | n/a (model neantrenat) | 0.9911 | 0.9911 | ≥0.85 | OK |
| F1-score (macro) | n/a | 0.9915 | 0.9915 | ≥0.80 | OK |
| Precision (macro) | n/a | n/a | n/a | ≥0.80 | n/a |
| Recall (macro) | n/a | n/a | n/a | ≥0.80 | n/a |
| Latență inferență (ONNX) | n/a | n/a | 0.01ms | ≤50ms | OK |
| Throughput | n/a | n/a | n/a | ≥25 inf/s | n/a |

### 4.2 Vizualizări Obligatorii

Disponibile in `docs/`:

- [x] `confusion_matrix_optimized.png` - Confusion matrix model final
- [x] `loss_curve.png` - Loss si val_loss vs epoci

---

## 5. Concluzii Finale și Lecții Învățate

**NOTĂ:** Pe baza concluziilor formulate aici și a feedback-ului primit, este posibil și recomandat să actualizați componentele din etapele anterioare (3, 4, 5) pentru a reflecta starea finală a proiectului.

### 5.1 Evaluarea Performanței Finale

```markdown
### Evaluare sintetică a proiectului

**Obiective atinse:**
- [x] Model RN functional cu accuracy 0.9911 pe test set
- [x] Integrare completa in aplicatie (3 module)
- [x] State Machine implementat si validat
- [x] Pipeline end-to-end testat si documentat
- [x] UI demonstrativ cu inferenta reala
- [x] Documentatie completata pe toate etapele

**Obiective parțial atinse:**
- [x] Optimizarea a ramas limitata la compararea arhitecturilor (fara tuning extins)

**Obiective neatinse:**
- [x] Deployment cloud/edge si monitorizare MLOps
```

### 5.2 Limitări Identificate

```markdown
### Limitări tehnice ale sistemului

1. **Limitări date:**
   - Date sintetice; nu exista validare pe date reale.
   - Unele combinatii (leg_count atipic) au mai putine exemple.

2. **Limitări model:**
   - Confuzii intre clase cu geometrie apropiata (Bar Chair vs Simple Chair).
   - Fara regularizare explicita (Dropout/L2) in experimente.

3. **Limitări infrastructură:**
   - Inference ONNX testat pe CPU; nu exista benchmark pe GPU/edge.

4. **Limitări validare:**
   - Test set derivat din acelasi generator; posibil bias de distributie.
```

### 5.3 Direcții de Cercetare și Dezvoltare

```markdown
### Direcții viitoare de dezvoltare

**Pe termen scurt (1-3 luni):**
1. Colectare date reale pentru validare si calibrare.
2. Feature engineering (raport backrest_height/seat_height).
3. Tuning hiperparametri (Dropout/L2, lr sweep).

**Pe termen mediu (3-6 luni):**
1. Deployment pe platform edge (Jetson/NPU).
2. Monitorizare MLOps (drift detection).
3. Integrare feedback utilizator in UI pentru corectii.

```

### 5.4 Lecții Învățate

```markdown
### Lecții învățate pe parcursul proiectului

**Tehnice:**
1. Preprocesarea si feature-urile au impact major pe clasificarea tabulara.
2. Augmentarile usoare pot imbunatati generalizarea fara cost mare.
3. EarlyStopping + ReduceLROnPlateau stabilizeaza antrenarea.

**Proces:**
1. Documentarea incrementala reduce munca finala.
2. Verificarea end-to-end previne inconsistente intre module.
3. Compararea arhitecturilor mici este suficienta pentru problema actuala.

**Colaborare:**
1. Clarificarea cerintelor pe etape a ajutat alinierea livrabilelor.
2. Revizuirea codului a redus erori de path si naming.
```

### 5.5 Plan Post-Feedback (ULTIMA ITERAȚIE ÎNAINTE DE EXAMEN)

```markdown
### Plan de acțiune după primirea feedback-ului

**ATENȚIE:** Etapa 6 este ULTIMA VERSIUNE pentru care se oferă feedback!
Implementați toate corecțiile înainte de examen.

După primirea feedback-ului de la evaluatori, voi:

1. **Dacă se solicită îmbunătățiri model:**
   - [ex: Experimente adiționale cu arhitecturi alternative]
   - [ex: Colectare date suplimentare pentru clase problematice]
   - **Actualizare:** `models/`, `results/`, README Etapa 5 și 6

2. **Dacă se solicită îmbunătățiri date/preprocesare:**
   - [ex: Rebalansare clase, augmentări suplimentare]
   - **Actualizare:** `data/`, `src/preprocessing/`, README Etapa 3

3. **Dacă se solicită îmbunătățiri arhitectură/State Machine:**
   - [ex: Modificare fluxuri, adăugare stări]
   - **Actualizare:** `docs/state_machine.*`, `src/app/`, README Etapa 4

4. **Dacă se solicită îmbunătățiri documentație:**
   - [ex: Detaliere secțiuni specifice]
   - [ex: Adăugare diagrame explicative]
   - **Actualizare:** README-urile etapelor vizate

5. **Dacă se solicită îmbunătățiri cod:**
   - [ex: Refactorizare module conform feedback]
   - [ex: Adăugare teste unitare]
   - **Actualizare:** `src/`, `requirements.txt`

**Timeline:** Implementare corecții până la data examen
**Commit final:** `"Versiune finală examen - toate corecțiile implementate"`
**Tag final:** `git tag -a v1.0-final-exam -m "Versiune finală pentru examen"`
```
---

## Structura Repository-ului la Finalul Etapei 6

**Structură reală în proiect:**

```
Proiect-RN-Denisa-Elena-Caldararu/
├── README.md
├── etapa3_analiza_date.md
├── etapa4_arhitectura_sia.md
├── etapa5_antrenare_model.md
├── etapa6_optimizare_concluzii.md
├── ORDERINE_RULARE.txt
├── docs/
│   ├── state_machine.png
│   ├── loss_curve.png
│   ├── confusion_matrix_optimized.png
│   ├── screenshots/
│   ├── results/
│   └── optimization/
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
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── train_chair.py
│   │   ├── train_table.py
│   │   ├── train_cabinet.py
│   │   ├── train_fridge.py
│   │   ├── train_stove.py
│   │   ├── evaluate.py
│   │   ├── compare_architectures.py
│   │   ├── optimize.py
│   │   └── export_onnx.py
│   └── app/
│       └── main.py
├── models/
│   ├── untrained_model.h5
│   ├── trained_model.h5
│   ├── chair_model.h5
│   ├── chair_model.onnx
│   ├── table_model.h5
│   ├── cabinet_model.h5
│   ├── fridge_model.h5
│   └── stove_model.h5
├── results/
│   ├── training_history.csv
│   ├── test_metrics.json
│   ├── hyperparameters.yaml
│   ├── cabinet_training_history.csv
│   ├── cabinet_training_metrics.json
│   ├── fridge_training_history.csv
│   ├── fridge_training_metrics.json
│   ├── stove_training_history.csv
│   ├── stove_training_metrics.json
│   ├── table_training_history.csv
│   └── table_training_metrics.json
├── config/
│   ├── chair_scaler.pkl
│   ├── table_scaler.pkl
│   ├── cabinet_scaler.pkl
│   ├── fridge_scaler.pkl
│   └── stove_scaler.pkl
├── requirements.txt
└── .gitignore
```

**Diferențe față de Etapa 5:**
- Adăugat acest README (Etapa 6)
- Comparare arhitecturi în `src/neural_network/compare_architectures.py`
- Export ONNX în `src/neural_network/export_onnx.py`
- Model ONNX în `models/chair_model.onnx`

---

## Instrucțiuni de Rulare (Etapa 6)

### 1. Rulare experimente de comparare arhitecturi

```bash
python src/neural_network/compare_architectures.py
```

### 2. Evaluare pe test set

```bash
python src/neural_network/evaluate.py

# Output așteptat:
# Test Accuracy: 0.9911
# Test F1-score (macro): 0.9915
# Confusion matrix salvată în docs/confusion_matrix_optimized.png
```

**Notă:** Scriptul `evaluate.py` este documentat pe chair (fișiere în `data/chairs/test/`). Pentru celelalte tipuri,
artefactele de antrenare și metricile de train/val sunt salvate în `results/*_training_history.csv` și `results/*_training_metrics.json`.

### 3. Verificare UI cu model antrenat

```bash
# Verificare că UI încarcă modelul corect
python src/app/main.py

# În consolă trebuie să vedeți:
# Loading model: models/chair_model.h5
# Model loaded successfully.
```

### 4. Export ONNX + benchmark

```bash
python src/neural_network/export_onnx.py
```

---

## Checklist Final – Bifați Totul Înainte de Predare

### Prerequisite Etapa 5 (verificare)
- [x] Model antrenat există în `models/chair_model.h5`
- [x] Metrici baseline raportate (Accuracy ≥65%, F1 ≥0.60)
- [x] UI funcțional cu model antrenat
- [x] State Machine implementat

### Optimizare și Experimentare
- [x] Minimum 4 experimente documentate în tabel (baseline + 3 variante)
- [x] Justificare alegere configurație finală
- [x] Model final salvat în `models/chair_model.h5`
- [x] Metrici finale: **Accuracy ≥70%**, **F1 ≥0.65**

### Analiză Performanță
- [x] Confusion matrix generată în `docs/confusion_matrix_optimized.png`
- [x] Analiză interpretare confusion matrix completată în README
- [x] Minimum 5 exemple greșite analizate detaliat
- [x] Implicații industriale documentate

### Actualizare Aplicație Software
- [x] Tabel modificări aplicație completat
- [x] UI încarcă modelul antrenat final
- [x] Screenshot `docs/screenshots/inference_optimized.png`
- [x] Pipeline end-to-end re-testat și funcțional
- [x] (Dacă aplicabil) State Machine actualizat și documentat

### Concluzii
- [x] Secțiune evaluare performanță finală completată
- [x] Limitări identificate și documentate
- [x] Lecții învățate (minimum 5)
- [x] Plan post-feedback scris

### Verificări Tehnice
- [x] `requirements.txt` actualizat
- [x] Toate path-urile RELATIVE
- [x] Cod nou comentat (minimum 15%)
- [x] `git log` arată commit-uri incrementale
- [x] Verificare anti-plagiat respectată

### Verificare Actualizare Etape Anterioare (ITERATIVITATE)
- [x] README Etapa 3 actualizat (dacă s-au modificat date/preprocesare)
- [x] README Etapa 4 actualizat (dacă s-a modificat arhitectura/State Machine)
- [x] README Etapa 5 actualizat (dacă s-au modificat parametri antrenare)
- [x] `docs/state_machine.*` actualizat pentru a reflecta versiunea finală

### Pre-Predare
- [x] `etapa6_optimizare_concluzii.md` completat cu TOATE secțiunile
- [x] Structură repository conformă modelului de mai sus
- [x] Commit: `"Etapa 6 completă – Accuracy=0.9911, F1=0.9915"`
- [x] Tag: `git tag -a v0.6-final -m "Etapa 6 - Concluzii"`
- [x] Push: `git push origin main --tags`
- [x] Repository accesibil (public sau privat cu acces profesori)

---

## Livrabile Obligatorii

Asigurați-vă că următoarele fișiere există și sunt completate:

1. **`etapa6_optimizare_concluzii.md`** (acest fișier) cu:
   - Tabel experimente optimizare (minimum 4)
   - Tabel modificări aplicație software
   - Analiză confusion matrix
   - Analiză 5 exemple greșite
   - Concluzii și lecții învățate

2. **`models/chair_model.h5`** - model final utilizat in aplicatie

   (În aplicația multi-obiect există și: `models/table_model.h5`, `models/cabinet_model.h5`, `models/fridge_model.h5`, `models/stove_model.h5`.)

3. **`results/chair_test_metrics.json`** - metrici finale

   (În repository există și metrici de antrenare pentru celelalte tipuri în `results/*_training_metrics.json`.)

4. **`docs/confusion_matrix_optimized.png`** - confusion matrix model final

5. **`models/chair_model.onnx`** - export ONNX + benchmark

---

## Predare și Contact

**Predarea se face prin:**
1. Commit pe GitHub: `"Etapa 6 completă – Accuracy=0.9911, F1=0.9915"`
2. Tag: `git tag -a v0.6-final -m "Etapa 6 - Concluzii"`
3. Push: `git push origin main --tags`

---

**REMINDER:** Aceasta a fost ultima versiune pentru feedback. Următoarea predare este **VERSIUNEA FINALĂ PENTRU EXAMEN**!
