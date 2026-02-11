# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Caldararu Denisa Elena  
**Data:** 10.10.2025  

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care se analizează și se preprocesează setul de date necesar proiectului „Rețele Neuronale". Scopul etapei este pregătirea corectă a datelor pentru instruirea modelului RN, respectând bunele practici privind calitatea, consistența și reproductibilitatea datelor.

In aceasta etapa lucram doar pe varianta **chair**, o varianta de element pentru generare , relativ simpla; (table, cabinet, fridge, etc) sunt ca plan de viitor, daca se reuseste cu **chair**.

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```
project-name/
├── README.md
├── docs/
│   └── datasets/          # descriere seturi de date, surse, diagrame
├── data/
│   ├── raw/               # date brute
│   ├── processed/         # date curățate și transformate
│   ├── train/             # set de instruire
│   ├── validation/        # set de validare
│   └── test/              # set de testare
├── src/
│   ├── preprocessing/     # funcții pentru preprocesare
│   ├── data_acquisition/  # generare / achiziție date (dacă există)
│   └── neural_network/    # implementarea RN (în etapa următoare)
├── config/                # fișiere de configurare
└── requirements.txt       # dependențe Python (dacă aplicabil)
```

---

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** date sintetice generate programatic (script Python)
* **Modul de achiziție:** ☑ Generare programatică
* **Perioada / condițiile colectării:** n/a – datele au fost generate local, determinist (seed fix)

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** 15,000
* **Număr de caracteristici (features):** 8
* **Tipuri de date:** ☑ Numerice (toate caracteristicile sunt numerice)
* **Format fișiere:** ☑ CSV

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
|-------------------|---------|-------------|---------------|--------------------|
| seat_height | numeric | m | înălțimea șezutului | 0.40–0.80 |
| seat_width | numeric | m | lățimea șezutului | 0.35–0.60 |
| seat_depth | numeric | m | adâncimea șezutului | 0.35–0.60 |
| leg_count | numeric (int) | – | număr picioare | {3, 4, 5} |
| leg_thickness | numeric | m | grosimea picioarelor | 0.03–0.08 |
| has_backrest | numeric (int) | – | existența spătarului | {0, 1} |
| backrest_height | numeric | m | înălțimea spătarului | 0.00 sau 0.20–0.50 |
| style_variant | numeric (int) | – | variantă stil | {0, 1, 2} |

**Fișier recomandat:** descrierea a fost centralizată în `data/README.md`.

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

* **Medie, mediană, deviație standard** pentru a confirma ca valorile sunt in plajele asteptate.
* **Min–max** pentru fiecare caracteristica, ca verificare rapida a limitelor.
* **Distribuții pe caracteristici** pentru a observa daca apar concentrari sau valori in afara intervalelor.

Exemple de verificare:
* seat_height trebuie sa ramana in 0.40–0.80 (min/max confirmate).
* leg_count trebuie sa fie doar 3, 4 sau 5 (fara alte valori).
* backrest_height are 0.00 cand has_backrest = 0 si 0.20–0.50 cand has_backrest = 1.

### 3.2 Analiza calitatii datelor

* **Detectarea valorilor lipsa:** nu au aparut valori lipsa in setul generat.
* **Detectarea valorilor inconsistente:** s-a verificat logica intre atribute (ex: `backrest_height > 0` cand `has_backrest = 0`).

### 3.3 Probleme identificate

* Nu s-au observat valori lipsa in verificarile de baza; setul este sintetic si generat determinist.
* Nu s-au observat inconsistente raportate de regulile definite, in limitele verificarilor aplicate.

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* **Eliminare duplicatelor:** nu a fost necesară (date generate programatic)
* **Tratarea valorilor lipsă:** nu s-au găsit valori lipsă
* **Tratarea outlierilor:** nu s-a aplicat (intervale controlate la generare)

### 4.2 Transformarea caracteristicilor

* **Standardizare:** StandardScaler pe toate cele 8 caracteristici
* **Encoding:** nu a fost necesar (nu există variabile categoriale non-numerice)
* **Ajustarea dezechilibrului de clasă:** nu a fost aplicată (distribuția este controlată la generare)

### 4.3 Structurarea seturilor de date

**Împărțire recomandată:**
* 70–80% – train
* 10–15% – validation
* 10–15% – test

**Împărțire folosita (chair, etapa curenta):**
* Train: 10,500 (70%)
* Validation: 2,250 (15%)
* Test: 2,250 (15%)
* Total: 15,000 observatii

**Principii respectate:**
* Stratificare pentru clasificare
* Scalarea a fost aplicată înainte de split pentru a păstra un singur scaler determinist reutilizat în inferență

### 4.4 Salvarea rezultatelor preprocesării

* Date preprocesate în `data/processed/`
* Seturi train/val/test în foldere dedicate
* Parametrii de preprocesare în `config/chair_scaler.pkl`

---

##  5. Fișiere Generate în Această Etapă

* `data/raw/` – date brute
* `data/processed/` – date curățate & transformate
* `data/chairs/train/`, `data/chairs/validation/`, `data/chairs/test/` – seturi finale
* `src/preprocessing/` – codul de preprocesare
* `data/README.md` – descrierea dataset-ului

---

## Addendum (extindere proiect)

Ulterior Etapei 3, proiectul a fost extins cu obiecte noi (table, cabinet,
fridge si stove), fiecare cu dataset separat, reguli deterministe si pipeline
de preprocesare independent. Documentatia comuna a dataset-urilor este
centralizata in:

- `data/README.md`

---

##  6. Stare Etapă (de completat de student)

- [x] Structură repository configurată
- [x] Dataset analizat (EDA realizată)
- [x] Date preprocesate
- [x] Seturi train/val/test generate
- [x] Documentație actualizată în README + `data/README.md`

---
