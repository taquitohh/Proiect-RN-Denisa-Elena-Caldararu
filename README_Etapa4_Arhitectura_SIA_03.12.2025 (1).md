# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Caldararu Denisa  
**Link Repository GitHub**
**Data:** 03.12.2025  
---

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Trebuie să livrați un SCHELET COMPLET și FUNCȚIONAL al întregului Sistem cu Inteligență Artificială (SIA). In acest stadiu modelul RN este doar definit și compilat (fără antrenare serioasă).**

---

## 4.1 Definirea problemei și a SIA-ului

### 1. Definirea problemei (pe românește, clar)

Construim un sistem care generează automat scripturi Python pentru Blender, destinate creării rapide de obiecte 3D de mobilier pentru bucătărie (în faza inițială: scaune). Scopul este prototiparea rapidă, content procedural și asset-uri pentru jocuri, nu realism fizic industrial extrem.

### 2. Ce face efectiv rețeaua neuronală?

Rețeaua neuronală nu generează mesh-uri direct. Ea primește parametri geometrici numerici și clasifică tipul de obiect, returnând tipul de script Blender ce trebuie generat. După inferență, un generator determinist de cod Blender folosește clasa pentru a produce scriptul .py.

**Input RN (features):**
- seat_height
- seat_width
- seat_depth
- leg_count
- leg_thickness
- has_backrest
- backrest_height
- style_variant

**Output RN (clasă):**
| Label | Semnificație |
|------:|--------------|
| 0 | scaun simplu |
| 1 | scaun cu spătar |
| 2 | scaun de bar |
| 3 | taburet |

### 3. Nevoia reală (de ce există sistemul?)

Crearea manuală de obiecte 3D pentru jocuri este lentă și repetitivă, necesitând ajustarea manuală a parametrilor geometrici și a scripturilor Blender pentru fiecare variantă de obiect. Sistemul propus utilizează o rețea neuronală pentru a clasifica automat tipul de obiect de mobilier pe baza parametrilor geometrici, generând automat un script Blender Python corespunzător, reducând semnificativ timpul de creare a asset-urilor.

## Tabel Nevoie Reală → Soluție SIA → Modul Software

| Nevoie reală concretă | Cum o rezolvă SIA-ul | Modul software responsabil |
|-----------------------|----------------------|----------------------------|
| Crearea manuală lentă a obiectelor 3D de mobilier pentru jocuri | Clasifică automat tipul de obiect pe baza parametrilor geometrici și generează script Blender corespunzător în <1 secundă | Rețea Neuronală + Generator Blender |
| Necesitatea prototipării rapide a mai multor variante de mobilier | Permite generarea procedurală a obiectelor prin introducerea parametrilor numerici într-o interfață web | UI Web + RN |
| Consistență între obiectele generate (stil și proporții) | Utilizează reguli deterministe + RN pentru a asigura coerență între obiectele din același set | Data Acquisition + RN |

---

## 4.2 State Machine-ul SIA

### Tipul aplicației

Aplicație de clasificare + generare procedurală la cerere (user-triggered). Utilizatorul introduce parametri, sistemul generează rezultatul și apoi revine la starea inițială.

### State Machine-ul Sistemului

Sistemul propus utilizează o arhitectură de tip State Machine pentru a controla
fluxul aplicației de la introducerea parametrilor de către utilizator până la
generarea scriptului Blender.

Stările principale sunt:

1. **IDLE** – sistemul așteaptă interacțiunea utilizatorului prin interfața web.
2. **INPUT_VALIDATION** – parametrii introduși sunt validați pentru a preveni
  valori invalide sau incompatibile cu regulile de generare.
3. **PREPROCESS** – datele sunt normalizate folosind parametrii de scalare
  calculați în Etapa 3, asigurând consistența cu datele de antrenare.
4. **RN_INFERENCE** – modelul de rețea neuronală efectuează inferența pentru
  clasificarea tipului de obiect de mobilier.
5. **SCRIPT_GENERATION** – pe baza clasei prezise, sistemul generează un script
  Python pentru Blender corespunzător tipului de obiect.
6. **OUTPUT_DISPLAY** – scriptul generat este afișat utilizatorului sau oferit
  pentru descărcare.

Tranzițiile dintre stări sunt declanșate de acțiuni ale utilizatorului sau de
finalizarea cu succes a fiecărei etape din pipeline.

### Flux principal (happy path)

IDLE → INPUT_VALIDATION → PREPROCESS → RN_INFERENCE → SCRIPT_GENERATION → OUTPUT_DISPLAY → IDLE

### Tranziții de eroare (obligatoriu)

| De la | La | Condiție |
|-------|----|----------|
| INPUT_VALIDATION | IDLE | parametri invalizi |
| PREPROCESS | IDLE | eroare încărcare scaler |
| RN_INFERENCE | IDLE | model indisponibil |

### Diagramă State Machine

Diagrama este salvată în `docs/state_machine.png` și include fluxul principal și tranzițiile de eroare către IDLE.

### Justificarea State Machine-ului ales

Aplicația dezvoltată este un sistem de tip *user-triggered*, în care utilizatorul
introduce manual parametri geometrici pentru un obiect de mobilier, iar sistemul
generează automat un script Python pentru Blender. Din acest motiv, arhitectura
aplicației este modelată sub forma unui State Machine, care permite controlul
clar și determinist al fluxului de execuție.

State Machine-ul ales reflectă etapele logice prin care trece o cerere de la
momentul inițial până la livrarea rezultatului final, evitând ambiguitățile și
asigurând o separare clară a responsabilităților fiecărui modul software.

**Descrierea stărilor principale:**

1. **IDLE**  
  Aceasta este starea inițială a sistemului, în care aplicația așteaptă
  interacțiunea utilizatorului prin interfața web. În această stare nu se
  efectuează calcule, iar sistemul este pregătit să primească un nou set de
  parametri.

2. **INPUT_VALIDATION**  
  După trimiterea parametrilor de către utilizator, sistemul verifică validitatea
  acestora (existența tuturor câmpurilor, respectarea intervalelor acceptate,
  coerența logică între parametri). Această stare previne propagarea unor date
  invalide către etapele ulterioare ale pipeline-ului.

3. **PREPROCESS**  
  În această stare, datele de intrare sunt transformate și normalizate folosind
  parametrii de preprocesare calculați în Etapa 3 (de exemplu, StandardScaler).
  Astfel se asigură consistența dintre datele de inferență și cele utilizate la
  antrenarea rețelei neuronale.

4. **RN_INFERENCE**  
  Rețeaua neuronală este utilizată pentru a efectua inferența pe baza datelor
  preprocesate. Modelul clasifică tipul obiectului de mobilier (de exemplu:
  scaun simplu, scaun cu spătar, scaun de bar, taburet). În Etapa 4, modelul este
  definit și compilat, dar nu este încă antrenat cu performanță ridicată.

5. **SCRIPT_GENERATION**  
  Pe baza clasei prezise de rețeaua neuronală, sistemul generează un script Python
  pentru Blender. Scriptul conține instrucțiuni specifice pentru crearea obiectului
  3D corespunzător tipului identificat.

6. **OUTPUT_DISPLAY**  
  În această stare, scriptul Blender generat este afișat utilizatorului sau pus la
  dispoziție pentru descărcare. După livrarea rezultatului, sistemul revine în
  starea IDLE, fiind pregătit pentru o nouă cerere.

**Tranzițiile dintre stări:**

Tranzițiile sunt declanșate fie de acțiuni ale utilizatorului (de exemplu, trimiterea
formularului), fie de finalizarea cu succes a unei etape din pipeline. În cazul în
care apare o eroare (input invalid, indisponibilitatea modelului sau a parametrilor
de preprocesare), sistemul revine în starea IDLE, prevenind blocarea aplicației și
permițând reluarea procesului.

Această arhitectură bazată pe State Machine oferă un flux clar, extensibil și ușor
de întreținut, fiind potrivită pentru dezvoltarea incrementală a aplicației în
etapele următoare ale proiectului.

### IMPORTANT - Ce înseamnă "schelet funcțional":

 **CE TREBUIE SĂ FUNCȚIONEZE:**
- Toate modulele pornesc fără erori
- Pipeline-ul complet rulează end-to-end (de la date → până la output UI)
- Modelul RN este definit și compilat (arhitectura există)
- Web Service/UI primește input și returnează output

 **CE NU E NECESAR ÎN ETAPA 4:**
- Model RN antrenat cu performanță bună
- Hiperparametri optimizați
- Acuratețe mare pe test set
- Web Service/UI cu funcționalități avansate

**Scopul anti-plagiat:** Nu puteți copia un notebook + model pre-antrenat de pe internet, pentru că modelul vostru este NEANTRENAT în această etapă. Demonstrați că înțelegeți arhitectura și că ați construit sistemul de la zero.

---

##  Livrabile Obligatorii

### 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software (max ½ pagină)
Completați in acest readme tabelul următor cu **minimum 2-3 rânduri** care leagă nevoia identificată în Etapa 1-2 cu modulele software pe care le construiți (metrici măsurabile obligatoriu):

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| Ex: Detectarea automată a fisurilor în suduri robotizate | Clasificare imagine radiografică → alertă operator în < 2 secunde | RN + Web Service |
| Ex: Predicția uzurii lagărelor în turbine eoliene | Analiză vibrații în timp real → alertă preventivă cu 95% acuratețe | Data Logging + RN + UI |
| Ex: Optimizarea traiectoriilor robotului mobil în depozit | Predicție timp traversare → reducere 20% consum energetic | RN + Control Module |
| Reducerea timpului de creare a obiectelor 3D parametrice | Clasifică tipul de scaun și generează script Blender corespunzător în câteva secunde | RN + Generator Blender |
| Eliminarea erorilor de input la generarea scaunului | Validează parametrii (intervale și reguli spătar) înainte de inferență | UI Web (Flask) + Preprocess |

**Instrucțiuni:**
- Fiți concreti (nu vagi): "detectare fisuri sudură" ✓, "îmbunătățire proces" ✗
- Specificați metrici măsurabile: "< 2 secunde", "> 95% acuratețe", "reducere 20%"
- Legați fiecare nevoie de modulele software pe care le dezvoltați

---

### 2. Contribuția Voastră Originală la Setul de Date – MINIM 40% din Totalul Observațiilor Finale

**Regula generală:** Din totalul de **N observații finale** în `data/processed/`, **minimum 40%** trebuie să fie **contribuția voastră originală**.

#### Cum se calculează 40%:

**Exemplu 1 - Dataset DOAR public în Etapa 3:**
```
Etapa 3: Ați folosit 10,000 samples dintr-o sursa externa (ex: Kaggle)
Etapa 4: Trebuie să generați/achiziționați date astfel încât:
  
Opțiune A: Adăugați 6,666 samples noi → Total 16,666 (6,666/16,666 = 40%)
Opțiune B: Păstrați 6,000 publice + 4,000 generate → Total 10,000 (4,000/10,000 = 40%)
```

**Exemplu 2 - Dataset parțial original în Etapa 3:**
```
Etapa 3: Ați avut deja 3,000 samples generate + 7,000 publice = 10,000 total
Etapa 4: 3,000 samples existente numără ca "originale"
        Dacă 3,000/10,000 = 30% < 40% → trebuie să generați încă ~1,700 samples
        pentru a ajunge la 4,700/10,000 = 47% > 40% ✓
```

**Exemplu 3 - Dataset complet original:**
```
Etapa 3-4: Generați toate datele (simulare, senzori proprii, etichetare manuală - varianta recomandata)
           → 100% original ✓ (depășește cu mult 40% - FOARTE BINE!)
```

#### Tipuri de contribuții acceptate (exemple din inginerie):

Alegeți UNA sau MAI MULTE dintre variantele de mai jos și **demonstrați clar în repository**:

| **Tip contribuție** | **Exemple concrete din inginerie** | **Dovada minimă cerută** |
|---------------------|-------------------------------------|--------------------------|
| **Date generate prin simulare fizică** | • Traiectorii robot în Gazebo<br>• Vibrații motor cu zgomot aleator calibrat<br>• Consumuri energetice proces industrial simulat | Cod Python/LabVIEW funcțional + grafice comparative (simulat vs real din literatură) + justificare parametri |
| **Date achiziționate cu senzori proprii** | • 500-2000 măsurători accelerometru pe motor<br>• 100-1000 imagini capturate cu cameră montată pe robot<br>• 200-1000 semnale GPS/IMU de pe platformă mobilă<br>• Temperaturi/presiuni procesate din Arduino/ESP32 | Foto setup experimental + CSV-uri produse + descriere protocol achiziție (frecvență, durata, condiții) |
| **Etichetare/adnotare manuală** | • Etichetat manual 1000+ imagini defecte sudură<br>• Anotat 500+ secvențe video cu comportamente robot<br>• Clasificat manual 2000+ semnale vibrații (normal/anomalie)<br>• Marcat manual 1500+ puncte de interes în planuri tehnice | Fișier Excel/JSON cu labels + capturi ecran tool etichetare + log timestamp-uri lucru |
| **Date sintetice prin metode avansate** | • Simulări FEM/CFD pentru date dinamice proces | Cod implementare metodă + exemple before/after + justificare hiperparametri + validare pe subset real |

#### Declarație obligatorie în README:

Scrieți clar în acest README (Secțiunea 2):

```markdown
### Contribuția originală la setul de date:

**Total observații finale:** 15,000
**Observații originale:** 15,000 (100%)

**Tipul contribuției:**
[ ] Date generate prin simulare fizică  
[ ] Date achiziționate cu senzori proprii  
[ ] Etichetare/adnotare manuală  
[X] Date sintetice generate programatic  

**Descriere detaliată:**
Datele au fost generate programatic printr-un script Python care produce
15,000 observații sintetice folosind intervale controlate pentru 8
parametri geometrici ai scaunului. Etichetarea claselor a fost deterministă,
bazată pe reguli logice (de exemplu, înălțimea șezutului și existența spătarului).
Acest mecanism a asigurat consistență, reproductibilitate (seed fix) și o
distribuție coerentă a claselor pentru o problemă de clasificare multi-clasă.

**Locația codului:** `src/data_acquisition/generate_chairs.py`
**Locația datelor:** `data/generated/chairs_dataset.csv`

**Dovezi:**
- Distribuție clase afișată în rularea scriptului (log în consolă)
- Fișier CSV generat în `data/generated/`
```

#### Exemple pentru "contribuție originală":
-Simulări fizice realiste cu ecuații și parametri justificați  
-Date reale achiziționate cu senzori proprii (setup documentat)  
-Augmentări avansate cu justificare fizică (ex: simulare perspective camera industrială)  


#### Atenție - Ce NU este considerat "contribuție originală":

- Augmentări simple (rotații, flips, crop) pe date publice  
- Aplicare filtre standard (Gaussian blur, contrast) pe imagini publice  
- Normalizare/standardizare (aceasta e preprocesare, nu generare)  
- Subset dintr-un dataset public (ex: selectat 40% din ImageNet)


---

### 3. Diagrama State Machine a Întregului Sistem (OBLIGATORIE)

**Cerințe:**
- **Minimum 4-6 stări clare** cu tranziții între ele
- **Formate acceptate:** PNG/SVG, pptx, draw.io 
- **Locație:** `docs/state_machine.*` (orice extensie)
- **Legendă obligatorie:** 1-2 paragrafe în acest README: "De ce ați ales acest State Machine pentru nevoia voastră?"

**Stări tipice pentru un SIA:**
```
IDLE → ACQUIRE_DATA → PREPROCESS → INFERENCE → DISPLAY/ACT → LOG → [ERROR] → STOP
                ↑______________________________________________|
```

**Exemple concrete per domeniu de inginerie:**

#### A. Monitorizare continuă proces industrial (vibrații motor, temperaturi, presiuni):
```
IDLE → START_ACQUISITION → COLLECT_SENSOR_DATA → BUFFER_CHECK → 
PREPROCESS (filtrare, FFT) → RN_INFERENCE → THRESHOLD_CHECK → 
  ├─ [Normal] → LOG_RESULT → UPDATE_DASHBOARD → COLLECT_SENSOR_DATA (loop)
  └─ [Anomalie] → TRIGGER_ALERT → NOTIFY_OPERATOR → LOG_INCIDENT → 
                  COLLECT_SENSOR_DATA (loop)
       ↓ [User stop / Emergency]
     SAFE_SHUTDOWN → STOP
```

#### B. Clasificare imagini defecte producție (suduri, suprafețe, piese):
```
IDLE → WAIT_TRIGGER (senzor trecere piesă) → CAPTURE_IMAGE → 
VALIDATE_IMAGE (blur check, brightness) → 
  ├─ [Valid] → PREPROCESS (resize, normalize) → RN_INFERENCE → 
              CLASSIFY_DEFECT → 
                ├─ [OK] → LOG_OK → CONVEYOR_PASS → IDLE
                └─ [DEFECT] → LOG_DEFECT → TRIGGER_REJECTION → IDLE
  └─ [Invalid] → ERROR_IMAGE_QUALITY → RETRY_CAPTURE (max 3×) → IDLE
       ↓ [Shift end]
     GENERATE_REPORT → STOP
```

#### C. Predicție traiectorii robot mobil (AGV, AMR în depozit):
```
IDLE → LOAD_MAP → RECEIVE_TARGET → PLAN_PATH → 
VALIDATE_PATH (obstacle check) →
  ├─ [Clear] → EXECUTE_SEGMENT → ACQUIRE_SENSORS (LIDAR, IMU) → 
              RN_PREDICT_NEXT_STATE → UPDATE_TRAJECTORY → 
                ├─ [Target reached] → STOP_AT_TARGET → LOG_MISSION → IDLE
                └─ [In progress] → EXECUTE_SEGMENT (loop)
  └─ [Obstacle detected] → REPLAN_PATH → VALIDATE_PATH
       ↓ [Emergency stop / Battery low]
     SAFE_STOP → LOG_STATUS → STOP
```

#### D. Predicție consum energetic (turbine eoliene, procese batch):
```
IDLE → LOAD_HISTORICAL_DATA → ACQUIRE_CURRENT_CONDITIONS 
(vânt, temperatură, demand) → PREPROCESS_FEATURES → 
RN_FORECAST (24h ahead) → VALIDATE_FORECAST (sanity checks) →
  ├─ [Valid] → DISPLAY_FORECAST → UPDATE_CONTROL_STRATEGY → 
              LOG_PREDICTION → WAIT_INTERVAL (1h) → 
              ACQUIRE_CURRENT_CONDITIONS (loop)
  └─ [Invalid] → ERROR_FORECAST → USE_FALLBACK_MODEL → LOG_ERROR → 
                ACQUIRE_CURRENT_CONDITIONS (loop)
       ↓ [User request report]
     GENERATE_DAILY_REPORT → STOP
```

**Notă pentru proiecte simple:**
Chiar dacă aplicația voastră este o clasificare simplă (user upload → classify → display), trebuie să modelați fluxul ca un State Machine. Acest exercițiu vă învață să gândiți modular și să anticipați toate stările posibile (inclusiv erori).

**Legendă obligatorie (scrieți în README):**
```markdown
### Justificarea State Machine-ului ales:

A fost aleasă o arhitectură de tip clasificare la cerere (user-triggered),
deoarece utilizatorul introduce parametri geometrici și solicită generarea
unui script Blender pentru un scaun. Fluxul a fost modelat ca State Machine
pentru a delimita clar validarea inputului, preprocesarea, inferența RN și
generarea scriptului determinist.

Stările principale au fost:
1. **IDLE**: aplicația așteaptă inputul utilizatorului în UI.
2. **USER_INPUT**: parametrii au fost colectați și validați (intervale, reguli spătar).
3. **PREPROCESS**: scalerul salvat a fost aplicat pe datele de intrare.
4. **RN_INFERENCE**: modelul RN a prezis clasa scaunului.
5. **SCRIPT_GENERATION**: scriptul Blender a fost generat determinist.
6. **DISPLAY**: rezultatele au fost afișate utilizatorului.

Tranzițiile critice au fost:
- **USER_INPUT → IDLE**: parametri invalizi (ex: backrest_height > 0 când has_backrest = 0).
- **PREPROCESS → IDLE**: eroare la încărcarea scalerului.
- **RN_INFERENCE → IDLE**: model indisponibil sau eroare la inferență.

Starea de eroare a fost tratată prin revenire la IDLE și afișarea unui mesaj,
pentru a permite reluarea corectă a cererii fără blocarea aplicației.
```

---

### 4. Scheletul Complet al celor 3 Module Cerute la Curs (slide 7)

Toate cele 3 module trebuie să **pornească și să ruleze fără erori** la predare. Nu trebuie să fie perfecte, dar trebuie să demonstreze că înțelegeți arhitectura.

| **Modul** | **Python (exemple tehnologii)** | **LabVIEW** | **Cerință minimă funcțională (la predare)** |
|-----------|----------------------------------|-------------|----------------------------------------------|
| **1. Data Logging / Acquisition** | `src/data_acquisition/` | LLB cu VI-uri de generare/achiziție | **MUST:** Produce CSV cu datele voastre (inclusiv cele 40% originale). Cod rulează fără erori și generează minimum 100 samples demonstrative. |
| **2. Neural Network Module** | `src/neural_network/model.py` sau folder dedicat | LLB cu VI-uri RN | **MUST:** Modelul RN definit, compilat, poate fi încărcat. **NOT required:** Model antrenat cu performanță bună (poate avea weights random/inițializați). |
| **3. Web Service / UI** | Flask | WebVI sau Web Publishing Tool | **MUST:** Primește input de la user și afișează un output. **NOT required:** UI frumos, funcționalități avansate. |

#### Detalii per modul:

#### **Modul 1: Data Logging / Acquisition**

**Funcționalități obligatorii:**
- [x] Cod rulează fără erori: `python src/data_acquisition/generate.py` sau echivalent LabVIEW
- [x] Generează CSV în format compatibil cu preprocesarea din Etapa 3
- [x] Include minimum 40% date originale în dataset-ul final
- [x] Documentație în cod: ce date generează, cu ce parametri

#### **Modul 2: Neural Network Module**

**Funcționalități obligatorii:**
- [x] Arhitectură RN definită și compilată fără erori
- [x] Model poate fi salvat și reîncărcat
- [x] Include justificare pentru arhitectura aleasă (în docstring sau README)
- [x] **NU trebuie antrenat** cu performanță bună (weights pot fi random)


#### **Modul 3: Web Service / UI**

**Funcționalități MINIME obligatorii:**
- [x] Propunere Interfață ce primește input de la user (formular, file upload, sau API endpoint)
- [x] Includeți un screenshot demonstrativ în `docs/screenshots/`

**Ce NU e necesar în Etapa 4:**
- UI frumos/profesionist cu grafică avansată
- Funcționalități multiple (istorice, comparații, statistici)
- Predicții corecte (modelul e neantrenat, e normal să fie incorect)
- Deployment în cloud sau server de producție

**Scop:** Prima demonstrație că pipeline-ul end-to-end funcționează: input user → preprocess → model → output.

---

## 4.3 Modulul Rețea Neuronală (RN)

Modulul de Rețea Neuronală are rolul de a clasifica tipul obiectului de mobilier
pe baza parametrilor geometrici introduși de utilizator. Problema este formulată
ca o clasificare multi-clasă, cu patru clase posibile.

**Date de intrare:**
Modelul primește un vector de intrare format din 8 caracteristici numerice,
corespunzătoare parametrilor geometrici ai obiectului (înălțime șezut, lățime,
adâncime, număr picioare, grosime picioare, existența spătarului, înălțimea
spătarului, variantă de stil).

**Date de ieșire:**
Ieșirea modelului este un vector de probabilități pentru cele 4 clase posibile
de obiecte de mobilier. Clasa cu probabilitatea maximă este selectată ca rezultat
al inferenței.

**Arhitectura modelului:**
Rețeaua neuronală este de tip feedforward (MLP) și are următoarea structură:
- strat de intrare cu 8 neuroni
- două straturi ascunse cu funcții de activare ReLU
- strat de ieșire cu 4 neuroni și activare Softmax

În Etapa 4, modelul este doar definit și compilat, fără a fi antrenat cu
performanță ridicată, conform cerințelor proiectului.

---

## 4.4 Modulul UI / Web Service

Modulul de interfață utilizator reprezintă punctul de interacțiune dintre
utilizator și Sistemul cu Inteligență Artificială. Acesta permite introducerea
manuală a parametrilor geometrici ai obiectului de mobilier și afișează
rezultatul clasificării realizate de rețeaua neuronală.

Interfața este implementată sub forma unei aplicații web folosind Flask,
asigurând un flux end-to-end complet:
input utilizator → preprocesare → inferență RN → afișare rezultat.

În Etapa 4, interfața este funcțională, dar utilizează un model neantrenat,
având rol demonstrativ pentru validarea arhitecturii și a fluxurilor de date.


## Structura Repository-ului la Finalul Etapei 4 (OBLIGATORIE)

**Verificare consistență cu Etapa 3:**

```
proiect-rn-[nume-prenume]/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── generated/  # Date originale
│   ├── train/
│   ├── validation/
│   └── test/
├── src/
│   ├── data_acquisition/
│   ├── preprocessing/  # Din Etapa 3
│   ├── neural_network/
│   └── app/  # UI schelet
├── docs/
│   ├── state_machine.*           #(state_machine.png sau state_machine.pptx sau state_machine.drawio)
│   └── [alte dovezi]
├── models/  # Untrained model
├── config/
├── README.md
├── README_Etapa3.md              # (deja existent)
├── README_Etapa4_Arhitectura_SIA.md              # ← acest fișier completat (în rădăcină)
└── requirements.txt  # Sau .lvproj
```

**Diferențe față de Etapa 3:**
- Adăugat `data/generated/` pentru contribuția dvs originală
- Adăugat `src/data_acquisition/` - MODUL 1
- Adăugat `src/neural_network/` - MODUL 2
- Adăugat `src/app/` - MODUL 3
- Adăugat `models/` pentru model neantrenat
- Adăugat `docs/state_machine.png` - OBLIGATORIU
- Adăugat `docs/screenshots/` pentru demonstrație UI

---

## Checklist Final – Bifați Totul Înainte de Predare

### Documentație și Structură
- [x] Tabelul Nevoie → Soluție → Modul complet (minimum 2 rânduri cu exemple concrete completate in README_Etapa4_Arhitectura_SIA.md)
- [x] Declarație contribuție 40% date originale completată în README_Etapa4_Arhitectura_SIA.md
- [x] Cod generare/achiziție date funcțional și documentat
- [x] Dovezi contribuție originală: grafice + log + statistici în `docs/`
- [x] Diagrama State Machine creată și salvată în `docs/state_machine.*`
- [x] Legendă State Machine scrisă în README_Etapa4_Arhitectura_SIA.md (minimum 1-2 paragrafe cu justificare)
- [x] Repository structurat conform modelului de mai sus (verificat consistență cu Etapa 3)

### Modul 1: Data Logging / Acquisition
- [x] Cod rulează fără erori (`python src/data_acquisition/...` sau echivalent LabVIEW)
- [x] Produce minimum 40% date originale din dataset-ul final
- [x] CSV generat în format compatibil cu preprocesarea din Etapa 3
- [x] Documentație în `src/data_acquisition/README.md` cu:
  - [x] Metodă de generare/achiziție explicată
  - [x] Parametri folosiți (frecvență, durată, zgomot, etc.)
  - [x] Justificare relevanță date pentru problema voastră
- [x] Fișiere în `data/generated/` conform structurii

### Modul 2: Neural Network
- [x] Arhitectură RN definită și documentată în cod (docstring detaliat) - versiunea inițială 
- [x] README în `src/neural_network/` cu detalii arhitectură curentă

### Modul 3: Web Service / UI
- [x] Propunere Interfață ce pornește fără erori (comanda de lansare testată)
- [x] Screenshot demonstrativ în `docs/screenshots/ui_demo.png`
- [x] README în `src/app/` cu instrucțiuni lansare (comenzi exacte)

---

**Predarea se face prin commit pe GitHub cu mesajul:**  
`"Etapa 4 completă - Arhitectură SIA funcțională"`

**Tag obligatoriu:**  
`git tag -a v0.4-architecture -m "Etapa 4 - Skeleton complet SIA"`


