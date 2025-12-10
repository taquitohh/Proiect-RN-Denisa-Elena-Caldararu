# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Căldăraru Denisa Elena  
**Link Repository GitHub:** [https://github.com/taquitohh/Proiect_RN](https://github.com/taquitohh/Proiect_RN)  
**Data:** 9 Decembrie 2025  
---

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Trebuie să livrați un SCHELET COMPLET și FUNCȚIONAL al întregului Sistem cu Inteligență Artificială (SIA). In acest stadiu modelul RN este doar definit și compilat (fără antrenare serioasă).**

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
| Generarea automată de cod Blender Python din descrieri text în limba română | Clasificare intenție + extragere parametri → generare script în < 1 secundă | TextPreprocessor + RN + BlenderGenerator |
| Reducerea timpului de prototipare 3D pentru designeri fără experiență de programare | Interfață chat conversațională → cod Python gata de rulat în Blender | Web UI (React) + Flask API |
| Standardizarea pieselor 3D și eliminarea erorilor umane de cod | Template-uri validate + parametri extrași automat → cod corect sintactic 100% | BlenderScriptGenerator + Validare |

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

**Total observații finale:** [N] (după Etapa 3 + Etapa 4)
**Observații originale:** [M] ([X]%)

**Tipul contribuției:**
[X] Date generate prin simulare fizică  
[ ] Date achiziționate cu senzori proprii  
[ ] Etichetare/adnotare manuală  
[ ] Date sintetice prin metode avansate  

**Descriere detaliată:**
[Explicați în 2-3 paragrafe cum ați generat datele, ce metode ați folosit, 
de ce sunt relevante pentru problema voastră, cu ce parametri ați rulat simularea/achiziția]

**Locația codului:** `src/data_acquisition/[numele_scriptului]`
**Locația datelor:** `data/generated/` sau `data/raw/original/`

**Dovezi:**
- Grafic comparativ: `docs/generated_vs_real.png`
- Setup experimental: `docs/acquisition_setup.jpg` (dacă aplicabil)
- Tabel statistici: `docs/data_statistics.csv`
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

### ✅ Contribuția originală la setul de date (COMPLETAT):

**Total observații finale:** 200+ perechi text-intenție + 40 scripturi Blender
**Observații originale:** 200+ (100%)

**Tipul contribuției:**
- [ ] Date generate prin simulare fizică  
- [ ] Date achiziționate cu senzori proprii  
- [X] Etichetare/adnotare manuală  
- [ ] Date sintetice prin metode avansate  

**Descriere detaliată:**

Toate datele din acest proiect sunt **100% originale**, create manual pentru domeniul specific Text-to-Blender:

1. **Scripturi Blender Python** (40+ fișiere în `data/raw/blender_scripts/`):
   - Fiecare script reprezintă o operație Blender specifică
   - Exemple: `create_cube_basic.py`, `apply_material_red.py`, `add_modifier_bevel.py`
   - Codul este scris manual, testat și validat în Blender 3.x/4.x

2. **Dataset de antrenare** (`data/raw/blender_training_data.json`):
   - Perechi text natural (română) → intenție clasificabilă
   - Variații de formulare pentru fiecare intenție (5-10 per categorie)
   - Exemple: "creează un cub roșu" → `create_cube` + `apply_material_red`

3. **Template-uri parametrizate** (`src/generators/blender_generator.py`):
   - Fiecare template permite extragerea automată de parametri
   - Parametri: dimensiuni, poziții, culori, proprietăți materiale

**Locația codului:** `src/data_acquisition/data_loader.py`, `src/generators/blender_generator.py`
**Locația datelor:** `data/raw/blender_scripts/`, `data/raw/blender_training_data.json`

**Dovezi:**
- Scripturi Python: `data/raw/blender_scripts/*.py` (40+ fișiere)
- Dataset JSON: `data/raw/blender_training_data.json`
- Generator cu template-uri: `src/generators/blender_generator.py`

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

---

### ✅ Justificarea State Machine-ului ales (COMPLETAT):

**Diagrama:** `docs/state_machine.png`

Am ales arhitectura **clasificare la cerere (on-demand inference)** pentru că proiectul Text-to-Blender procesează comenzi text individuale de la utilizator și returnează cod Python instantaneu.

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      STATE MACHINE - Text to Blender                          ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║   ┌────────────┐    user input    ┌────────────┐    text ok    ┌──────────┐  ║
║   │            │─────────────────►│  RECEIVE   │──────────────►│ VALIDATE │  ║
║   │    IDLE    │                  │   INPUT    │               │  INPUT   │  ║
║   │            │◄─────────────────┤            │               │          │  ║
║   └────────────┘   new request    └────────────┘               └──────────┘  ║
║         ▲                                                           │        ║
║         │                                              ┌────────────┴───┐    ║
║         │                                              │                │    ║
║         │                                          valid            invalid  ║
║         │                                              │                │    ║
║         │                                              ▼                │    ║
║         │                                      ┌──────────────┐         │    ║
║         │                                      │  CLASSIFY    │         │    ║
║         │                                      │   INTENT     │         │    ║
║         │                                      │   (RN)       │         │    ║
║         │                                      └──────────────┘         │    ║
║         │                                              │                │    ║
║         │                                         classified            │    ║
║         │                                              │                │    ║
║         │                                              ▼                │    ║
║         │                                      ┌──────────────┐         │    ║
║         │                                      │   EXTRACT    │         │    ║
║         │                                      │   PARAMS     │         │    ║
║         │                                      │  (Regex/NLP) │         │    ║
║         │                                      └──────────────┘         │    ║
║         │                                              │                │    ║
║         │                                          extracted            │    ║
║         │                                              │                │    ║
║         │                                              ▼                │    ║
║         │                                      ┌──────────────┐         │    ║
║         │                                      │  GENERATE    │         │    ║
║         │                                      │    CODE      │         │    ║
║         │                                      │ (Templates)  │         │    ║
║         │                                      └──────────────┘         │    ║
║         │                                              │                │    ║
║         │                                    ┌─────────┴─────────┐      │    ║
║         │                                    │                   │      │    ║
║         │                                 success              error    │    ║
║         │                                    │                   │      │    ║
║         │                                    ▼                   ▼      │    ║
║         │                            ┌──────────────┐    ┌──────────────┐    ║
║         │          done              │   DISPLAY    │    │    ERROR     │◄───┘
║         └────────────────────────────│   OUTPUT     │    │   HANDLER    │    ║
║                                      │              │    │              │    ║
║                                      └──────────────┘    └──────────────┘    ║
║                                                                 │            ║
║                                                            retry/reset       ║
║                                                                 │            ║
║                                      ┌──────────────────────────┘            ║
║                                      │                                       ║
║                                      ▼                                       ║
║                               (revine la IDLE)                               ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

**Stările principale sunt:**
1. **IDLE**: Sistemul așteaptă input de la utilizator (chat UI gol)
2. **RECEIVE_INPUT**: Primește textul introdus de user în chat
3. **VALIDATE_INPUT**: Verifică dacă textul este valid (non-empty, lungime acceptabilă)
4. **CLASSIFY_INTENT**: Rețeaua neuronală clasifică intenția (create_cube, apply_material, etc.)
5. **EXTRACT_PARAMS**: Extrage parametrii numerici din text (size=2, color=red)
6. **GENERATE_CODE**: Generează script Python folosind template-uri
7. **DISPLAY_OUTPUT**: Afișează codul generat în UI cu syntax highlighting
8. **ERROR_HANDLER**: Gestionează erori și oferă mesaje informative

**Tranzițiile critice sunt:**
- **IDLE → RECEIVE_INPUT**: când utilizatorul apasă Enter sau butonul Send
- **VALIDATE_INPUT → ERROR**: când textul este gol sau prea lung (>500 caractere)
- **CLASSIFY_INTENT → EXTRACT_PARAMS**: când modelul returnează o intenție validă
- **GENERATE_CODE → ERROR**: când nu există template pentru intenția detectată

**Starea ERROR este esențială pentru că:**
- Utilizatorul poate introduce text incomprehensibil
- Backend-ul poate fi temporar indisponibil (offline mode cu fallback local)
- Template-ul pentru o anumită intenție poate lipsi
- Sistemul trebuie să ofere feedback clar și să revină la IDLE

**Bucla de feedback funcționează astfel:**
- După DISPLAY_OUTPUT, utilizatorul poate rafina cererea bazându-se pe codul generat
- Feedback-ul vizual (cod afișat) ajută utilizatorul să-și ajusteze formularea
- Sistemul revine la IDLE pentru a primi următoarea comandă

---

### 4. Scheletul Complet al celor 3 Module Cerute la Curs (slide 7)

Toate cele 3 module trebuie să **pornească și să ruleze fără erori** la predare. Nu trebuie să fie perfecte, dar trebuie să demonstreze că înțelegeți arhitectura.

| **Modul** | **Python (exemple tehnologii)** | **LabVIEW** | **Cerință minimă funcțională (la predare)** |
|-----------|----------------------------------|-------------|----------------------------------------------|
| **1. Data Logging / Acquisition** | `src/data_acquisition/` | LLB cu VI-uri de generare/achiziție | **MUST:** Produce CSV cu datele voastre (inclusiv cele 40% originale). Cod rulează fără erori și generează minimum 100 samples demonstrative. |
| **2. Neural Network Module** | `src/neural_network/model.py` sau folder dedicat | LLB cu VI-uri RN | **MUST:** Modelul RN definit, compilat, poate fi încărcat. **NOT required:** Model antrenat cu performanță bună (poate avea weights random/inițializați). |
| **3. Web Service / UI** | Streamlit, Gradio, FastAPI, Flask, Dash | WebVI sau Web Publishing Tool | **MUST:** Primește input de la user și afișează un output. **NOT required:** UI frumos, funcționalități avansate. |

#### Detalii per modul:

#### **Modul 1: Data Logging / Acquisition**

**Funcționalități obligatorii:**
- [ ] Cod rulează fără erori: `python src/data_acquisition/generate.py` sau echivalent LabVIEW
- [ ] Generează CSV în format compatibil cu preprocesarea din Etapa 3
- [ ] Include minimum 40% date originale în dataset-ul final
- [ ] Documentație în cod: ce date generează, cu ce parametri

#### **Modul 2: Neural Network Module**

**Funcționalități obligatorii:**
- [ ] Arhitectură RN definită și compilată fără erori
- [ ] Model poate fi salvat și reîncărcat
- [ ] Include justificare pentru arhitectura aleasă (în docstring sau README)
- [ ] **NU trebuie antrenat** cu performanță bună (weights pot fi random)


#### **Modul 3: Web Service / UI**

**Funcționalități MINIME obligatorii:**
- [ ] Propunere Interfață ce primește input de la user (formular, file upload, sau API endpoint)
- [ ] Includeți un screenshot demonstrativ în `docs/screenshots/`

**Ce NU e necesar în Etapa 4:**
- UI frumos/profesionist cu grafică avansată
- Funcționalități multiple (istorice, comparații, statistici)
- Predicții corecte (modelul e neantrenat, e normal să fie incorect)
- Deployment în cloud sau server de producție

**Scop:** Prima demonstrație că pipeline-ul end-to-end funcționează: input user → preprocess → model → output.


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


