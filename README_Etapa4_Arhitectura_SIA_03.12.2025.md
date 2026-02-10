# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Caldararu Denisa Elena  
**Link Repository GitHub:** https://github.com/taquitohh/Proiect_RN  
**Data:** 03.12.2025  
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
| Identificarea rapida a tipului de obiect (chair) din parametri geometrici | RN returneaza clasa + probabilitati in < 1s pentru inputul utilizatorului | RN + Web Service (Flask UI) |
| Generarea automata a modelului 3D procedural pentru obiectul prezis | Script Blender generat in < 2s, pregatit pentru randare | Blender scripts + Web Service |
| Validarea functionarii pipeline-ului end-to-end (RN + Blender) | UI afiseaza status RN/Blender si feedback imediat la cereri | UI + Blender API |

**Instrucțiuni:**
- Fiți concreti (nu vagi): "detectare fisuri sudură" ✓, "îmbunătățire proces" ✗
- Specificați metrici măsurabile: "< 2 secunde", "> 95% acuratețe", "reducere 20%"
- Legați fiecare nevoie de modulele software pe care le dezvoltați

---

### 2. Contribuția Voastră Originală la Setul de Date – MINIM 40% din Totalul Observațiilor Finale

**Regula generală:** Din totalul de **N observații finale** în `data/processed/`, **minimum 40%** trebuie să fie **contribuția voastră originală**.


#### Declarație obligatorie în README:

Scrieți clar în acest README (Secțiunea 2):

### Contribuția originală la setul de date:

**Total observații finale (chair):** 15,000
**Observații originale:** 15,000 (100%)

**Tipul contribuției:** 
[ ] Date generate prin simulare fizică  
[ ] Date achiziționate cu senzori proprii  
[ ] Etichetare/adnotare manuală  
[X] Date sintetice prin metode avansate  

**Descriere detaliată:**
Datele au fost generate programatic, folosind intervale realiste pentru dimensiuni
si reguli deterministe pentru corectitudinea combinatiilor (ex: spatar doar cand has_backrest = 1).
Generatorul foloseste seed fix pentru reproductibilitate, iar etichetele sunt atribuite logic
in functie de parametrii geometrici, fara interventie manuala.

**Locația codului:** `src/data_acquisition/generate_chairs.py`
**Locația datelor:** `data/generated/chairs_dataset.csv`

**Dovezi:**
- Dataset generat: `data/generated/chairs_dataset.csv`
- Script generare: `src/data_acquisition/generate_chairs.py`
- Descriere centralizata: `data/README.md`

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
### Justificarea State Machine-ului ales:

Am ales o arhitectura de tip clasificare interactiva (input user → inferenta → output),
deoarece proiectul nostru porneste de la parametri geometrici introdusi manual si trebuie sa
returneze rapid tipul de mobilier si scriptul Blender.

Starile principale sunt:
1. IDLE: sistemul asteapta inputul utilizatorului.
2. PREPROCESS: validarea si scalarea caracteristicilor.
3. RN_INFERENCE: inferenta RN si calculul probabilitatilor.
4. GENERATE_SCRIPT: generarea scriptului Blender in functie de clasa prezisa.
5. RENDER_PREVIEW: cerere catre Blender API pentru randare.
6. DISPLAY_RESULT: afisarea rezultatelor in UI.

Schema (simplificata):
```
[IDLE]
  ↓ (input valid)
[PREPROCESS]
  ↓
[RN_INFERENCE]
  ↓
[GENERATE_SCRIPT]
  ↓
[RENDER_PREVIEW]
   ├─(OK)→ [DISPLAY_RESULT] → [IDLE]
   └─(ERROR)→ [ERROR] → [DISPLAY_RESULT] → [IDLE]
```

Tranzitii critice:
- PREPROCESS → RN_INFERENCE: cand inputul este validat.
- RENDER_PREVIEW → ERROR: daca Blender API nu raspunde.

Starea ERROR este necesara pentru a gestiona lipsa Blender API si pentru a afisa un mesaj
clar utilizatorului, fara a intrerupe restul pipeline-ului.

Tranzițiile critice sunt:
- [STARE_A] → [STARE_B]: [când se întâmplă - ex: "când buffer-ul atinge 1024 samples"]
- [STARE_X] → [ERROR]: [condiții - ex: "când senzorul nu răspunde > 100ms"]

Starea ERROR este esențială pentru că [explicați ce erori pot apărea în contextul 
aplicației voastre industriale - ex: "senzorul se poate deconecta în mediul industrial 
cu vibrații și temperatură variabilă, trebuie să gestionăm reconnect automat"].

Bucla de feedback [dacă există] funcționează astfel: [ex: "rezultatul inferenței 
actualizează parametrii controlerului PID pentru reglarea vitezei motorului"].
```

---

### 4. Scheletul Complet al celor 3 Module Cerute la Curs (slide 7)

Toate cele 3 module trebuie să **pornească și să ruleze fără erori** la predare. Nu trebuie să fie perfecte, dar trebuie să demonstreze că înțelegeți arhitectura.

| **Modul** | **Python (exemple tehnologii)** | **LabVIEW** | **Cerință minimă funcțională (la predare)** |
|-----------|----------------------------------|-------------|----------------------------------------------|
| **1. Data Logging / Acquisition** | `src/data_acquisition/generate_chairs.py` (plus extensii pentru alte obiecte) | LLB cu VI-uri de generare/achiziție | **MUST:** Produce CSV cu datele voastre (inclusiv cele 40% originale). Cod rulează fără erori și generează minimum 100 samples demonstrative. |
| **2. Neural Network Module** | `src/neural_network/model.py` | LLB cu VI-uri RN | **MUST:** Modelul RN definit, compilat, poate fi încărcat. **NOT required:** Model antrenat cu performanță bună (poate avea weights random/inițializați). |
| **3. Web Service / UI** | Flask UI in `src/app/main.py` | WebVI sau Web Publishing Tool | **MUST:** Primește input de la user și afișează un output. **NOT required:** UI frumos, funcționalități avansate. |

#### Detalii per modul:

#### **Modul 1: Data Logging / Acquisition**

- Adăugat `src/app/` - MODUL 3
- Adăugat `models/` pentru model neantrenat
- Adăugat `docs/state_machine.png` - OBLIGATORIU
- Adăugat `docs/screenshots/` pentru demonstrație UI

---

## Checklist Final – Bifați Totul Înainte de Predare

### Documentație și Structură
- [x] Tabelul Nevoie → Soluție → Modul complet (minimum 2 rânduri cu exemple concrete)
- [x] Declarație contribuție 40% date originale completată
- [x] Cod generare/achiziție date funcțional și documentat
- [x] Dovezi contribuție originală in `data/generated/` si `data/README.md`
- [x] Diagrama State Machine creată și salvată în `docs/state_machine.png`
- [x] Legendă State Machine scrisă în acest README
- [x] Repository structurat conform modelului de mai sus

### Modul 1: Data Logging / Acquisition
- [x] Cod rulează fără erori (`python src/data_acquisition/generate_chairs.py`)
- [x] Produce minimum 40% date originale din dataset-ul final (100% original)
- [x] CSV generat în format compatibil cu preprocesarea din Etapa 3
- [x] Documentație în `src/data_acquisition/README.md` cu:
  - [x] Metodă de generare/achiziție explicată
  - [x] Parametri folosiți (intervale, reguli, seed)
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


