# Neural Network

## Scop
Modulul RN realizeaza clasificarea tipului de obiect pe baza parametrilor geometrici.
In Etapa 4, arhitectura este definita si compilata, fara antrenare serioasa.

## Arhitectura curenta
- Tip: MLP (feed-forward) pentru date tabulare.
- Input: vector de caracteristici scalate (ex: 8 pentru chair).
- Output: probabilitati pe clase (softmax).

## Flux
1. Incarcare model si scaler.
2. Normalizare/scalare input.
3. Inferenta RN.
4. Returnare clasa prezisa + probabilitati.

## Fisiere relevante
- `model.py`: definirea si compilarea modelului.
- `train_chair.py` / `train_*.py`: antrenare specifica per obiect.
- `evaluate.py`: evaluare model.

## Note
- In Etapa 4 nu este necesara performanta ridicata; modelul poate rula cu weights initializate.
- Detaliile complete de antrenare sunt tratate in Etapa 5.
