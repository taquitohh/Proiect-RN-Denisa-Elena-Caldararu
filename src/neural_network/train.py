"""Entry-point de antrenare (Etapa 5/6).

Repository-ul contine scripturi de antrenare per tip de obiect (ex: `train_chair.py`).
Acest fisier exista ca entrypoint stabil (`train.py`) pentru structura de predare,
fara a schimba implementarea existenta.

Implicit ruleaza antrenarea pentru `chair`.
"""

# Nota:
# - Foloseste path-uri relative la repo definite in acest modul.
# - Este destinat folosirii in pipeline-ul proiectului.
# - Genereaza artefacte in folderele proiectului cand este cazul.
# - Presupune schema de intrare din data/README.md (cand este cazul).
# - Determinismul este aplicat cand exista un seed definit.
# - Pastreaza output-ul in consola minim pentru claritate.


from __future__ import annotations

from src.neural_network.train_chair import train


if __name__ == "__main__":
    train()
