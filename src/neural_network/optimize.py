"""Script de optimizare / tuning (Etapa 6).

In acest proiect, compararea configuratiilor (arhitecturi MLP) este implementata
in `compare_architectures.py`. Acest fisier exista ca entrypoint stabil pentru
structura de predare (optimize.py), fara a schimba logica existenta.
"""

# Nota:
# - Foloseste path-uri relative la repo definite in acest modul.
# - Este destinat folosirii in pipeline-ul proiectului.
# - Genereaza artefacte in folderele proiectului cand este cazul.
# - Presupune schema de intrare din data/README.md (cand este cazul).
# - Determinismul este aplicat cand exista un seed definit.
# - Pastreaza output-ul in consola minim pentru claritate.


from __future__ import annotations

from src.neural_network.compare_architectures import main


if __name__ == "__main__":
    main()
