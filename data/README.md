# Prezentare dataset

Acest proiect foloseste dataset-uri sintetice, deterministe, pentru mai multe tipuri de obiecte.
Toate dataset-urile sunt generate programatic cu seed fix pentru reproductibilitate.

## Dataset scaune (chairs_dataset.csv)

- Locatie: data/generated/chairs_dataset.csv
- Esantioane: 15000
- Caracteristici (8):
  - seat_height, seat_width, seat_depth
  - leg_count, leg_thickness
  - has_backrest, backrest_height
  - style_variant
- Etichete (4):
  - 0 = simple chair
  - 1 = chair with backrest
  - 2 = bar chair
  - 3 = stool

## Dataset mese (tables_dataset.csv)

- Locatie: data/generated/tables_dataset.csv
- Esantioane: 12000
- Caracteristici (7):
  - table_height, table_width, table_depth
  - leg_count, leg_thickness
  - has_apron
  - style_variant
- Etichete (3):
  - 0 = low_table
  - 1 = dining_table
  - 2 = bar_table

## Dataset dulapuri (cabinets_dataset.csv)

- Locatie: data/generated/cabinets_dataset.csv
- Esantioane: 12000
- Caracteristici (7):
  - cabinet_height, cabinet_width, cabinet_depth
  - wall_thickness
  - door_type, door_count
  - style_variant
- Etichete (3):
  - 0 = single_door
  - 1 = double_door
  - 2 = tall_cabinet

## Dataset frigidere (fridges_dataset.csv)

- Locatie: data/generated/fridges_dataset.csv
- Esantioane: 12000
- Caracteristici (8):
  - fridge_height, fridge_width, fridge_depth
  - door_thickness, handle_length
  - freezer_ratio, freezer_position
  - style_variant
- Etichete (2):
  - 0 = top_freezer
  - 1 = bottom_freezer

## Dataset aragazuri (stoves_dataset.csv)

- Locatie: data/generated/stoves_dataset.csv
- Esantioane: 12000
- Caracteristici (9):
  - stove_height, stove_width, stove_depth
  - oven_height_ratio
  - handle_length, glass_thickness
  - style_variant
  - burner_count, knob_count
- Etichete (3):
  - 0 = gas
  - 1 = electric
  - 2 = induction

## Rezultate preprocesare

- Date curatate: data/processed/*_clean.csv
- Date scalate: data/processed/*_scaled.csv
- Seturi splitate:
  - Scaune: data/chairs/train, data/chairs/validation, data/chairs/test
  - Mese: data/tables/train, data/tables/validation, data/tables/test
  - Dulapuri: data/cabinets/train, data/cabinets/validation, data/cabinets/test
  - Frigidere: data/fridges/train, data/fridges/validation, data/fridges/test
  - Aragazuri: data/stoves/train, data/stoves/validation, data/stoves/test
