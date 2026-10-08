# Model calibration (2026-10-08T19:29Z)

AUC = how often a real find scores above a comparison point (0.5 = no skill, 1 = perfect).
Habitat: finds vs all fungi records in the region (spatial cross-validation).
Timing: find days vs nearby other days at the same place (trained on earlier years, tested on the last two).

## Habitat

| Region | Species | Finds | Rules | Learned (CV) | Used |
|---|---|---|---|---|---|
| massif-central-alps | Liberty cap | 3 | - | - | no |
| massif-central-alps | Field mushroom (rosé des prés) | 12 | 0.77 | - | no |
| massif-central-alps | Cep (cèpe) | 104 | 0.70 | 0.74 | yes |
| massif-central-alps | Chanterelle (girolle) | 43 | 0.59 | 0.68 | yes |
| massif-central-alps | Winter chanterelle | 31 | 0.78 | - | no |
| massif-central-alps | Horn of plenty (trompette) | 15 | 0.48 | - | no |
| massif-central-alps | Hedgehog (pied de mouton) | 39 | 0.57 | - | no |
| brasov-fagaras | Liberty cap | 0 | - | - | no |
| brasov-fagaras | Field mushroom (rosé des prés) | 2 | - | - | no |
| brasov-fagaras | Cep (cèpe) | 22 | 0.46 | - | no |
| brasov-fagaras | Chanterelle (girolle) | 10 | 0.36 | - | no |
| brasov-fagaras | Winter chanterelle | 2 | - | - | no |
| brasov-fagaras | Horn of plenty (trompette) | 4 | - | - | no |
| brasov-fagaras | Hedgehog (pied de mouton) | 0 | - | - | no |
| scotland-south | Liberty cap | 8 | 0.50 | - | no |
| scotland-south | Field mushroom (rosé des prés) | 96 | 0.60 | 0.61 | no |
| scotland-south | Cep (cèpe) | 380 | 0.55 | 0.56 | no |
| scotland-south | Chanterelle (girolle) | 340 | 0.50 | 0.56 | no |
| scotland-south | Winter chanterelle | 96 | 0.48 | 0.66 | yes |
| scotland-south | Horn of plenty (trompette) | 22 | 0.68 | - | no |
| scotland-south | Hedgehog (pied de mouton) | 276 | 0.55 | 0.57 | no |
| scotland-highland | Liberty cap | 103 | 0.68 | 0.57 | no |
| scotland-highland | Field mushroom (rosé des prés) | 60 | 0.60 | 0.49 | no |
| scotland-highland | Cep (cèpe) | 979 | 0.53 | 0.51 | no |
| scotland-highland | Chanterelle (girolle) | 1024 | 0.44 | 0.49 | no |
| scotland-highland | Winter chanterelle | 254 | 0.65 | 0.59 | no |
| scotland-highland | Horn of plenty (trompette) | 11 | 0.50 | - | no |
| scotland-highland | Hedgehog (pied de mouton) | 467 | 0.46 | 0.52 | no |

## Timing

| Country | Species | Dated finds | Rules (test) | Learned (test) | Learned params | Used |
|---|---|---|---|---|---|---|
| FR | Liberty cap | 0 | - | - |  | no |
| FR | Field mushroom (rosé des prés) | 7 | - | - |  | no |
| FR | Cep (cèpe) | 39 | 0.64 | - |  | no |
| FR | Chanterelle (girolle) | 27 | 0.34 | - |  | no |
| FR | Winter chanterelle | 19 | 0.67 | - |  | no |
| FR | Horn of plenty (trompette) | 9 | 0.68 | - |  | no |
| FR | Hedgehog (pied de mouton) | 13 | - | - |  | no |
| RO | Liberty cap | 0 | - | - |  | no |
| RO | Field mushroom (rosé des prés) | 2 | - | - |  | no |
| RO | Cep (cèpe) | 13 | 0.50 | - |  | no |
| RO | Chanterelle (girolle) | 8 | 0.61 | - |  | no |
| RO | Winter chanterelle | 2 | - | - |  | no |
| RO | Horn of plenty (trompette) | 3 | - | - |  | no |
| RO | Hedgehog (pied de mouton) | 0 | - | - |  | no |
| GB-SCT | Liberty cap | 6 | 0.19 | - |  | no |
| GB-SCT | Field mushroom (rosé des prés) | 34 | 0.59 | - |  | no |
| GB-SCT | Cep (cèpe) | 372 | 0.67 | 0.63 | {'lag': [14, 19], 'rain_need': 50, 'temp': [6, 9, 19, 22]} | no |
| GB-SCT | Chanterelle (girolle) | 441 | 0.58 | 0.62 | {'lag': [3, 8], 'rain_need': 50, 'temp': [6, 9, 15, 18]} | yes |
| GB-SCT | Winter chanterelle | 80 | 0.49 | 0.54 | {'lag': [3, 8], 'rain_need': 35, 'temp': [-4, -1, 13, 16]} | no |
| GB-SCT | Horn of plenty (trompette) | 6 | 0.49 | - |  | no |
| GB-SCT | Hedgehog (pied de mouton) | 178 | 0.64 | 0.68 | {'lag': [10, 15], 'rain_need': 10, 'temp': [4, 7, 13, 16]} | yes |

## Season (month weights Jan..Dec)

- FR Cep (cèpe) (n=524): [0.07, 0.02, 0.02, 0.08, 0.3, 0.87, 1.0, 0.82, 0.9, 0.83, 0.51, 0.25]
- FR Chanterelle (girolle) (n=245): [0.02, 0.02, 0.02, 0.03, 0.3, 0.98, 1.0, 0.47, 0.22, 0.14, 0.12, 0.09]
- FR Winter chanterelle (n=157): [0.07, 0.15, 0.2, 0.13, 0.02, 0.03, 0.17, 0.4, 0.67, 1.0, 0.79, 0.21]
- FR Horn of plenty (trompette) (n=117): [0.1, 0.02, 0.04, 0.11, 0.04, 0.06, 0.37, 0.57, 0.63, 1.0, 1.0, 0.39]
- FR Hedgehog (pied de mouton) (n=318): [0.11, 0.04, 0.03, 0.04, 0.03, 0.12, 0.45, 0.83, 1.0, 0.95, 0.74, 0.36]
- GB-SCT Liberty cap (n=505): [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.09, 0.35, 0.8, 1.0, 0.54, 0.1]
- GB-SCT Field mushroom (rosé des prés) (n=303): [0.02, 0.02, 0.02, 0.02, 0.07, 0.21, 0.42, 0.84, 1.0, 0.63, 0.23, 0.03]
- GB-SCT Cep (cèpe) (n=2873): [0.02, 0.02, 0.02, 0.02, 0.02, 0.07, 0.28, 0.78, 1.0, 0.52, 0.11, 0.02]
- GB-SCT Chanterelle (girolle) (n=2972): [0.02, 0.02, 0.02, 0.02, 0.02, 0.25, 0.78, 1.0, 0.75, 0.37, 0.16, 0.07]
- GB-SCT Winter chanterelle (n=660): [0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.06, 0.38, 0.87, 1.0, 0.64, 0.24]
- GB-SCT Horn of plenty (trompette) (n=72): [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.39, 1.0, 0.8, 0.21, 0.02]
- GB-SCT Hedgehog (pied de mouton) (n=1292): [0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.06, 0.42, 1.0, 0.96, 0.5, 0.21]
