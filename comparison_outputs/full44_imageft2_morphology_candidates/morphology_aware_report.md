# Morphology-aware CPFLOW image generation

This run uses the current CPFLOW checkpoint and improves image display by generating multiple candidates,
ranking them against real Cell Painting statistics, and plotting the best-scoring samples.

Drugs: anisomycin, monastrol, puromycin, tunicamycin, neomycin, nocodazole, atropine, nystatin, h-7, methotrexate, cisplatin, docetaxel, simvastatin, chlorambucil, aphidicolin, olomoucine, forskolin, floxuridine, etoposide, vincristine, pp-2, quercetin, staurosporine, cyclophosphamide, mg-132, sb-203580, pd-98059, temozolomide, ly-294002, sb-202190, alsterpaullone, roscovitine, emetine, hydroxyurea, doxorubicin, camptothecin, acyclovir, mitoxantrone, vinblastine, y-27632, raloxifene, calpeptin, genistein, colchicine
Candidates per drug: 32
Kept per drug: 1

| Drug | Best candidate | Best score | Foreground | Sharpness | Blockiness |
|---|---:|---:|---:|---:|---:|
| anisomycin | 10 | -0.223 | 0.624 | 0.056 | 0.509 |
| monastrol | 0 | -0.316 | 0.798 | 0.068 | 0.503 |
| puromycin | 7 | -0.098 | 0.689 | 0.067 | 0.492 |
| tunicamycin | 2 | -0.223 | 0.554 | 0.065 | 0.500 |
| neomycin | 21 | -0.146 | 0.821 | 0.071 | 0.513 |
| nocodazole | 7 | -0.335 | 0.350 | 0.050 | 0.502 |
| atropine | 25 | -0.184 | 0.503 | 0.063 | 0.484 |
| nystatin | 30 | -0.096 | 0.245 | 0.048 | 0.482 |
| h-7 | 27 | -0.115 | 0.429 | 0.064 | 0.507 |
| methotrexate | 16 | -0.226 | 0.565 | 0.062 | 0.504 |
| cisplatin | 10 | -0.173 | 0.811 | 0.070 | 0.511 |
| docetaxel | 26 | -0.145 | 0.552 | 0.060 | 0.510 |
| simvastatin | 24 | -0.225 | 0.233 | 0.042 | 0.515 |
| chlorambucil | 22 | -0.188 | 0.835 | 0.067 | 0.515 |
| aphidicolin | 1 | -0.292 | 0.716 | 0.064 | 0.505 |
| olomoucine | 9 | -0.137 | 0.683 | 0.069 | 0.521 |
| forskolin | 4 | -0.217 | 0.592 | 0.063 | 0.500 |
| floxuridine | 27 | -0.213 | 0.442 | 0.059 | 0.495 |
| etoposide | 6 | -0.228 | 0.838 | 0.084 | 0.505 |
| vincristine | 9 | -0.142 | 0.285 | 0.035 | 0.512 |
| pp-2 | 30 | -0.165 | 0.433 | 0.060 | 0.491 |
| quercetin | 14 | -0.105 | 0.340 | 0.056 | 0.492 |
| staurosporine | 7 | -0.606 | 0.400 | 0.058 | 0.499 |
| cyclophosphamide | 9 | -0.112 | 0.414 | 0.058 | 0.497 |
| mg-132 | 15 | -0.353 | 0.442 | 0.055 | 0.505 |
| sb-203580 | 27 | -0.159 | 0.850 | 0.077 | 0.505 |
| pd-98059 | 24 | -0.103 | 0.799 | 0.062 | 0.501 |
| temozolomide | 28 | -0.177 | 0.826 | 0.072 | 0.502 |
| ly-294002 | 8 | -0.173 | 0.839 | 0.070 | 0.504 |
| sb-202190 | 8 | -0.317 | 0.724 | 0.060 | 0.499 |
| alsterpaullone | 20 | -0.465 | 0.899 | 0.067 | 0.499 |
| roscovitine | 26 | -0.223 | 0.414 | 0.054 | 0.486 |
| emetine | 17 | -0.151 | 0.319 | 0.048 | 0.507 |
| hydroxyurea | 12 | -0.104 | 0.581 | 0.059 | 0.503 |
| doxorubicin | 28 | -0.390 | 0.829 | 0.083 | 0.502 |
| camptothecin | 7 | -0.477 | 0.251 | 0.048 | 0.521 |
| acyclovir | 11 | -0.126 | 0.564 | 0.055 | 0.485 |
| mitoxantrone | 29 | -0.384 | 0.361 | 0.055 | 0.493 |
| vinblastine | 28 | -0.140 | 0.134 | 0.038 | 0.480 |
| y-27632 | 26 | -0.214 | 0.838 | 0.073 | 0.507 |
| raloxifene | 28 | -0.192 | 0.645 | 0.065 | 0.488 |
| calpeptin | 0 | -0.216 | 0.386 | 0.056 | 0.490 |
| genistein | 11 | -0.170 | 0.516 | 0.059 | 0.494 |
| colchicine | 6 | -0.347 | 0.788 | 0.076 | 0.498 |

Generated files:

- `figure_morphology_aware_cpflow_plate.png/svg/pdf`
- `candidate_quality_scores.csv`
- `images/<drug>/candidate_*_raw.png` and `candidate_*_display.png`

Important limitation: this is morphology-aware sampling and candidate selection, not a newly retrained image model.
It can select cleaner samples from the current model, but true article-level image fidelity likely requires image-focused fine-tuning.
