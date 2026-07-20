# Ether & Ester Combustion Kinetics Predictor

This Streamlit application allows you to upload an Excel or CSV file containing ether and ester reaction data and perform rate constant ($log_{10}(k)$) predictions using the trained models.

The models have been pre-trained on the full kinetics dataset and serialized into the `saved_models/` directory, so the application runs immediately without retraining.

---

## 🚀 Live Application

- **Web App URL:** [https://bjvsaaxxmyxmvb9zxbhcpn.streamlit.app/](https://bjvsaaxxmyxmvb9zxbhcpn.streamlit.app/)

## Instructions

### 1. Install Packages
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

### 2. Train Models Once (Optional)
To run cross-validation and output metrics as standard in the codebase:
```bash
python train_rf.py
python train_xgb.py
python train_fnn.py
python train_gfnn.py
```
*Note: The serialized prediction models are already included in the `saved_models/` folder, so you do not need to retrain them to use the web application.*

### 3. Launch App
Launch the Streamlit web application:
```bash
streamlit run app.py
```

---

## Upload File Format

The uploaded file (.xlsx or .csv) should contain the following columns:
- **`reactant_SMILES`** (Required): The SMILES string of the reactant.
- **`reaction_class`** (Required): One of the five classes (`H-abstraction`, `H-abstraction / overall OH reaction`, `O2 addition`, `Unimolecular decomposition`, `beta scission`).
- **`temperature_K`** (Required): Temperature in Kelvin.
- **`reaction_id`** (Optional): ID of the reaction.
- **`experimental_log10_k`** (Optional): If present, the application will automatically calculate R², RMSE, MAE, and render a parity plot.

---

## Literature References & Data Sources

The dataset of 36 combustion reactions across ether and ester families was compiled from the following 20 primary kinetics studies:

1. **Bänsch et al. (2013)** - Dimethyl Ether H-abstraction | [10.1021/jp405724a](https://doi.org/10.1021/jp405724a)
2. **Bänsch & Olzmann (2019)** - Dimethoxymethane H-abstraction | [10.1016/j.cplett.2019.01.053](https://doi.org/10.1016/j.cplett.2019.01.053)
3. **Mach et al. (2025)** - Methyl Formate & OME2 Oxidation | [10.1021/acs.jpca.4c02524](https://doi.org/10.1021/acs.jpca.4c02524), [10.1002/kin.21794](https://doi.org/10.1002/kin.21794)
4. **Lee (1981)** - Dimethyl Ether + OH Kinetics | [10.1063/1.441456](https://doi.org/10.1063/1.441456)
5. **Belmekki et al. (2021)** - Diethyl Ether Oxidation | [10.1021/acs.energyfuels.1c01408](https://doi.org/10.1021/acs.energyfuels.1c01408)
6. **Mellouki et al. (1995)** - Aliphatic Ethers + OH Kinetics | [10.1002/kin.550270806](https://doi.org/10.1002/kin.550270806)
7. **Arif et al. (1997)** - MTBE + OH Reaction | [10.1021/jp963119w](https://doi.org/10.1021/jp963119w)
8. **Lam et al. (2012)** - Methyl Esters Shock Tube Kinetics | [10.1021/jp310256j](https://doi.org/10.1021/jp310256j)
9. **El Boudali et al. (1996)** - Alkyl Acetates + OH Kinetics | [10.1021/jp9606218](https://doi.org/10.1021/jp9606218)
10. **Curran et al. (1998, 2000)** - Dimethyl Ether Radical Reactions | [10.1002/1097-4601(2000)32:12<741::AID-KIN2>3.0.CO;2-9](https://doi.org/10.1002/1097-4601(2000)32:12<741::AID-KIN2>3.0.CO;2-9)
11. **Auzmendi-Murua & Bozzelli (2014)** - Tetrahydrofuran Radical O2 Addition | [10.1021/jp412590g](https://doi.org/10.1021/jp412590g)
12. **Westbrook et al. (2009)** - Gamma-butyrolactone O2 Addition | [10.1016/j.proci.2008.06.106](https://doi.org/10.1016/j.proci.2008.06.106)
13. **Yamamoto et al. (1992)** - Gamma-butyrolactone Pyrolysis | [10.1246/bcsj.65.3112](https://doi.org/10.1246/bcsj.65.3112)
14. **Blades (1954)** - Ethyl Acetate Pyrolysis | [10.1139/v54-049](https://doi.org/10.1139/v54-049)
15. **Yang et al. (2013)** - Methyl Acetate Radical Beta-scission | [10.1016/j.combustflame.2013.06.017](https://doi.org/10.1016/j.combustflame.2013.06.017)
16. **Hochgreb & Dryer (1992)** - 1,3,5-Trioxane Decomposition | [10.1021/j100180a055](https://doi.org/10.1021/j100180a055)
17. **Saheb & Bahadori (2020)** - 1,3,5-Trioxane H-abstraction | [10.1177/1468678319899252](https://doi.org/10.1177/1468678319899252)
18. **Le Calvé et al. (1997)** - Alkyl Esters OH Kinetics | [10.1021/jp972369p](https://doi.org/10.1021/jp972369p)
19. **J. Phys. Chem. A (2021)** - 2-Methyltetrahydrofuran Decomposition | [10.1021/acs.jpca.0c11490](https://doi.org/10.1021/acs.jpca.0c11490)
20. **Liu & Farooq (2023)** - Gamma-valerolactone Kinetics | [10.1016/j.combustflame.2023.112771](https://doi.org/10.1016/j.combustflame.2023.112771)
