# Ether & Ester Combustion Kinetics Predictor

This Streamlit application allows you to upload an Excel or CSV file containing ether and ester reaction data and perform rate constant ($log_{10}(k)$) predictions using the trained models.

The models have been pre-trained on the full kinetics dataset and serialized into the `saved_models/` directory, so the application runs immediately without retraining.

---

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
