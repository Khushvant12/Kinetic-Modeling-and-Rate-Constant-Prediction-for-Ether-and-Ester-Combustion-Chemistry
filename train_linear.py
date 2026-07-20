import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MolSurf
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

sys.stdout.reconfigure(encoding='utf-8')

def load_data():
    excel_path = "complete rxn dataset.xlsx"
    df_master = pd.read_excel(excel_path, sheet_name="Master_Reactions")
    df_kt = pd.read_excel(excel_path, sheet_name="kT_Generator")
    
    list_of_15 = [
        'MolWt', 'ExactMolWt', 'MolLogP', 'TPSA', 'HeavyAtomCount', 
        'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds', 'RingCount', 
        'NumAliphaticRings', 'NumSaturatedRings', 'FractionCSP3', 
        'LabuteASA', 'BalabanJ', 'NumValenceElectrons'
    ]
    
    def get_descriptor(mol, name):
        func = getattr(Descriptors, name, None)
        if func is None:
            func = getattr(Lipinski, name, None)
        if func is None:
            func = getattr(MolSurf, name, None)
        if func is not None:
            return func(mol)
        raise ValueError(f"Descriptor {name} not found in RDKit!")
        
    unique_reactants = df_master[['reaction_id', 'reactant_SMILES', 'cyclic_or_acyclic']].drop_duplicates().copy()
    reactant_desc = {}
    for idx, row in unique_reactants.iterrows():
        rxn_id = row['reaction_id']
        smiles = str(row['reactant_SMILES']).strip()
        mol = Chem.MolFromSmiles(smiles)
        desc_vals = {}
        for name in list_of_15:
            val = get_descriptor(mol, name)
            desc_vals[name] = 0.0 if (np.isnan(val) or np.isinf(val)) else val
        reactant_desc[rxn_id] = desc_vals
        
    df_desc_unique = pd.DataFrame.from_dict(reactant_desc, orient='index')
    corr_matrix = df_desc_unique.corr().abs()
    
    retained_descriptors = []
    for desc in list_of_15:
        keep = True
        for other in retained_descriptors:
            if corr_matrix.loc[desc, other] > 0.95:
                keep = False
                break
        if keep:
            retained_descriptors.append(desc)
            
    unique_features = df_desc_unique[retained_descriptors].copy()
    cyclic_map = {rxn: (1.0 if str(cyc).strip().lower() == 'cyclic' else 0.0) 
                  for rxn, cyc in zip(unique_reactants['reaction_id'], unique_reactants['cyclic_or_acyclic'])}
    unique_features['cyclic_or_acyclic'] = unique_features.index.map(cyclic_map)
    
    df_merged = df_kt.merge(unique_features, left_on='reaction_id', right_index=True, how='inner')
    df_encoded = pd.get_dummies(df_merged, columns=['reaction_class'], drop_first=False)
    
    encoded_class_cols = [col for col in df_encoded.columns if col.startswith('reaction_class_')]
    numeric_cols = retained_descriptors + ['invT_1000_per_K']
    feature_cols = numeric_cols + ['cyclic_or_acyclic'] + encoded_class_cols
    
    X = df_encoded[feature_cols].copy()
    y = df_encoded['log10_k'].values
    groups = df_encoded['reaction_id'].values
    
    for col in encoded_class_cols:
        X[col] = X[col].astype(float)
        
    return df_encoded, X, y, groups, numeric_cols, feature_cols

def main():
    df, X, y, groups, numeric_cols, feature_cols = load_data()
    
    # Verify Target Variable Limits
    print(f"Minimum log10(k): {np.min(y):.6f}")
    print(f"Maximum log10(k): {np.max(y):.6f}")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols)
        ],
        remainder='passthrough'
    )
    
    # Use Ridge Regression to handle descriptor multicollinearity
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', Ridge(alpha=0.1))
    ])
    
    gkf = GroupKFold(n_splits=5)
    
    all_true = []
    all_pred = []
    
    fold_metrics = []
    print("\nTraining Linear Regression (Ridge α=0.1) with GroupKFold...")
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        
        all_true.extend(y_test)
        all_pred.extend(preds)
        
        fold_r2 = r2_score(y_test, preds)
        fold_rmse = np.sqrt(mean_squared_error(y_test, preds))
        fold_mae = mean_absolute_error(y_test, preds)
        fold_metrics.append((fold_r2, fold_rmse, fold_mae))
        print(f"  Fold {fold+1}: R2 = {fold_r2:.4f}, RMSE = {fold_rmse:.4f}, MAE = {fold_mae:.4f}")
        
    overall_r2 = r2_score(all_true, all_pred)
    overall_rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    overall_mae = mean_absolute_error(all_true, all_pred)
    
    print(f"\nOverall Metrics:")
    print(f"R²:   {overall_r2:.4f}")
    print(f"RMSE: {overall_rmse:.4f}")
    print(f"MAE:  {overall_mae:.4f}")
    
    out_dir = "linear_outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    # Save parity plot
    plt.figure(figsize=(6, 6))
    plt.scatter(all_true, all_pred, alpha=0.6, color='#1f77b4', edgecolors='k')
    min_val = min(min(all_true), min(all_pred)) - 0.5
    max_val = max(max(all_true), max(all_pred)) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel("True log10(k)")
    plt.ylabel("Predicted log10(k)")
    plt.title("Linear Regression (Ridge) Parity Plot")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lr_parity.png"), dpi=300)
    plt.close()
    
    # Save metrics txt
    with open(os.path.join(out_dir, "lr_metrics.txt"), 'w', encoding='utf-8') as f:
        f.write("Model: Linear Regression\n\n")
        f.write("Fold-wise Metrics:\n")
        for i, (r2, rmse, mae) in enumerate(fold_metrics):
            f.write(f"  Fold {i+1} -> R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}\n")
        f.write("\nOverall Metrics (Out-of-Fold):\n")
        f.write(f"R2: {overall_r2:.6f}\n")
        f.write(f"RMSE: {overall_rmse:.6f}\n")
        f.write(f"MAE: {overall_mae:.6f}\n")

if __name__ == "__main__":
    main()
