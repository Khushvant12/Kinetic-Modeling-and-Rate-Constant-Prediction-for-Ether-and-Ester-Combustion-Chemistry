# Dependencies:
# - PyTorch (torch)
# - RDKit (rdkit)
# - Pandas (pandas)
# - Numpy (numpy)
# - Scikit-learn (scikit-learn)
# - Matplotlib (matplotlib)
# Note: PyTorch Geometric is not required as graph representations and message passing layers are implemented manually in PyTorch.

import os
import sys
import time
import random
import copy
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MolSurf
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.stdout.reconfigure(encoding='utf-8')

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==============================================================================
# FEATURIZATION FUNCTIONS
# ==============================================================================

def one_hot_encoding(val, choices):
    encoding = [0.0] * len(choices)
    if val in choices:
        encoding[choices.index(val)] = 1.0
    return encoding

def get_atom_features(atom):
    # Atomic number (H=1, C=6, O=8)
    at_num_enc = one_hot_encoding(atom.GetAtomicNum(), [1, 6, 8])
    # Degree
    degree_enc = one_hot_encoding(atom.GetTotalDegree(), [0, 1, 2, 3, 4])
    # Hybridization
    hybrid = atom.GetHybridization()
    hybrid_enc = one_hot_encoding(hybrid, [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3
    ])
    # Aromaticity
    arom = [1.0 if atom.GetIsAromatic() else 0.0]
    # Formal charge
    charge = [float(atom.GetFormalCharge())]
    # Number of attached hydrogens
    hs_enc = one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    
    # Total node features length = 3 + 5 + 3 + 1 + 1 + 5 = 18
    return np.array(at_num_enc + degree_enc + hybrid_enc + arom + charge + hs_enc, dtype=np.float32)

def get_bond_features(bond):
    # Bond type
    bt = bond.GetBondType()
    bt_enc = one_hot_encoding(bt, [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ])
    # Conjugation
    conj = [1.0 if bond.GetIsConjugated() else 0.0]
    # Ring membership
    ring = [1.0 if bond.IsInRing() else 0.0]
    
    # Total bond features length = 4 + 1 + 1 = 6
    return np.array(bt_enc + conj + ring, dtype=np.float32)

def build_reactant_graphs(df_master):
    unique_reactants = df_master[['reaction_id', 'reactant_SMILES']].drop_duplicates()
    graphs = {}
    for idx, row in unique_reactants.iterrows():
        rxn_id = row['reaction_id']
        smiles = str(row['reactant_SMILES']).strip()
        mol = Chem.MolFromSmiles(smiles)
        
        # Build node features
        node_feats = []
        for atom in mol.GetAtoms():
            node_feats.append(get_atom_features(atom))
        node_feats = np.array(node_feats, dtype=np.float32)
        
        # Build edge features and edge index
        edge_index = []
        edge_feats = []
        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            bf = get_bond_features(bond)
            
            # Undirected graph representation
            edge_index.append([u, v])
            edge_feats.append(bf)
            edge_index.append([v, u])
            edge_feats.append(bf)
            
        if len(edge_index) == 0:
            # Fallback for single-node molecule without bonds
            edge_index = [[0, 0]]
            edge_feats = [np.zeros(6, dtype=np.float32)]
            
        edge_index = np.array(edge_index, dtype=np.int64).T
        edge_feats = np.array(edge_feats, dtype=np.float32)
        
        graphs[rxn_id] = {
            'x': torch.tensor(node_feats, dtype=torch.float32),
            'edge_index': torch.tensor(edge_index, dtype=torch.long),
            'edge_attr': torch.tensor(edge_feats, dtype=torch.float32)
        }
    return graphs

# ==============================================================================
# DATA LOADING
# ==============================================================================

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
        
    return df_encoded, X, y, groups, numeric_cols, feature_cols, df_master

# ==============================================================================
# PYTORCH GRAPH DATASET & COLLATOR
# ==============================================================================

class ReactionGraphDataset(Dataset):
    def __init__(self, X_scaled, y, groups, graphs):
        self.X_scaled = X_scaled
        self.y = y
        self.groups = groups
        self.graphs = graphs
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        rxn_id = self.groups[idx]
        graph = self.graphs[rxn_id]
        aux = torch.tensor(self.X_scaled[idx], dtype=torch.float32)
        target = float(self.y[idx])
        return {
            'x': graph['x'],
            'edge_index': graph['edge_index'],
            'edge_attr': graph['edge_attr'],
            'aux': aux,
            'y': target,
            'reaction_id': rxn_id
        }

def collate_graphs(graph_list):
    x_list = [g['x'] for g in graph_list]
    edge_index_list = [g['edge_index'] for g in graph_list]
    edge_attr_list = [g['edge_attr'] for g in graph_list]
    aux_list = [g['aux'] for g in graph_list]
    y_list = [g['y'] for g in graph_list]
    
    batch_idx = []
    accum_nodes = 0
    adjusted_edge_indices = []
    
    for i, x in enumerate(x_list):
        num_nodes = x.size(0)
        batch_idx.extend([i] * num_nodes)
        
        adj_edge = edge_index_list[i] + accum_nodes
        adjusted_edge_indices.append(adj_edge)
        
        accum_nodes += num_nodes
        
    x_batch = torch.cat(x_list, dim=0)
    edge_index_batch = torch.cat(adjusted_edge_indices, dim=1)
    edge_attr_batch = torch.cat(edge_attr_list, dim=0)
    batch_idx_batch = torch.tensor(batch_idx, dtype=torch.long)
    
    aux_batch = torch.stack(aux_list, dim=0)
    y_batch = torch.tensor(y_list, dtype=torch.float32)
    
    return {
        'x': x_batch,
        'edge_index': edge_index_batch,
        'edge_attr': edge_attr_batch,
        'batch': batch_idx_batch,
        'aux': aux_batch,
        'y': y_batch
    }

# ==============================================================================
# MODEL DEFINITION
# ==============================================================================

class MessagePassingLayer(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, out_dim):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_in_dim + edge_in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_in_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        x_j = x[col]
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        msg = self.msg_mlp(msg_input)
        
        num_atoms = x.size(0)
        agg_msg = torch.zeros(num_atoms, msg.size(1), device=x.device)
        agg_msg.index_add_(0, row, msg)
        
        update_input = torch.cat([x, agg_msg], dim=-1)
        out = self.update_mlp(update_input)
        return out

def global_mean_pool(x, batch):
    num_graphs = batch.max().item() + 1
    sum_pooled = torch.zeros(num_graphs, x.size(1), device=x.device)
    sum_pooled.index_add_(0, batch, x)
    
    ones = torch.ones(x.size(0), 1, device=x.device)
    counts = torch.zeros(num_graphs, 1, device=x.device)
    counts.index_add_(0, batch, ones)
    
    counts = torch.clamp(counts, min=1.0)
    return sum_pooled / counts

class GFNN(nn.Module):
    def __init__(self, node_in_dim=18, edge_in_dim=6, aux_dim=16, hidden_dim=64, out_dim=1):
        super().__init__()
        
        self.conv1 = MessagePassingLayer(node_in_dim, edge_in_dim, hidden_dim)
        self.conv2 = MessagePassingLayer(hidden_dim, edge_in_dim, hidden_dim)
        self.conv3 = MessagePassingLayer(hidden_dim, edge_in_dim, hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + aux_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )
        
    def forward(self, x, edge_index, edge_attr, batch, aux):
        h = self.conv1(x, edge_index, edge_attr)
        h = torch.relu(h)
        
        h = self.conv2(h, edge_index, edge_attr)
        h = torch.relu(h)
        
        h = self.conv3(h, edge_index, edge_attr)
        h = torch.relu(h)
        
        g = global_mean_pool(h, batch)
        
        combined = torch.cat([g, aux], dim=1)
        out = self.fc(combined)
        return out.squeeze(-1)

# ==============================================================================
# TRAINING & EVALUATION LOOP
# ==============================================================================

def train_one_fold(fold, train_loader, val_loader, node_in_dim, edge_in_dim, aux_dim, max_epochs=600, patience=60):
    model = GFNN(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim, aux_dim=aux_dim, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(
                batch['x'],
                batch['edge_index'],
                batch['edge_attr'],
                batch['batch'],
                batch['aux']
            )
            loss = criterion(pred, batch['y'])
            loss.backward()
            optimizer.step()
        
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(
                    batch['x'],
                    batch['edge_index'],
                    batch['edge_attr'],
                    batch['batch'],
                    batch['aux']
                )
                loss = criterion(pred, batch['y'])
                epoch_val_loss += loss.item() * len(batch['y'])
        epoch_val_loss /= len(val_loader.dataset)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    return best_model_state, best_epoch, best_val_loss

def update_final_comparison(overall_r2, overall_rmse, overall_mae):
    file_path = "final_comparison.txt"
    if not os.path.exists(file_path):
        return
        
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    r2_str = f" {overall_r2:.4f}".ljust(14)
    rmse_str = f" {overall_rmse:.4f}".ljust(14)
    mae_str = f" {overall_mae:.4f}".ljust(13)
    gfnn_line = f"GFNN                      |{r2_str}|{rmse_str}|{mae_str}\n"
    
    new_lines = []
    for line in lines:
        if line.strip().startswith("GFNN"):
            continue
        new_lines.append(line)
        
    while new_lines and new_lines[-1].strip() == "":
        new_lines.pop()
        
    new_lines.append(gfnn_line)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    set_seed(42)
    
    # Load dataset
    df, X, y, groups, numeric_cols, feature_cols, df_master = load_data()
    graphs = build_reactant_graphs(df_master)
    
    print(f"Minimum log10(k): {np.min(y):.6f}")
    print(f"Maximum log10(k): {np.max(y):.6f}")
    
    gkf = GroupKFold(n_splits=5)
    
    all_true = []
    all_pred = []
    fold_metrics = []
    optimal_epochs = []
    
    node_in_dim = 18
    edge_in_dim = 6
    aux_dim = X.shape[1]
    
    start_time = time.time()
    
    print("\nTraining Graph Feedforward Neural Network (GFNN) with GroupKFold...")
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standard scale numeric columns
        preprocessor = ColumnTransformer(
            transformers=[('num', StandardScaler(), numeric_cols)],
            remainder='passthrough'
        )
        
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_test_scaled = preprocessor.transform(X_test)
        
        # Build DataLoaders
        train_ds = ReactionGraphDataset(X_train_scaled, y_train, groups[train_idx], graphs)
        val_ds = ReactionGraphDataset(X_test_scaled, y_test, groups[test_idx], graphs)
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_graphs)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_graphs)
        
        # Train fold
        best_state, best_epoch, best_val_loss = train_one_fold(
            fold, train_loader, val_loader, node_in_dim, edge_in_dim, aux_dim
        )
        optimal_epochs.append(best_epoch)
        
        # Predict on fold's validation/test split
        best_model = GFNN(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim, aux_dim=aux_dim, hidden_dim=64)
        best_model.load_state_dict(best_state)
        best_model.eval()
        
        preds = []
        with torch.no_grad():
            for batch in val_loader:
                out = best_model(
                    batch['x'],
                    batch['edge_index'],
                    batch['edge_attr'],
                    batch['batch'],
                    batch['aux']
                )
                preds.extend(out.numpy())
                
        all_true.extend(y_test)
        all_pred.extend(preds)
        
        fold_r2 = r2_score(y_test, preds)
        fold_rmse = np.sqrt(mean_squared_error(y_test, preds))
        fold_mae = mean_absolute_error(y_test, preds)
        fold_metrics.append((fold_r2, fold_rmse, fold_mae))
        
        print(f"  Fold {fold+1}: R2 = {fold_r2:.4f}, RMSE = {fold_rmse:.4f}, MAE = {fold_mae:.4f} (Best Epoch = {best_epoch+1})")
        
    total_training_time = time.time() - start_time
    
    overall_r2 = r2_score(all_true, all_pred)
    overall_rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    overall_mae = mean_absolute_error(all_true, all_pred)
    
    print(f"\nOverall Metrics:")
    print(f"R²:   {overall_r2:.4f}")
    print(f"RMSE: {overall_rmse:.4f}")
    print(f"MAE:  {overall_mae:.4f}")
    print(f"Total CV Training Time: {total_training_time:.2f} seconds")
    
    out_dir = "gfnn_outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    # Save parity plot
    plt.figure(figsize=(6, 6))
    plt.scatter(all_true, all_pred, alpha=0.6, color='#ff7f0e', edgecolors='k')
    min_val = min(min(all_true), min(all_pred)) - 0.5
    max_val = max(max(all_true), max(all_pred)) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel("True log10(k)")
    plt.ylabel("Predicted log10(k)")
    plt.title("Graph Feedforward Neural Network (GFNN) Parity Plot")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gfnn_parity.png"), dpi=300)
    plt.close()
    
    # Save metrics txt
    with open(os.path.join(out_dir, "gfnn_metrics.txt"), 'w', encoding='utf-8') as f:
        f.write("Model: Graph Feedforward Neural Network (GFNN)\n\n")
        f.write("Fold-wise Metrics:\n")
        for i, (r2, rmse, mae) in enumerate(fold_metrics):
            f.write(f"  Fold {i+1} -> R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}\n")
        f.write("\nOverall Metrics (Out-of-Fold):\n")
        f.write(f"R2: {overall_r2:.6f}\n")
        f.write(f"RMSE: {overall_rmse:.6f}\n")
        f.write(f"MAE: {overall_mae:.6f}\n")
        f.write(f"\nTraining Time: {total_training_time:.2f} seconds\n")
        
    # Fit final model on full dataset
    print("\nTraining final model on full dataset...")
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), numeric_cols)],
        remainder='passthrough'
    )
    X_scaled = preprocessor.fit_transform(X)
    
    full_ds = ReactionGraphDataset(X_scaled, y, groups, graphs)
    full_loader = DataLoader(full_ds, batch_size=32, shuffle=True, collate_fn=collate_graphs)
    
    avg_epochs = int(np.round(np.mean(optimal_epochs)))
    print(f"Training for {avg_epochs} epochs (average of CV runs)...")
    
    final_model = GFNN(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim, aux_dim=aux_dim, hidden_dim=64)
    optimizer = optim.Adam(final_model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    loss_history = []
    
    for epoch in range(avg_epochs):
        final_model.train()
        epoch_loss = 0.0
        for batch in full_loader:
            optimizer.zero_grad()
            pred = final_model(
                batch['x'],
                batch['edge_index'],
                batch['edge_attr'],
                batch['batch'],
                batch['aux']
            )
            loss = criterion(pred, batch['y'])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch['y'])
        epoch_loss /= len(full_ds)
        loss_history.append(epoch_loss)
        
    # Save Loss Curve
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, color='#e67e22', lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("GFNN Loss Curve (Final Fit)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gfnn_loss.png"), dpi=300)
    plt.close()
    print("Saved GFNN loss curve.")
    
    # Update final comparison file
    update_final_comparison(overall_r2, overall_rmse, overall_mae)
    print("Updated final_comparison.txt with GFNN metrics.")

if __name__ == "__main__":
    main()
