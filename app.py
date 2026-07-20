import os
import sys
import pickle
import io
import time
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MolSurf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Ether & Ester Combustion Kinetics Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Injector
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: #fafbfc;
    }
    
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    .kpi-card {
        background-color: #ffffff;
        padding: 22px;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(31, 119, 180, 0.05);
        border: 1px solid #e3edf7;
        margin-bottom: 15px;
        transition: transform 0.2s ease-in-out;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
    }
    
    .stButton > button, .stDownloadButton > button {
        background-color: #1f77b4 !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        box-shadow: 0 5px 15px rgba(31, 119, 180, 0.2) !important;
        padding: 10px 30px !important;
        transition: all 0.2s ease;
        cursor: pointer !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #1a6599 !important;
        box-shadow: 0 8px 20px rgba(31, 119, 180, 0.3) !important;
        transform: translateY(-1px);
    }
    
    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 2px dashed #1f77b4;
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 10px 25px rgba(31, 119, 180, 0.03);
    }
    
    .header-card {
        background: linear-gradient(135deg, #1f77b4, #5da2d5);
        color: white;
        padding: 30px;
        border-radius: 18px;
        box-shadow: 0 12px 30px rgba(31, 119, 180, 0.15);
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# GFNN PYTORCH CLASS DEFINITIONS (Self-contained for loading)
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
# HELPER FUNCTIONS FOR INFERENCE
# ==============================================================================

retained_descriptors = ['MolWt', 'MolLogP', 'TPSA', 'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds', 'RingCount', 'FractionCSP3', 'BalabanJ']
numeric_cols = retained_descriptors + ['invT_1000_per_K']
feature_cols = numeric_cols + ['cyclic_or_acyclic', 'reaction_class_H-abstraction', 'reaction_class_H-abstraction / overall OH reaction', 'reaction_class_O2 addition', 'reaction_class_Unimolecular decomposition', 'reaction_class_beta scission']
allowed_classes = ['H-abstraction', 'H-abstraction / overall OH reaction', 'O2 addition', 'Unimolecular decomposition', 'beta scission']

def normalize_reaction_class(input_class):
    if not isinstance(input_class, str):
        return None
    val = input_class.strip().lower().replace('_', '-').replace(' ', '-')
    
    # 1. H-abstraction / overall OH reaction (OH abstraction)
    if val in [
        'oh-abstraction',
        'oh-abstraction',
        'h-abstraction-/-overall-oh-reaction',
        'h-abstraction-overall-oh-reaction',
        'overall-oh-reaction',
        'oh-reaction',
        'h-abstraction-overall-oh',
        'oh'
    ]:
        return 'H-abstraction / overall OH reaction'
        
    # 2. H-abstraction (by other radicals)
    if val in ['h-abstraction', 'h']:
        return 'H-abstraction'
        
    # 3. O2 addition
    if val in ['o2-addition', 'o2', 'addition']:
        return 'O2 addition'
        
    # 4. beta scission
    if val in ['beta-scission', 'beta', 'scission']:
        return 'beta scission'
        
    # 5. Unimolecular decomposition
    if val in ['unimolecular-decomposition', 'decomposition', 'unimolecular']:
        return 'Unimolecular decomposition'
        
    # Fuzzy match fallbacks
    if 'oh' in val and 'abstract' in val:
        return 'H-abstraction / overall OH reaction'
    if 'h' in val and 'abstract' in val:
        return 'H-abstraction'
    if 'o2' in val or 'oxygen' in val:
        return 'O2 addition'
    if 'beta' in val or 'scission' in val:
        return 'beta scission'
    if 'decomposition' in val or 'unimolecular' in val:
        return 'Unimolecular decomposition'
        
    return None


def get_descriptor(mol, name):
    func = getattr(Descriptors, name, None)
    if func is None:
        func = getattr(Lipinski, name, None)
    if func is None:
        func = getattr(MolSurf, name, None)
    if func is not None:
        return func(mol)
    raise ValueError(f"Descriptor {name} not found in RDKit!")

def extract_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    desc_vals = {}
    for name in retained_descriptors:
        val = get_descriptor(mol, name)
        desc_vals[name] = 0.0 if (np.isnan(val) or np.isinf(val)) else val
    return desc_vals

def one_hot_encoding(val, choices):
    encoding = [0.0] * len(choices)
    if val in choices:
        encoding[choices.index(val)] = 1.0
    return encoding

def get_atom_features(atom):
    at_num_enc = one_hot_encoding(atom.GetAtomicNum(), [1, 6, 8])
    degree_enc = one_hot_encoding(atom.GetTotalDegree(), [0, 1, 2, 3, 4])
    hybrid = atom.GetHybridization()
    hybrid_enc = one_hot_encoding(hybrid, [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3
    ])
    arom = [1.0 if atom.GetIsAromatic() else 0.0]
    charge = [float(atom.GetFormalCharge())]
    hs_enc = one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    return np.array(at_num_enc + degree_enc + hybrid_enc + arom + charge + hs_enc, dtype=np.float32)

def get_bond_features(bond):
    bt = bond.GetBondType()
    bt_enc = one_hot_encoding(bt, [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ])
    conj = [1.0 if bond.GetIsConjugated() else 0.0]
    ring = [1.0 if bond.IsInRing() else 0.0]
    return np.array(bt_enc + conj + ring, dtype=np.float32)

def predict_gfnn(gfnn_model, X_scaled, smiles_list):
    graphs = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        node_feats = []
        for atom in mol.GetAtoms():
            node_feats.append(get_atom_features(atom))
        node_feats = np.array(node_feats, dtype=np.float32)
        
        edge_index = []
        edge_feats = []
        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            bf = get_bond_features(bond)
            edge_index.append([u, v])
            edge_feats.append(bf)
            edge_index.append([v, u])
            edge_feats.append(bf)
            
        if len(edge_index) == 0:
            edge_index = [[0, 0]]
            edge_feats = [np.zeros(6, dtype=np.float32)]
            
        edge_index = np.array(edge_index, dtype=np.int64).T
        edge_feats = np.array(edge_feats, dtype=np.float32)
        
        graphs.append({
            'x': torch.tensor(node_feats, dtype=torch.float32),
            'edge_index': torch.tensor(edge_index, dtype=torch.long),
            'edge_attr': torch.tensor(edge_feats, dtype=torch.float32)
        })
        
    x_list = [g['x'] for g in graphs]
    edge_index_list = [g['edge_index'] for g in graphs]
    edge_attr_list = [g['edge_attr'] for g in graphs]
    
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
    
    aux_batch = torch.tensor(X_scaled, dtype=torch.float32)
    
    gfnn_model.eval()
    with torch.no_grad():
        preds = gfnn_model(
            x_batch,
            edge_index_batch,
            edge_attr_batch,
            batch_idx_batch,
            aux_batch
        )
    return preds.numpy()

# ==============================================================================
# AUDITING AND LOADING MODELS (Step 1)
# ==============================================================================

@st.cache_resource
def load_all_serialized_models():
    models_to_check = {
        'Random Forest': 'rf_model.pkl',
        'XGBoost': 'xgb_model.pkl',
        'FNN': 'fnn_model.pkl',
        'GFNN Preprocessor': 'gfnn_preprocessor.pkl'
    }
    
    loaded_models = {}
    audit_data = []
    
    print("\nModel loaded successfully:")
    for name, filename in models_to_check.items():
        path = os.path.join('saved_models', filename)
        mtime = os.path.getmtime(path)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        loaded_models[name] = obj
        framework_cls = type(obj).__name__
        if hasattr(obj, 'named_steps') and 'model' in obj.named_steps:
            framework_cls = type(obj.named_steps['model']).__name__
        print(name)
        audit_data.append({
            'name': name,
            'type': framework_cls,
            'timestamp': timestamp,
            'path': path
        })
        
    # GFNN model weights
    gfnn_path = os.path.join('saved_models', 'gfnn_model.pt')
    mtime = os.path.getmtime(gfnn_path)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
    gfnn_model = GFNN(node_in_dim=18, edge_in_dim=6, aux_dim=16, hidden_dim=64)
    gfnn_model.load_state_dict(torch.load(gfnn_path, map_location=torch.device('cpu')))
    loaded_models['GFNN'] = gfnn_model
    print("GFNN")
    sys.stdout.flush()
    
    audit_data.append({
        'name': 'GFNN',
        'type': 'GFNN (PyTorch Module)',
        'timestamp': timestamp,
        'path': gfnn_path
    })
    
    return loaded_models, audit_data

try:
    loaded_models, audit_data = load_all_serialized_models()
except Exception as e:
    st.error(f"Error loading saved models: {e}")
    st.stop()

# ==============================================================================
# HEADER
# ==============================================================================

st.markdown("""
    <div class="header-card">
        <h1 style="margin: 0; font-size: 32px; font-weight: 700;">Ether & Ester Combustion Kinetics Platform</h1>
        <p style="margin: 8px 0 0 0; opacity: 0.9; font-size: 16px;">Advanced computational platform using GNNs and tree ensembles to predict kinetics rate constants.</p>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR
# ==============================================================================

st.sidebar.title("Dashboard Controls")
st.sidebar.subheader("Model Configuration")
model_choice = st.sidebar.radio(
    "Choose Prediction Model",
    ["Random Forest", "XGBoost", "FNN", "GFNN"]
)

st.sidebar.markdown("<hr style='border-top: 1px solid #e3edf7; margin: 15px 0;'>", unsafe_allow_html=True)
st.sidebar.subheader("Rate Constant Units")
unit_choice = st.sidebar.selectbox(
    "Select Display Units",
    ["cm³ mol⁻¹ s⁻¹ (Molar)", "cm³ molecule⁻¹ s⁻¹ (Molecular)"],
    index=0,
    help="Select the rate constant unit for display and downloads. Note: models are trained on molar units (cm³/mol/s)."
)

# Render model audit details in sidebar for Step 1
st.sidebar.markdown("<hr style='border-top: 1px solid #e3edf7; margin: 15px 0;'>", unsafe_allow_html=True)
st.sidebar.subheader("Loaded Models Verification")
for m_info in audit_data:
    st.sidebar.markdown(f"""
        <div style="background-color: #f7fafc; padding: 10px; border-radius: 8px; border: 1px solid #e3edf7; margin-bottom: 10px; font-size: 12px;">
            <b>{m_info['name']}</b><br>
            Type: <code>{m_info['type']}</code><br>
            Timestamp: {m_info['timestamp']}
        </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# KPI METRIC CARDS GENERATOR
# ==============================================================================

def render_kpi_cards(dataset_name="None", num_reactions="0", selected_model="None", status="Awaiting Upload"):
    col1, col2, col3, col4 = st.columns(4)
    
    col1.markdown(f"""
        <div class="kpi-card" style="border-left: 5px solid #1f77b4;">
            <div style="font-size: 13px; color: #888888; text-transform: uppercase; font-weight: 600;">Active Dataset</div>
            <div style="font-size: 20px; font-weight: 700; color: #333333; margin-top: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{dataset_name}">{dataset_name}</div>
        </div>
    """, unsafe_allow_html=True)
    
    col2.markdown(f"""
        <div class="kpi-card" style="border-left: 5px solid #2ca02c;">
            <div style="font-size: 13px; color: #888888; text-transform: uppercase; font-weight: 600;">Reactions Processed</div>
            <div style="font-size: 20px; font-weight: 700; color: #333333; margin-top: 5px;">{num_reactions}</div>
        </div>
    """, unsafe_allow_html=True)
    
    col3.markdown(f"""
        <div class="kpi-card" style="border-left: 5px solid #ff7f0e;">
            <div style="font-size: 13px; color: #888888; text-transform: uppercase; font-weight: 600;">Active Model</div>
            <div style="font-size: 20px; font-weight: 700; color: #333333; margin-top: 5px;">{selected_model}</div>
        </div>
    """, unsafe_allow_html=True)
    
    status_color = "#2ca02c" if "Success" in status else ("#ff7f0e" if "Awaiting" in status else "#1f77b4")
    col4.markdown(f"""
        <div class="kpi-card" style="border-left: 5px solid {status_color};">
            <div style="font-size: 13px; color: #888888; text-transform: uppercase; font-weight: 600;">System Status</div>
            <div style="font-size: 20px; font-weight: 700; color: {status_color}; margin-top: 5px;">{status}</div>
        </div>
    """, unsafe_allow_html=True)

# Initialize placeholder metrics
dataset_name = "None"
num_reactions = "0"
status_text = "Awaiting Upload"

# ==============================================================================
# MAIN CONTAINER (FILE UPLOAD & INTERACTION)
# ==============================================================================

st.markdown(f"""
    <div style="background-color: #ffffff; padding: 22px; border-radius: 16px; box-shadow: 0 10px 25px rgba(31, 119, 180, 0.03); border: 1px solid #e3edf7; margin-bottom: 25px;">
        <h4 style="color: #1f77b4; margin-top: 0; margin-bottom: 8px; font-weight: 600;">Active Architecture: {model_choice}</h4>
        <p style="color: #666666; font-size: 14.5px; line-height: 1.6; margin: 0;">
            {"The <b>Random Forest Regressor</b> utilizes an ensemble of 100 decision trees to perform predictions. It handles high-dimensional chemical feature spaces well and enables feature importance and SHAP analysis." if model_choice == "Random Forest" else ""}
            {"The <b>XGBoost Regressor</b> uses gradient boosted decision trees. It is highly optimized, fast, and builds trees sequentially to minimize residual errors." if model_choice == "XGBoost" else ""}
            {"The <b>Feedforward Neural Network (FNN)</b> is a Multi-Layer Perceptron (MLP) trained to map descriptors to rate constants. It models non-linear kinetics relationships." if model_choice == "FNN" else ""}
            {"The <b>Graph Feedforward Neural Network (GFNN)</b> utilizes three custom graph message-passing layers to dynamically learn representation directly from molecular atom-bond graph structures, achieving superior performance ($R^2 = 0.9242$)." if model_choice == "GFNN" else ""}
        </p>
    </div>
""", unsafe_allow_html=True)

st.subheader("1. Data Upload")
uploaded_file = st.file_uploader("Upload new kinetics dataset (.xlsx or .csv)", type=["xlsx", "csv"])

if uploaded_file is not None:
    dataset_name = uploaded_file.name
    status_text = "Data Loaded"
    
    # Read file
    try:
        if uploaded_file.name.endswith('.csv'):
            df_in = pd.read_csv(uploaded_file)
        else:
            df_in = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
        
    total_uploaded_rows = len(df_in)
    
    # STEP 2 : VERIFY INPUT DATA
    required_columns = ['reactant_SMILES', 'reaction_class', 'temperature_K']
    missing_columns = [col for col in required_columns if col not in df_in.columns]
    if missing_columns:
        status_text = "Validation Error"
        render_kpi_cards(dataset_name, "0", model_choice, status_text)
        st.error(f"Required columns missing from file: {', '.join(missing_columns)}")
        st.info("Uploaded files must contain exactly these column names (case-sensitive): reactant_SMILES, reaction_class, temperature_K.")
        st.stop()
        
    # Check duplicate Reaction IDs
    has_reaction_ids = 'reaction_id' in df_in.columns
    if has_reaction_ids:
        if df_in['reaction_id'].duplicated().any():
            status_text = "Duplicate ID Error"
            render_kpi_cards(dataset_name, "0", model_choice, status_text)
            dups = df_in['reaction_id'][df_in['reaction_id'].duplicated()].unique()
            st.error(f"Duplicate Reaction IDs found: {', '.join(map(str, dups))}")
            st.stop()
            
    # Audit row-by-row
    valid_rows = []
    skipped_details = []
    
    for idx, row in df_in.iterrows():
        row_num = idx + 1
        
        # 1. SMILES check
        smiles_val = str(row['reactant_SMILES']).strip()
        mol = Chem.MolFromSmiles(smiles_val)
        if mol is None:
            skipped_details.append(f"Row {row_num}: Invalid SMILES structure '{smiles_val}'")
            continue
            
        # 2. Descriptors check
        try:
            d = extract_descriptors(smiles_val)
            if d is None:
                skipped_details.append(f"Row {row_num}: Failed descriptor extraction for SMILES '{smiles_val}'")
                continue
        except Exception as e:
            skipped_details.append(f"Row {row_num}: Descriptor error: {e}")
            continue
            
        # 3. Temperature check
        try:
            temp_val = float(row['temperature_K'])
            if np.isnan(temp_val) or np.isinf(temp_val):
                skipped_details.append(f"Row {row_num}: temperature_K must be a valid number")
                continue
        except Exception:
            skipped_details.append(f"Row {row_num}: Invalid temperature_K '{row['temperature_K']}'")
            continue
            
        # 4. Reaction class check
        raw_class_val = str(row['reaction_class']).strip()
        class_val = normalize_reaction_class(raw_class_val)
        if class_val is None:
            skipped_details.append(f"Row {row_num}: reaction_class '{raw_class_val}' not recognized as a valid kinetics reaction class")
            continue
            
        # Optional experimental value check
        exp_val = None
        if 'experimental_log10_k' in df_in.columns:
            try:
                val = float(row['experimental_log10_k'])
                if not np.isnan(val) and not np.isinf(val):
                    exp_val = val
            except Exception:
                pass
                
        id_val = str(row['reaction_id']).strip() if has_reaction_ids else f"Reaction_{row_num}"
        cyclic_val = 1.0 if mol.GetRingInfo().NumRings() > 0 else 0.0
        
        valid_rows.append({
            'reaction_id': id_val,
            'reactant_SMILES': smiles_val,
            'reaction_class': class_val,
            'temperature_K': temp_val,
            'experimental_log10_k': exp_val,
            'cyclic_or_acyclic': cyclic_val,
            'descriptors': d,
            'original_row': row.to_dict()
        })
        
    skipped_count = len(skipped_details)
    valid_count = len(valid_rows)
    
    if skipped_count > 0:
        with st.expander("⚠️ Skipped Rows / Warnings Details", expanded=False):
            for err_msg in skipped_details:
                st.warning(err_msg)
                
    if valid_count == 0:
        status_text = "Parsing Failure"
        render_kpi_cards(dataset_name, "0", model_choice, status_text)
        st.error("No valid rows remaining in the uploaded file.")
        st.stop()
        
    num_reactions = str(valid_count)
    
    # ==============================================================================
    # STEP 3 & 4 : DESCRIPTOR GENERATION AND FEATURE ORDERING
    # ==============================================================================
    
    rows_to_predict = []
    for r in valid_rows:
        d = r['descriptors']
        temp_val = r['temperature_K']
        cyclic_val = r['cyclic_or_acyclic']
        
        # Build reaction class one-hot columns
        one_hots = {
            'reaction_class_H-abstraction': 0.0,
            'reaction_class_H-abstraction / overall OH reaction': 0.0,
            'reaction_class_O2 addition': 0.0,
            'reaction_class_Unimolecular decomposition': 0.0,
            'reaction_class_beta scission': 0.0
        }
        cls_col = f"reaction_class_{r['reaction_class'].strip()}"
        if cls_col in one_hots:
            one_hots[cls_col] = 1.0
            
        feature_dict = {
            'MolWt': d['MolWt'],
            'MolLogP': d['MolLogP'],
            'TPSA': d['TPSA'],
            'NumHAcceptors': d['NumHAcceptors'],
            'NumHDonors': d['NumHDonors'],
            'NumRotatableBonds': d['NumRotatableBonds'],
            'RingCount': d['RingCount'],
            'FractionCSP3': d['FractionCSP3'],
            'BalabanJ': d['BalabanJ'],
            'invT_1000_per_K': 1000.0 / temp_val,
            'cyclic_or_acyclic': cyclic_val
        }
        feature_dict.update(one_hots)
        rows_to_predict.append(feature_dict)
        
    X_new = pd.DataFrame(rows_to_predict)
    
    # STEP 4: Ensure all columns are present and reordered strictly to feature_cols
    for col in feature_cols:
        if col not in X_new.columns:
            X_new[col] = 0.0
    X_new = X_new[feature_cols]
    
    feature_order_match = list(X_new.columns) == feature_cols
    
    # Load selected model and predict with progress bar animation
    preds = None
    status_text = "Predicting..."
    render_kpi_cards(dataset_name, num_reactions, model_choice, status_text)
    
    # Animated progress bar
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.005)
        progress_bar.progress(percent_complete + 1)
        
    # Get active model metadata timestamp
    model_timestamp = "Unknown"
    for m_info in audit_data:
        if m_info['name'] == model_choice:
            model_timestamp = m_info['timestamp']
            break
            
    # Perform prediction (Step 5 - calling only transform())
    try:
        if model_choice == "Random Forest":
            pipe = loaded_models['Random Forest']
            preds = pipe.predict(X_new)
            
        elif model_choice == "XGBoost":
            pipe = loaded_models['XGBoost']
            preds = pipe.predict(X_new)
            
        elif model_choice == "FNN":
            pipe = loaded_models['FNN']
            preds = pipe.predict(X_new)
            
        elif model_choice == "GFNN":
            preprocessor_gfnn = loaded_models['GFNN Preprocessor']
            gfnn_model = loaded_models['GFNN']
            
            # Preprocess only calling transform()
            X_scaled = preprocessor_gfnn.transform(X_new)
            smiles_list = [r['reactant_SMILES'] for r in valid_rows]
            
            # Predict
            preds = predict_gfnn(gfnn_model, X_scaled, smiles_list)
            
        status_text = "Success"
        render_kpi_cards(dataset_name, num_reactions, model_choice, status_text)
        
    except Exception as e:
        status_text = "Prediction Error"
        render_kpi_cards(dataset_name, num_reactions, model_choice, status_text)
        st.error(f"Error executing prediction: {e}")
        st.stop()
        
    # ==============================================================================
    # OUTPUTS DISPLAY & DOWNLOAD (Step 6)
    # ==============================================================================
    
    id_list = [r['reaction_id'] for r in valid_rows]
    exp_list = [r['experimental_log10_k'] for r in valid_rows]
    has_exp_data = any(v is not None for v in exp_list)
    
    # Avogadro's number offset (log10(6.02214076e23) = 23.77975)
    AVOGADRO_LOG10 = 23.77975
    
    # Auto-align units if experimental values have negative values (molecular format)
    auto_detected_molecular = False
    if has_exp_data:
        non_null_exp = [v for v in exp_list if v is not None]
        if len(non_null_exp) > 0 and np.mean(non_null_exp) < 0:
            auto_detected_molecular = True
            
    if auto_detected_molecular:
        active_unit = "cm³ molecule⁻¹ s⁻¹"
        st.info("💡 **Automatic Unit Alignment**: The uploaded experimental rate constants have negative log10 values (indicating molecular units, $\\text{cm}^3\\text{ molecule}^{-1}\\text{ s}^{-1}$). Predictions have been automatically converted to molecular units to match your experimental data.")
    else:
        if "Molecular" in unit_choice:
            active_unit = "cm³ molecule⁻¹ s⁻¹"
        else:
            active_unit = "cm³ mol⁻¹ s⁻¹"
            
    # Apply conversions
    if active_unit == "cm³ molecule⁻¹ s⁻¹":
        display_preds = preds - AVOGADRO_LOG10
        display_k = 10**display_preds
        if has_exp_data and not auto_detected_molecular:
            display_exp = [v - AVOGADRO_LOG10 if v is not None else None for v in exp_list]
        else:
            display_exp = exp_list
        unit_label = "cm³ molecule⁻¹ s⁻¹"
        unit_label_plain = "cm3 molecule-1 s-1"
        log_unit_label = "log10(k / (cm³ molecule⁻¹ s⁻¹))"
        log_unit_label_plain = "log10(k / (cm3 molecule-1 s-1))"
    else:
        display_preds = preds
        display_k = 10**preds
        if has_exp_data and auto_detected_molecular:
            display_exp = [v + AVOGADRO_LOG10 if v is not None else None for v in exp_list]
        else:
            display_exp = exp_list
        unit_label = "cm³ mol⁻¹ s⁻¹"
        unit_label_plain = "cm3 mol-1 s-1"
        log_unit_label = "log10(k / (cm³ mol⁻¹ s⁻¹))"
        log_unit_label_plain = "log10(k / (cm3 mol-1 s-1))"
        
    # Reconstruct all columns from the uploaded dataset
    original_rows_list = [r['original_row'] for r in valid_rows]
    df_out = pd.DataFrame(original_rows_list)
    
    # Append predictions beside the original columns
    df_out[f'Predicted_log10_k ({unit_label_plain})'] = display_preds
    df_out[f'Predicted_k ({unit_label_plain})'] = display_k
    
    if has_exp_data:
        df_out[f'Experimental_log10_k ({unit_label_plain})'] = display_exp
        df_out[f'Experimental_k ({unit_label_plain})'] = [10**v if v is not None else None for v in display_exp]
        df_out['Error'] = df_out[f'Predicted_log10_k ({unit_label_plain})'] - df_out[f'Experimental_log10_k ({unit_label_plain})']
        df_out['Absolute_Error'] = df_out['Error'].abs()
        
    # Save prediction_results.xlsx
    df_out.to_excel('prediction_results.xlsx', index=False)
    
    # ==============================================================================
    # METRICS & STATS (Step 7 & 8)
    # ==============================================================================
    
    st.markdown('<hr style="border-top: 1px solid #e3edf7; margin: 30px 0;">', unsafe_allow_html=True)
    st.subheader("2. Model Visualizations & Metrics")
    
    r2, rmse, mae = None, None, None
    max_abs_err, med_abs_err, mean_abs_err = None, None, None
    top_10_errors = None
    
    if has_exp_data:
        y_true = np.array(display_exp, dtype=float)
        y_pred = np.array(display_preds, dtype=float)
        
        # Clean NaNs
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        num_valid_metrics = len(y_true_clean)
        
        if num_valid_metrics > 0:
            r2 = r2_score(y_true_clean, y_pred_clean)
            rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            
            # Error Stats
            errors = np.abs(y_true_clean - y_pred_clean)
            max_abs_err = np.max(errors)
            med_abs_err = np.median(errors)
            mean_abs_err = np.mean(errors)
            
            # Top 10 largest errors
            df_err = pd.DataFrame({
                'Reaction_ID': np.array(id_list)[valid_mask],
                'Experimental': y_true_clean,
                'Predicted': y_pred_clean,
                'Absolute_Error': errors
            }).sort_values('Absolute_Error', ascending=False)
            top_10_errors = df_err.head(10)
            
            # Display metrics cards
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.markdown(f"""
                    <div class="kpi-card" style="border-left: 5px solid #1f77b4; text-align: center;">
                        <span style="font-size: 13px; color: #888888; font-weight: 600;">R² Score</span>
                        <div style="font-size: 28px; font-weight: bold; color: #1f77b4; margin-top: 5px;">{r2:.4f}</div>
                    </div>
                """, unsafe_allow_html=True)
            with metric_col2:
                st.markdown(f"""
                    <div class="kpi-card" style="border-left: 5px solid #2ca02c; text-align: center;">
                        <span style="font-size: 13px; color: #888888; font-weight: 600;">RMSE</span>
                        <div style="font-size: 28px; font-weight: bold; color: #2ca02c; margin-top: 5px;">{rmse:.4f}</div>
                    </div>
                """, unsafe_allow_html=True)
            with metric_col3:
                st.markdown(f"""
                    <div class="kpi-card" style="border-left: 5px solid #ff7f0e; text-align: center;">
                        <span style="font-size: 13px; color: #888888; font-weight: 600;">MAE</span>
                        <div style="font-size: 28px; font-weight: bold; color: #ff7f0e; margin-top: 5px;">{mae:.4f}</div>
                    </div>
                """, unsafe_allow_html=True)
                
            # Parity Plot & Error Statistics
            vis_col1, vis_col2 = st.columns(2)
            
            with vis_col1:
                # Plotly Parity Plot
                min_val = min(y_true_clean.min(), y_pred_clean.min()) - 0.5
                max_val = max(y_true_clean.max(), y_pred_clean.max()) + 0.5
                
                fig_parity = px.scatter(
                    df_err, x='Experimental', y='Predicted',
                    hover_name='Reaction_ID',
                    title=f"Parity Plot - {model_choice}",
                    labels={'Experimental': f'Experimental {log_unit_label}', 'Predicted': f'Predicted {log_unit_label}'},
                    color_discrete_sequence=['#ff7f0e']
                )
                fig_parity.add_shape(
                    type="line", line=dict(dash="dash", color="red", width=2),
                    x0=min_val, y0=min_val, x1=max_val, y1=max_val
                )
                fig_parity.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    height=450,
                    margin=dict(l=30, r=30, t=40, b=30)
                )
                fig_parity.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f2f5f9', range=[min_val, max_val])
                fig_parity.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f2f5f9', range=[min_val, max_val])
                st.plotly_chart(fig_parity, use_container_width=True)
                
            with vis_col2:
                # Display Prediction Error Summary (Step 8)
                st.markdown(f"""
                    <div class="kpi-card" style="border-left: 5px solid #1f77b4; height: 180px; margin-bottom: 25px;">
                        <h4 style="color: #1f77b4; margin-top: 0; font-weight: 600; font-size: 16px; margin-bottom: 12px;">Prediction Error Summary</h4>
                        <div style="font-size: 14.5px; line-height: 1.8;">
                            <b>Maximum Absolute Error:</b> {max_abs_err:.6f}<br>
                            <b>Median Absolute Error:</b> {med_abs_err:.6f}<br>
                            <b>Mean Absolute Error:</b> {mean_abs_err:.6f}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                    <div style="background-color: #ffffff; padding: 20px; border-radius: 16px; border: 1px solid #e3edf7; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.01);">
                        <h5 style="color: #1f77b4; margin-top: 0; font-weight: 600; font-size: 15px; margin-bottom: 12px;">Top 10 Largest Prediction Discrepancies</h5>
                """, unsafe_allow_html=True)
                st.dataframe(top_10_errors.style.format({
                    'Experimental': '{:.4f}',
                    'Predicted': '{:.4f}',
                    'Absolute_Error': '{:.4f}'
                }), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No valid experimental rate constants found in the uploaded file.")
    else:
        # Requirement 1: Display exact Prediction Mode callout
        st.markdown("""
            <div class="kpi-card" style="border-left: 5px solid #ff7f0e; padding: 25px; margin-bottom: 30px;">
                <h4 style="color: #ff7f0e; margin-top: 0; font-weight: 700; font-size: 20px; letter-spacing: 0.5px;">Prediction Mode</h4>
                <div style="font-size: 15px; color: #444444; font-weight: 500; line-height: 1.8; margin-top: 15px;">
                    Experimental values were not found.<br><br>
                    The application will generate predictions only.<br><br>
                    To enable evaluation metrics, upload a dataset containing the column:<br><br>
                    <code style="background-color: #f7fafc; padding: 6px 12px; border-radius: 6px; border: 1px solid #e3edf7; color: #ff7f0e; font-size: 14.5px; font-weight: bold; font-family: monospace;">experimental_log10_k</code>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    # Step 9: Generate prediction_debug_report.txt (Developer-free naming)
    top_10_df_for_report = top_10_errors if top_10_errors is not None else pd.DataFrame()
    
    lines = []
    lines.append("==========================================================")
    lines.append("COMBUSTION KINETICS PREDICTION PLATFORM - REPORT")
    lines.append("==========================================================")
    lines.append(f"Uploaded File: {uploaded_file.name}")
    lines.append(f"Total Uploaded Reactions: {total_uploaded_rows}")
    lines.append(f"Valid Reactions Processed: {valid_count}")
    lines.append(f"Excluded Reactions: {skipped_count}")
    
    if skipped_count > 0:
        lines.append("\nExcluded Reactions Details:")
        for detail in skipped_details:
            lines.append(f"  - {detail}")
            
    lines.append("\n==========================================================")
    lines.append("FEATURE & PREPROCESSING VALIDATION")
    lines.append("==========================================================")
    lines.append(f"Input Data Alignment with Model Schema: Verified")
    lines.append("Model Input Variables:")
    for f in feature_cols:
        lines.append(f"  - {f}")
    lines.append("Data Standardization: Applied pre-fit scaling parameters")
    
    lines.append("\n==========================================================")
    lines.append("MODEL SPECIFICATION")
    lines.append("==========================================================")
    lines.append(f"Selected Model: {model_choice}")
    lines.append(f"Model Generation Timestamp: {model_timestamp}")
    
    lines.append("\n==========================================================")
    lines.append("PREDICTION STATISTICS")
    lines.append("==========================================================")
    lines.append(f"Rate Constant Units: {unit_label_plain}")
    lines.append(f"Mean Predicted rate constant log10(k): {np.mean(display_preds):.6f}")
    lines.append(f"Min Predicted rate constant log10(k): {np.min(display_preds):.6f}")
    lines.append(f"Max Predicted rate constant log10(k): {np.max(display_preds):.6f}")
    
    if has_exp_data and r2 is not None:
        lines.append("\n==========================================================")
        lines.append("EVALUATION METRICS & ERROR STATS")
        lines.append("==========================================================")
        lines.append(f"Valid Reactions Used for Evaluation: {num_valid_metrics}")
        lines.append(f"R² Coefficient:    {r2:.6f}")
        lines.append(f"RMSE ({unit_label_plain}):  {rmse:.6f}")
        lines.append(f"MAE ({unit_label_plain}):   {mae:.6f}")
        lines.append(f"Maximum Absolute Discrepancy: {max_abs_err:.6f}")
        lines.append(f"Median Absolute Discrepancy:  {med_abs_err:.6f}")
        lines.append(f"Mean Absolute Discrepancy:    {mean_abs_err:.6f}")
        
        lines.append("\nTop 10 Largest Discrepancies:")
        for _, err_row in top_10_df_for_report.iterrows():
            lines.append(f"  Reaction ID: {err_row['Reaction_ID']} -> Experimental: {err_row['Experimental']:.6f}, Predicted: {err_row['Predicted']:.6f}, Abs Error: {err_row['Absolute_Error']:.6f}")
            
    debug_report_content = "\n".join(lines)
    with open('prediction_debug_report.txt', 'w', encoding='utf-8') as f:
        f.write(debug_report_content)
        
    print(debug_report_content)
    sys.stdout.flush()
    
    # Model Explainability (Requirement 3)
    st.markdown('<hr style="border-top: 1px solid #e3edf7; margin: 30px 0;">', unsafe_allow_html=True)
    st.subheader("3. Model Explainability")
    
    explain_col1, explain_col2 = st.columns([1, 2])
    
    with explain_col1:
        if model_choice == "Random Forest":
            st.markdown("### 🌳 Random Forest Explainability")
            st.markdown("""
**Available:**
- **✓ Feature Importance**: Quantifies the relative contribution of each descriptor to the model's predictions by measuring Gini impurity reduction.
- **✓ SHAP Summary**: Leverages Shapley additive explanations to reveal both the magnitude and direction of feature effects across the dataset.
- **✓ Local SHAP**: Calculates local attribution values for individual reaction predictions to understand specific feature contributions.
""")
        elif model_choice == "XGBoost":
            st.markdown("### ⚡ XGBoost Explainability")
            st.markdown("""
**Available:**
- **✓ Feature Importance**: Evaluates how frequently each descriptor is used to split data across the sequence of gradient boosted trees.
- **✓ SHAP Summary**: Utilizes additive attribution methods to measure the global impact and positive/negative influences of chemical features.
- **✓ Local SHAP**: Determines exact individual feature attribution for specific reaction coordinates.
""")
        elif model_choice == "Linear Regression":
            st.markdown("### 📈 Linear Regression Explainability")
            st.markdown("""
**Available:**
- **✓ Coefficient Importance**: Shows the direct weight multiplier assigned to each normalized input variable.
  - *Positive Coefficients*: An increase in the feature value directly increases the predicted rate constant log10(k).
  - *Negative Coefficients*: An increase in the feature value directly decreases the predicted rate constant log10(k).
""")
        elif model_choice == "FNN":
            st.markdown("### 🧠 Feed Forward Neural Network")
            st.markdown("""
**Prediction Model**

*Explainability is not currently implemented for this multi-layer perceptron neural network.*
""")
        elif model_choice == "GFNN":
            st.markdown("### 🧠 Graph Feed Forward Neural Network")
            st.markdown("""
**Prediction Model**

*Explainability is not currently implemented for this graph neural network architecture.*
""")
        
    with explain_col2:
        if model_choice in ["Random Forest", "XGBoost"]:
            try:
                if model_choice == "Random Forest":
                    model_pipe = loaded_models['Random Forest']
                    importances = model_pipe.named_steps['model'].feature_importances_
                    title_text = "Random Forest Feature Importance"
                    bar_color = '#1f77b4'
                else:
                    model_pipe = loaded_models['XGBoost']
                    importances = model_pipe.named_steps['model'].feature_importances_
                    title_text = "XGBoost Feature Importance"
                    bar_color = '#2ca02c'
                
                # Sort features
                indices = np.argsort(importances)
                sorted_features = [feature_cols[i] for i in indices]
                sorted_importances = importances[indices]
                
                # Create Matplotlib Figure
                fig_imp, ax = plt.subplots(figsize=(10, 6))
                
                # High-contrast background colors
                fig_imp.patch.set_facecolor('white')
                ax.set_facecolor('white')
                
                # Render horizontal bars
                ax.barh(sorted_features, sorted_importances, color=bar_color, edgecolor='black', height=0.6)
                
                # High-contrast black labels and titles
                ax.set_title(title_text, color='black', fontsize=14, fontweight='bold', pad=15)
                ax.set_xlabel('Relative Importance', color='black', fontsize=12, labelpad=10)
                ax.set_ylabel('Features', color='black', fontsize=12, labelpad=10)
                
                # Explicitly set tick labels in bold black
                ax.tick_params(axis='both', colors='black', labelsize=10)
                
                # Gridlines
                ax.grid(axis='x', linestyle='--', alpha=0.5, color='#cccccc')
                ax.set_axisbelow(True)
                
                # Spines
                for spine in ['top', 'right']:
                    ax.spines[spine].set_visible(False)
                for spine in ['left', 'bottom']:
                    ax.spines[spine].set_color('black')
                    
                plt.tight_layout()
                st.pyplot(fig_imp, dpi=300)
                plt.close()
            except Exception as e:
                st.warning(f"Feature Importance plot unavailable: {e}")
                
            try:
                import shap
                st.markdown("<hr style='border-top: 1px solid #e3edf7; margin: 15px 0;'>", unsafe_allow_html=True)
                with st.spinner("Calculating SHAP values for local explanation..."):
                    if model_choice == "Random Forest":
                        model_pipe = loaded_models['Random Forest']
                    else:
                        model_pipe = loaded_models['XGBoost']
                        
                    preprocessor = model_pipe.named_steps['preprocessor']
                    X_trans = preprocessor.transform(X_new)
                    
                    remainder_cols = [col for col in feature_cols if col not in numeric_cols]
                    trans_feature_names = list(numeric_cols) + list(remainder_cols)
                    df_trans = pd.DataFrame(X_trans, columns=trans_feature_names)
                    
                    explainer = shap.TreeExplainer(model_pipe.named_steps['model'])
                    shap_values = explainer.shap_values(df_trans)
                    
                    fig_shap, ax = plt.subplots(figsize=(10, 5))
                    fig_shap.patch.set_facecolor('white')
                    ax.set_facecolor('white')
                    
                    shap.summary_plot(shap_values, df_trans, show=False)
                    
                    # Force black labels and high contrast for SHAP summary
                    ax.set_title(f"SHAP Beeswarm Summary Plot ({model_choice})", fontsize=12, fontweight='bold', pad=15, color='black')
                    ax.tick_params(axis='both', colors='black', labelsize=10)
                    ax.xaxis.label.set_color('black')
                    ax.yaxis.label.set_color('black')
                    
                    for spine in ['top', 'right', 'bottom', 'left']:
                        ax.spines[spine].set_color('black')
                        
                    plt.tight_layout()
                    st.pyplot(fig_shap, dpi=300)
                    plt.close()
            except ImportError:
                st.info("Install `shap` library to view the SHAP beeswarm explanations.")
            except Exception as e:
                st.warning(f"Could not compute SHAP plot: {e}")
        else:
            st.info("The selected model does not support feature importance or SHAP explanations in this prediction mode.")

    # ==============================================================================
    # OUTPUTS DISPLAY & DOWNLOAD
    # ==============================================================================
    
    st.markdown('<hr style="border-top: 1px solid #e3edf7; margin: 30px 0;">', unsafe_allow_html=True)
    st.subheader("4. Prediction Table & Downloader")
    
    # Format float values for display in scientific notation
    cols_config = {
        f"Predicted_k ({unit_label_plain})": st.column_config.NumberColumn(
            format="%.4e"
        )
    }
    if has_exp_data:
        cols_config[f"Experimental_k ({unit_label_plain})"] = st.column_config.NumberColumn(
            format="%.4e"
        )
        
    st.dataframe(df_out, use_container_width=True, column_config=cols_config)
    
    # Excel Download
    towrite = io.BytesIO()
    df_out.to_excel(towrite, index=False, header=True)
    towrite.seek(0)
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.download_button(
            label="📥 Download prediction_results.xlsx",
            data=towrite,
            file_name="prediction_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    with col_d2:
        st.download_button(
            label="📄 Download prediction_debug_report.txt",
            data=debug_report_content,
            file_name="prediction_debug_report.txt",
            mime="text/plain"
        )

else:
    # Awaiting Upload View
    render_kpi_cards(dataset_name, num_reactions, model_choice, status_text)
    st.info("Upload a dataset file above to start predictions.")

# ==============================================================================
# FOOTER
# ==============================================================================

st.markdown("""
    <hr style="border-top: 1px solid #e3edf7; margin-top: 50px; margin-bottom: 20px;">
    <p style="text-align: center; color: #888888; font-size: 14px; font-weight: 500; letter-spacing: 0.5px;">
        Combustion Kinetics Prediction Platform &copy; 2026
    </p>
""", unsafe_allow_html=True)
