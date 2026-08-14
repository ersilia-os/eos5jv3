"""MycoPermeNet-v2 inference.

MycoPermeNet-v2 predicts the standardized residual of mycomembrane permeation in
Mycobacterium tuberculosis from a SMILES string. Unlike MycoPermeNet-v1 (a single
Chemprop D-MPNN readout), v2 is a two-stage fusion model:

    1. A PyG-reimplemented directed MPNN encoder (best_GNN_v2.pt) maps the
       molecular graph to a 300-dim learned embedding.
    2. RDKit physicochemical descriptors are computed, MinMax-normalized with a
       saved scaler (descriptors_minmax_scaler.pkl) and concatenated ("fused")
       to the graph embedding.
    3. A feature scaler (mlp_feature_scaler_v2.pt) standardizes the graph node/
       edge features; a small MLP (mlp_v2.pt, hidden 128-64-16) regresses the
       fused representation; the target scaler (mlp_target_scaler_v2.pt) maps the
       prediction back to the original residual scale.

The v2 weights were trained with the "Fusion Noisy Student Self-Distillation"
strategy that exploits unlabeled data (Zhang et al., J. Chem. Inf. Model. 2026,
66, 2985-2996; https://doi.org/10.1021/acs.jcim.5c02435).

Output semantics are IDENTICAL to v1: the predicted standardized residual is
inversely proportional to permeation, i.e. LOWER values correspond to HIGHER
mycomembrane permeability. Training-set labels range from about -3.1 (most
permeable) to 1.58 (least permeable) with a mean near 0 (Zhang et al. 2026).

Checkpoint provenance
---------------------
Source: https://github.com/SAGE-Lab-UMass/MycoPermeNet-v2-pub (best_MPN/*_v2.*),
the released best MycoPermeNet-v2 artifacts (MIT license). The inference path
mirrors the repository's own script/evaluate.py, restricted to the v2 model and
made label-free so it runs on arbitrary SMILES.
"""

import os
import copy

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from data_tools.evaluate_utils import (
    get_eval_dataset,
    get_eval_dataloaders,
    get_eval_representations,
    compute_descriptors_for_smiles,
)
from models.chemprop import DMPNNEncoder
from models.mlp import MLPRegressor

# The saved scalers/feature-column lists are pickled Python objects, so torch's
# default weights_only=True (torch >= 2.4) would refuse to load them.
_orig_torch_load = torch.load


def _torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load

import joblib  # noqa: E402  (after torch.load shim; unaffected but grouped here)
import pickle  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
CK = os.path.abspath(os.path.join(ROOT, "..", "..", "checkpoints"))
DEVICE = "cpu"


def _ck(name):
    return os.path.join(CK, name)


# Load once at import time (single, small CPU model).
_feature_scaler = torch.load(_ck("mlp_feature_scaler_v2.pt"), map_location="cpu")
_target_scaler = torch.load(_ck("mlp_target_scaler_v2.pt"), map_location="cpu")
_feature_cols = joblib.load(_ck("mlp_feature_cols_v2.pkl"))

with open(_ck("descriptors_minmax_scaler.pkl"), "rb") as _f:
    _desc_pack = pickle.load(_f)
_desc_cols = _desc_pack["columns"]
_desc_scaler = _desc_pack["scaler"]

_gnn = DMPNNEncoder(
    hidden_size=300, node_fdim=133, edge_fdim=14, depth=3, dropout=0
).to(DEVICE)
_gnn.load_state_dict(torch.load(_ck("best_GNN_v2.pt"), map_location=DEVICE))
_gnn.eval()

_mlp = MLPRegressor(input_dim=len(_feature_cols), hidden_layer_sizes=(128, 64, 16)).to(DEVICE)
_mlp.load_state_dict(torch.load(_ck("mlp_v2.pt"), map_location=DEVICE))
_mlp.eval()


def _predict_valid(smiles_valid):
    """Run the full v2 pipeline on a list of RDKit-parseable SMILES."""
    df = pd.DataFrame({"Smiles": list(smiles_valid)})

    dataset = get_eval_dataset(df)
    loader = get_eval_dataloaders(
        copy.deepcopy(dataset),
        batch_size=64,
        feature_scaler=_feature_scaler,
        target_scaler=None,
    )
    # smile=True keeps the SMILES column aligned with the embeddings.
    X = get_eval_representations(_gnn, loader, DEVICE, smile=True)

    # Fuse MinMax-normalized RDKit descriptors onto the graph embeddings.
    smiles_list = X["Smiles"].tolist()
    df_desc = compute_descriptors_for_smiles(smiles_list, _desc_cols)
    desc_raw = df_desc[_desc_cols].to_numpy(dtype=np.float64)

    # Match evaluate.py: fill non-finite descriptor cells with the per-feature
    # training minimum before scaling, for consistent normalization.
    data_min = _desc_scaler.data_min_.astype(np.float64)
    nan_mask = ~np.isfinite(desc_raw)
    if nan_mask.any():
        desc_raw[nan_mask] = np.take(data_min, np.where(nan_mask)[1])
    desc_norm = _desc_scaler.transform(desc_raw)
    df_desc_norm = pd.DataFrame(desc_norm, columns=_desc_cols)

    # NOTE (applicability domain): this wrapper faithfully reproduces the
    # released model (matching the source script/evaluate.py). The model was
    # trained on small azide-tagged compounds (MW ~82-570, median ~173; up to
    # ~50 heavy atoms) and its outputs have a dynamic range of about -3.1 to
    # +1.6. For compounds OUTSIDE this training domain -- notably larger
    # molecules (higher MW / atom count) -- predictions may fall outside this
    # range and should be treated as unreliable extrapolations. Restrict inputs
    # to the training domain (or flag out-of-range outputs) before interpreting
    # scores.

    X = pd.concat(
        [X.reset_index(drop=True), df_desc_norm.reset_index(drop=True)], axis=1
    )
    X = X[_feature_cols]

    y = _mlp.predict(X)
    y = _target_scaler.inverse_transform(y)
    return np.asarray(y, dtype=np.float64).reshape(-1)


def _is_valid(smi):
    """A SMILES is usable only if it parses AND yields a molecular graph with
    at least one bond. RDKit parses "" and single-atom inputs into bond-less
    molecules that would crash the directed-MPNN graph construction (empty
    edge_index), so those are treated as invalid and returned as NaN.
    """
    mol = Chem.MolFromSmiles(str(smi))
    return mol is not None and mol.GetNumBonds() > 0


def predict(smiles_list):
    """Return the predicted Mtb mycomembrane permeation residual per SMILES.

    Invalid / unparseable / bond-less SMILES yield np.nan while preserving
    input order.
    """
    smiles_list = list(smiles_list)
    valid_idx = [i for i, s in enumerate(smiles_list) if _is_valid(s)]

    out = np.full(len(smiles_list), np.nan, dtype=np.float64)
    if valid_idx:
        preds = _predict_valid([smiles_list[i] for i in valid_idx])
        for pos, i in enumerate(valid_idx):
            out[i] = preds[pos]
    return out.tolist()
