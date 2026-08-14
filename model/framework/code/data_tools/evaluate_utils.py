import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors

from torch_geometric.loader import DataLoader
from data_tools.pyg_chemprop_utils import smiles2data, RevIndexedDataset


def get_eval_dataset(df):
    data_list = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        smi = row["Smiles"]
        data = smiles2data(smi, explicit_h=False)
        data.smiles = smi

        data_list.append(data)
    return RevIndexedDataset(data_list)


def get_eval_dataloaders(dataset, batch_size, feature_scaler, target_scaler=None):
    # Feature standardization
    dataset = feature_scaler.transform(dataset)

    if target_scaler is not None:
        # Target standardization
        dataset = target_scaler.transform(dataset)

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("Evaluation dataset size:", len(dataset))
    return test_loader


@torch.no_grad()
def get_eval_representations(model, loader, device, smile=False):
    model.eval()
    smiles_list = []
    embeddings = []

    for data in tqdm(loader):
        data = data.to(device)

        # Forward to get representations
        rep = model.representation(data)  # shape: (batch_size, dim)
        embeddings.append(rep.cpu().numpy())

        # Assumes `data.smiles` is a list of SMILES strings of length batch_size
        smiles_batch = data.smiles if isinstance(data.smiles, list) else list(data.smiles)
        smiles_list.extend(smiles_batch)

    # Stack all embeddings and labels
    embeddings = np.concatenate(embeddings)  # shape: (N, emb_dim)
    df_embed = pd.DataFrame(embeddings, columns=[f'ft_{i}' for i in range(embeddings.shape[1])])

    if smile:
        df_embed.insert(0, "Smiles", smiles_list)  # insert at column 0

    return df_embed


def compute_descriptors_for_smiles(smiles_list, desc_names):
    desc_func_map = {name: fn for name, fn in Descriptors._descList}
    rows = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        row = {"Smiles": smi}

        if mol is None:
            for d in desc_names:
                row[d] = np.nan
            rows.append(row)
            continue

        for d in desc_names:
            fn = desc_func_map.get(d)
            if fn is None:
                row[d] = np.nan
                continue
            try:
                val = fn(mol)
                row[d] = float(val) if val is not None else np.nan
            except Exception:
                row[d] = np.nan

        rows.append(row)

    return pd.DataFrame(rows, columns=["Smiles"] + list(desc_names))
