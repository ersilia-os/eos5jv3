from typing import List, Union
import numpy as np
import pandas as pd
from random import Random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.data.data import size_repr
from torch_geometric.datasets import MoleculeNet, QM9
from torch_scatter import scatter_sum

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from rdkit.Chem import Descriptors
from collections import defaultdict, OrderedDict
from tqdm import tqdm

import json
import os
import traceback
import random
import re
from .compute_descriptors import calc_descriptors


class FeatureScaler:
    def __init__(self, targets: List[str], replace_nan_token: float = 0.0):
        """
        :param targets: list of attributes in PyG Data to normalize, e.g., ['x', 'edge_attr']
        :param replace_nan_token: token to replace NaNs in final output
        """
        self.targets = targets
        self.replace_nan_token = replace_nan_token
        self.means = {}  # field -> np.ndarray
        self.stds = {}   # field -> np.ndarray

    def fit(self, dataset: List[Data]):
        for field in self.targets:
            all_feats = []
            for data in dataset:
                if hasattr(data, field):
                    feat = getattr(data, field).cpu().numpy().astype(float)  # shape [n, d]
                    all_feats.append(feat)

            if not all_feats:
                raise ValueError(f"No field '{field}' found in dataset.")

            X = np.concatenate(all_feats, axis=0)  # shape [N_total, dim]
            means = np.nanmean(X, axis=0)
            stds = np.nanstd(X, axis=0)

            means = np.where(np.isnan(means), 0.0, means)
            stds = np.where(np.isnan(stds) | (stds == 0), 1.0, stds)

            self.means[field] = means
            self.stds[field] = stds

    def transform(self, dataset: List[Data]) -> List[Data]:
        for data in dataset:
            for field in self.targets:
                if hasattr(data, field):
                    feat = getattr(data, field).cpu().numpy().astype(float)
                    normed = (feat - self.means[field]) / self.stds[field]
                    normed = np.where(np.isnan(normed), self.replace_nan_token, normed)
                    setattr(data, field, torch.tensor(normed, dtype=torch.float))
        return dataset

    def fit_transform(self, dataset: List[Data]) -> List[Data]:
        self.fit(dataset)
        return self.transform(dataset)


class TargetScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, dataset):
        targets = [data.y for data in dataset]
        targets = torch.stack(targets)
        self.mean = targets.mean(dim=0)
        self.std = targets.std(dim=0)
        self.std[self.std == 0] = 1.0

    def transform(self, dataset):
        for data in dataset:
            data.y = (data.y - self.mean) / self.std
        return dataset

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

    def inverse_transform(self, y):
        if isinstance(y, torch.Tensor):
            return y * self.std + self.mean
        elif isinstance(y, np.ndarray):
            std = self.std.cpu().numpy() if isinstance(self.std, torch.Tensor) else self.std
            mean = self.mean.cpu().numpy() if isinstance(self.mean, torch.Tensor) else self.mean
            return y * std + mean
        else:
            raise TypeError("Unsupported input type for inverse_transform")


def mol2data(mol):
    atom_feat = [atom_features(atom) for atom in mol.GetAtoms()]

    edge_attr = []
    edge_index = []

    for bond in mol.GetBonds():
        # eid = bond.GetIdx()
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.extend([(i, j), (j, i)])
        b = bond_features(bond)
        edge_attr.extend([b, b.copy()])

    x = torch.FloatTensor(atom_feat)
    edge_attr = torch.FloatTensor(edge_attr)
    edge_index = torch.LongTensor(edge_index).T

    return Data(x=x, edge_attr=edge_attr, edge_index=edge_index)


def smiles2data(smi, explicit_h=True):
    mol = Chem.MolFromSmiles(smi)
    if explicit_h:
        mol = Chem.AddHs(mol)
    return mol2data(mol)


# from
# https://github.com/chemprop/chemprop/blob/master/chemprop/features/featurization.py

# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    "atomic_num": list(range(MAX_ATOMIC_NUM)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-1, -2, 1, 2, 0],
    "chiral_tag": [0, 1, 2, 3],
    "num_Hs": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

BOND_FDIM = 14


def onek_encoding_unk(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def atom_features(atom):
    features = (
        onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES["atomic_num"])
        + onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES["degree"])
        + onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
        + onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES["chiral_tag"])
        + onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES["num_Hs"])
        + onek_encoding_unk(
            int(atom.GetHybridization()), ATOM_FEATURES["hybridization"]
        )
        + [1 if atom.GetIsAromatic() else 0]
        + [atom.GetMass() * 0.01]
    )  # scaled to about the same range as other features
    return features


def bond_features(bond):
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0),
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


# from
# https://github.com/chemprop/chemprop/blob/master/chemprop/nn_utils.py

def initialize_weights(model: nn.Module) -> None:
    """
    Initializes the weights of a model in place.
    :param model: An PyTorch model.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.
    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where :code:`warmup_steps = warmup_epochs * steps_per_epoch`).
    Then the learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr` over the
    course of the remaining :code:`total_steps - warmup_steps` (where :code:`total_steps =
    total_epochs * steps_per_epoch`). This is roughly based on the learning rate
    schedule from `Attention is All You Need <https://arxiv.org/abs/1706.03762>`_, section 5.3.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: List[Union[float, int]],
        total_epochs: List[int],
        steps_per_epoch: int,
        init_lr: List[float],
        max_lr: List[float],
        final_lr: List[float],
    ):
        """
        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after :code:`warmup_epochs`).
        :param final_lr: The final learning rate (achieved after :code:`total_epochs`).
        """
        assert (
            len(optimizer.param_groups)
            == len(warmup_epochs)
            == len(total_epochs)
            == len(init_lr)
            == len(max_lr)
            == len(final_lr)
        )

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (
            1 / (self.total_steps - self.warmup_steps)
        )

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """
        Gets a list of the current learning rates.
        :return: A list of the current learning rates.
        """
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.
        :param current_step: Optionally specify what step to set the learning rate to.
                             If None, :code:`current_step = self.current_step + 1`.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = (
                    self.init_lr[i] + self.current_step * self.linear_increment[i]
                )
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (
                    self.exponential_gamma[i]
                    ** (self.current_step - self.warmup_steps[i])
                )
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]["lr"] = self.lr[i]


class RevIndexedData(Data):
    def __init__(self, orig):
        super(RevIndexedData, self).__init__()
        if orig:
            for key in orig.keys():
                self[key] = orig[key]
            edge_index = self["edge_index"]
            revedge_index = torch.zeros(edge_index.shape[1]).long()
            for k, (i, j) in enumerate(zip(*edge_index)):
                edge_to_i = edge_index[1] == i
                edge_from_j = edge_index[0] == j
                revedge_index[k] = torch.where(edge_to_i & edge_from_j)[0].item()
            self["revedge_index"] = revedge_index

    def __inc__(self, key, value, *args, **kwargs):
        if key == "revedge_index":
            return self.revedge_index.max().item() + 1
        else:
            return super().__inc__(key, value)

    def __repr__(self):
        cls = str(self.__class__.__name__)
        has_dict = any([isinstance(item, dict) for _, item in self])

        if not has_dict:
            info = [size_repr(key, item) for key, item in self]
            return "{}({})".format(cls, ", ".join(info))
        else:
            info = [size_repr(key, item, indent=2) for key, item in self]
            return "{}(\n{}\n)".format(cls, ",\n".join(info))


class RevIndexedDataset(Dataset):
    def __init__(self, orig):
        super(RevIndexedDataset, self).__init__()
        self.dataset = [RevIndexedData(data) for data in orig]

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


def directed_mp(message, edge_index, revedge_index):
    m = scatter_sum(message, edge_index[1], dim=0)
    m_all = m[edge_index[0]]
    m_rev = message[revedge_index]
    return m_all - m_rev


def aggregate_at_nodes(num_nodes, message, edge_index):
    m = scatter_sum(message, edge_index[1], dim=0, dim_size=num_nodes)
    return m[torch.arange(num_nodes)]


def pad_pe(data, pe_dim=20):
    if not hasattr(data, 'pe') or data.pe is None:
        data.pe = torch.zeros((data.num_nodes, pe_dim), dtype=torch.float)
    elif data.pe.size(1) < pe_dim:
        pad_width = pe_dim - data.pe.size(1)
        data.pe = F.pad(data.pe, (0, pad_width), value=0)
    elif data.pe.size(1) > pe_dim:
        data.pe = data.pe[:, :pe_dim]
    return data


def get_dataset(df, smiles_to_fingerprint=None):
    data_list = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        smi = row["Smiles"]
        data = smiles2data(smi, explicit_h=False)
        data.smiles = smi
        data.y = torch.tensor([[row["MTB Standardized Residuals"]]], dtype=torch.float)

        # # Add Laplacian eigenvector positional encoding
        # k = min(20, data.num_nodes - 1)
        # pe_transform = AddLaplacianEigenvectorPE(k=k, attr_name='pe')
        # data = pe_transform(data)
        # data = pad_pe(data, 20)

        # Add functional group vector or rdkit descriptors
        if smiles_to_fingerprint is not None:
            fingerprint = smiles_to_fingerprint[smi]  # already a 1D array (D,)
            fingerprint = np.array(fingerprint, dtype=np.float32)

            # shape: (1, D) to ensure batch becomes (B, D) not (B,)
            data.fg = torch.from_numpy(fingerprint).unsqueeze(0)

        data_list.append(data)
    return RevIndexedDataset(data_list)


# Data splitting
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    else:
        return None


def scaffold_balanced_split(dataset, val_ratio=0.2, seed=42):
    scaffolds = []
    valid_indices = []

    for i, data in enumerate(dataset):
        s = get_scaffold(data.smiles)
        if s is not None:
            scaffolds.append(s)
            valid_indices.append(i)

    # Filter valid data
    dataset = [dataset[i] for i in valid_indices]
    scaffolds = [scaffolds[i] for i in range(len(valid_indices))]

    # Projection of scaffolds to indices
    scaffold_to_indices = {}
    for idx, scaffold in enumerate(scaffolds):
        scaffold_to_indices.setdefault(scaffold, []).append(idx)

    scaffold_index_sets = list(scaffold_to_indices.values())
    total_sample_count = len(dataset)
    val_size_target = int(val_ratio * total_sample_count)
    train_size_target = total_sample_count - val_size_target

    # Priporitize larger scaffolds for training
    big_sets, small_sets = [], []
    for index_set in scaffold_index_sets:
        if len(index_set) > val_size_target // 2:
            big_sets.append(index_set)
        else:
            small_sets.append(index_set)

    random = Random(seed)
    random.seed(seed)
    random.shuffle(big_sets)
    random.shuffle(small_sets)

    all_index_sets = big_sets + small_sets

    train_indices = []
    val_indices = []

    for index_set in all_index_sets:
        if len(train_indices) + len(index_set) <= train_size_target:
            train_indices.extend(index_set)
        else:
            val_indices.extend(index_set)

    # Map from sample index to scaffold string
    index_to_scaffold = {}
    for scaffold, indices in scaffold_to_indices.items():
        for idx in indices:
            index_to_scaffold[idx] = scaffold

    # Get unique scaffolds in train and val splits
    train_scaffolds = set(index_to_scaffold[idx] for idx in train_indices)
    val_scaffolds = set(index_to_scaffold[idx] for idx in val_indices)

    print(f"Train scaffold: {len(train_scaffolds)} | Val scaffold: {len(val_scaffolds)}")

    return train_indices, val_indices, dataset


def get_dataloaders(train_val_dataset, test_dataset, batch_size=64, seed=42, generator=None):
    train_indices, val_indices, filtered_dataset = scaffold_balanced_split(train_val_dataset, seed=seed)

    with open('scaffold_split_indices1.json', 'w') as f:
        json.dump({'train_idx': train_indices, 'val_idx': val_indices}, f)

    train_dataset = [filtered_dataset[i] for i in train_indices]
    val_dataset = [filtered_dataset[i] for i in val_indices]

    # Feature standardization
    feature_scaler = FeatureScaler(targets=["x", "edge_attr"])
    train_dataset = feature_scaler.fit_transform(train_dataset)
    val_dataset = feature_scaler.transform(val_dataset)
    test_dataset = feature_scaler.transform(test_dataset)

    # Target standardization
    target_scaler = TargetScaler()
    train_dataset = target_scaler.fit_transform(train_dataset)
    val_dataset = target_scaler.transform(val_dataset)
    test_dataset = target_scaler.transform(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)} | Test size: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, target_scaler


def get_emb_dataloaders(train_val_dataset, test_dataset, seed=42):
    train_indices, val_indices, filtered_dataset = scaffold_balanced_split(train_val_dataset, seed=seed)

    with open('scaffold_split_indices2.json', 'w') as f:
        json.dump({'train_idx': train_indices, 'val_idx': val_indices}, f)

    train_dataset = [filtered_dataset[i] for i in train_indices]
    val_dataset = [filtered_dataset[i] for i in val_indices]

    feature_scaler = FeatureScaler(targets=["x", "edge_attr"])
    train_dataset = feature_scaler.fit_transform(train_dataset)
    val_dataset = feature_scaler.transform(val_dataset)
    test_dataset = feature_scaler.transform(test_dataset)

    # Target standardization
    target_scaler = TargetScaler()
    train_dataset = target_scaler.fit_transform(train_dataset)
    val_dataset = target_scaler.transform(val_dataset)
    test_dataset = target_scaler.transform(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    return train_loader, val_loader, test_loader, target_scaler


# def functional_groups_from_smiles(smiles_list):
#     # Remove duplicates while keeping order
#     seen = set()
#     smiles_list_unique = []
#     for smi in smiles_list:
#         if smi not in seen:
#             seen.add(smi)
#             smiles_list_unique.append(smi)

#     # RDKit factory
#     fdef_path = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
#     factory = ChemicalFeatures.BuildFeatureFactory(fdef_path)

#     fg_counter = defaultdict(int)
#     mol_fg_list = []

#     for smi in smiles_list_unique:
#         mol = Chem.MolFromSmiles(smi)
#         if mol is None:
#             mol_fg_list.append([])
#             continue

#         features = factory.GetFeaturesForMol(mol)
#         fg_types = [f.GetType() for f in features]
#         mol_fg_list.append(fg_types)

#         for fg in fg_types:
#             fg_counter[fg] += 1

#     # Get all unique functional group types
#     all_fg_types = sorted(set(fg for fgs in mol_fg_list for fg in fgs))
#     smiles_to_fingerprint = {}

#     for smi, fg_list in zip(smiles_list_unique, mol_fg_list):
#         # onehot_vec = [1 if fg in fg_list else 0 for fg in all_fg_types]
#         onehot_vec = np.array(
#             [1 if fg in fg_list else 0 for fg in all_fg_types],
#             dtype=np.float32
#         )
#         smiles_to_fingerprint[smi] = onehot_vec

#     return smiles_to_fingerprint, all_fg_types, fg_counter


def functional_groups_from_smiles(smiles_list):
    # Remove duplicates while keeping order
    seen = set()
    smiles_list_unique = []
    for smi in smiles_list:
        if smi not in seen:
            seen.add(smi)
            smiles_list_unique.append(smi)

    # Load functional group SMARTS definitions
    fg_smarts = {}
    path = os.path.join(RDConfig.RDDataDir, "Functional_Group_Hierarchy.txt")
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or comments
            if not line or line.startswith("//"):
                continue

            # Split into: Name, SMARTS, Label
            parts = re.split(r'\s{2,}|\t+', line)
            # print("line", line)
            # print("parts",parts)
            if len(parts) < 2:
                continue

            name = parts[0].strip()
            smarts = parts[1].strip()

            # Skip if no SMARTS is provided
            if not smarts:
                continue

            # Parse SMARTS
            try:
                patt = Chem.MolFromSmarts(smarts)
                if patt is not None:
                    fg_smarts[name] = patt
            except Exception:
                continue

    fg_counter = defaultdict(int)
    mol_fg_list = []

    for smi in smiles_list_unique:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol_fg_list.append([])
            continue

        matched_groups = []
        for name, patt in fg_smarts.items():
            if mol.HasSubstructMatch(patt):
                matched_groups.append(name)
                fg_counter[name] += 1

        mol_fg_list.append(matched_groups)

    all_fg_types = sorted(set(fg for fgs in mol_fg_list for fg in fgs))
    smiles_to_fingerprint = {}

    for smi, fg_list in zip(smiles_list_unique, mol_fg_list):
        onehot_vec = np.array(
            [1 if fg in fg_list else 0 for fg in all_fg_types],
            dtype=np.float32
        )
        smiles_to_fingerprint[smi] = onehot_vec

    return smiles_to_fingerprint, all_fg_types, fg_counter


def descriptors_from_smiles(smiles_list, missingVal=None):
    desc_list = Descriptors._descList
    descriptor_names = [name for name, _ in desc_list]

    # Drop duplicates
    seen = set()
    smiles_unique = []
    for smi in smiles_list:
        if smi not in seen:
            seen.add(smi)
            smiles_unique.append(smi)

    descriptor_rows = []
    valid_smiles = []

    for smi in tqdm(smiles_unique, desc="Computing descriptors"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        values = []
        for _, fn in desc_list:
            try:
                val = fn(mol)
            except:
                traceback.print_exc()
                val = missingVal
            values.append(val)

        descriptor_rows.append(values)
        valid_smiles.append(smi)

    desc_df = pd.DataFrame(descriptor_rows, columns=descriptor_names)
    desc_df.insert(0, "Smiles", valid_smiles)

    # Exclude descriptors that are all zero or any NaN
    descriptor_only = desc_df.drop(columns=["Smiles"])
    nonzero_mask = ~(descriptor_only == 0).all(axis=0)
    nonnan_mask = ~descriptor_only.isna().any(axis=0)
    valid_mask = nonzero_mask & nonnan_mask
    valid_descriptor_names = descriptor_only.columns[valid_mask]

    desc_df = desc_df[["Smiles"] + valid_descriptor_names.tolist()]

    # Stardardization
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    desc_values = scaler.fit_transform(desc_df[valid_descriptor_names])
    desc_array = np.array(desc_values, dtype=np.float32)

    # Remove any descriptor columns with NaN/inf
    is_finite_col = np.isfinite(desc_array).all(axis=0)
    final_descriptor_names = valid_descriptor_names[is_finite_col]
    desc_array = desc_array[:, is_finite_col]

    # Reconstruct final DataFrame
    desc_df = pd.DataFrame({
        "Smiles": desc_df["Smiles"].values  # or .tolist()
    })
    descriptor_df = pd.DataFrame(desc_array, columns=final_descriptor_names)
    desc_df = pd.concat([desc_df, descriptor_df], axis=1)

    smiles_to_descriptor = {
        smi: desc_array[i] for i, smi in enumerate(desc_df["Smiles"])
    }

    return smiles_to_descriptor, final_descriptor_names.tolist(), scaler


def descriptors_from_names(smiles_list, descriptor_names, missingVal=None):
    name_to_fn = dict(Descriptors._descList)
    selected_fns = {name: name_to_fn[name] for name in descriptor_names if name in name_to_fn}

    descriptor_rows = []
    valid_smiles = []

    for smi in tqdm(smiles_list, desc="Computing selected descriptors"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        values = []
        for name in descriptor_names:
            fn = selected_fns.get(name, None)
            if fn is None:
                values.append(missingVal)
                continue
            try:
                val = fn(mol)
            except:
                val = missingVal
            values.append(val)
        descriptor_rows.append(values)
        valid_smiles.append(smi)

    desc_df = pd.DataFrame(descriptor_rows, columns=descriptor_names)
    desc_df.insert(0, "Smiles", valid_smiles)

    # Remove rows with all zeros or any NaNs
    descriptor_only = desc_df.drop(columns=["Smiles"])
    nonzero_mask = ~(descriptor_only == 0).all(axis=1)
    nonnan_mask = ~descriptor_only.isna().any(axis=1)
    valid_mask = nonzero_mask & nonnan_mask

    # Filter rows accordingly
    desc_df = desc_df[valid_mask]

    # Now safely get all descriptor column names
    valid_descriptor_names = desc_df.columns.drop("Smiles").tolist()

    # Keep only Smiles + valid descriptor columns
    desc_df = desc_df[["Smiles"] + valid_descriptor_names]
    descriptor_only = desc_df.drop(columns=["Smiles"])

    # Normalization
    scaler = MinMaxScaler()
    desc_values = scaler.fit_transform(descriptor_only)
    desc_array = np.array(desc_values, dtype=np.float32)

    # Remove molecules with NaN or inf in any descriptor
    is_finite_row = np.isfinite(desc_array).all(axis=1)
    desc_array = desc_array[is_finite_row]
    final_smiles = desc_df["Smiles"].values[is_finite_row]

    # Create final mapping
    smiles_to_descriptor = {
        smi: desc_array[i] for i, smi in enumerate(final_smiles)
    }

    return smiles_to_descriptor, valid_descriptor_names, scaler


def mtb_descriptors_from_names(smiles_list, descriptor_names, missingVal=None):
    """
    Compute selected custom descriptors for a list of SMILES strings using calc_descriptors.
    :param smiles_list: List of SMILES strings.
    :param descriptor_names: List of descriptor names to select from calc_descriptors output.
    :param missingVal: Value to use for missing descriptors (not used here since NaNs are filtered).
    :return: (smiles_to_descriptor, valid_descriptor_names, scaler)
    """
    full_desc_df = calc_descriptors(pd.Series(smiles_list))
    full_desc_df.insert(0, "Smiles", smiles_list)

    # Keep descriptors from descriptor_names
    selected_columns = [name for name in descriptor_names if name in full_desc_df.columns]
    desc_df = full_desc_df[["Smiles"] + selected_columns]

    # Remove rows with all zeros or any NaNs
    descriptor_only = desc_df.drop(columns=["Smiles"])
    nonzero_mask = ~(descriptor_only == 0).all(axis=1)
    nonnan_mask = ~descriptor_only.isna().any(axis=1)
    valid_mask = nonzero_mask & nonnan_mask
    desc_df = desc_df[valid_mask]

    valid_descriptor_names = desc_df.columns.drop("Smiles").tolist()

    # Normalization
    descriptor_only = desc_df.drop(columns=["Smiles"])
    scaler = MinMaxScaler()
    desc_values = scaler.fit_transform(descriptor_only)
    desc_array = np.array(desc_values, dtype=np.float32)

    # Remove molecules with NaN or inf in any descriptor
    is_finite_row = np.isfinite(desc_array).all(axis=1)
    desc_array = desc_array[is_finite_row]
    final_smiles = desc_df["Smiles"].values[is_finite_row]

    smiles_to_descriptor = {
        smi: desc_array[i] for i, smi in enumerate(final_smiles)
    }

    return smiles_to_descriptor, valid_descriptor_names, scaler


def get_dataset_from_molnet(data_name, mine=False, fusion=False):
    data_list = []
    if data_name == "permeability":
        df = pd.read_csv("./data/siegrist_clean_mtb1600.csv")
        smiles_list = df["Smiles"].tolist()
        iterable = df.iterrows()
    else:
        molnet_dataset = MoleculeNet(root='./data', name=data_name)
        smiles_list = [d.smiles for d in molnet_dataset]
        iterable = molnet_dataset

    if mine:
        smiles_to_fingerprint, _, fg_counter = functional_groups_from_smiles(smiles_list)
        print("functional group count:\n", len(fg_counter))

    if fusion:
        smiles_to_descriptor, descriptor_names, _ = descriptors_from_smiles(smiles_list)
        print("molecular descriptor dim:", len(descriptor_names))

    for item in tqdm(iterable):
        if data_name == "permeability":
            _, mol_data = item
            smi = mol_data["Smiles"]
            y_val = mol_data["MTB Standardized Residuals"]
        else:
            mol_data = item
            smi = mol_data.smiles
            y_val = mol_data.y.item() if torch.is_tensor(mol_data.y) else float(mol_data.y)

        # data = smiles2data(smi, explicit_h=False)
        try:
            data = smiles2data(smi, explicit_h=False)
            if data.edge_index.size(1) == 0:
                print(f"[Warning] Skipping molecule with no edges: {smi}")
                continue
        except Exception as e:
            print(f"[Error] Failed to process {smi}: {e}")
            continue
        data.smiles = smi
        data.y = torch.tensor([[y_val]], dtype=torch.float)

        # # Add Laplacian PE
        # k = min(20, data.num_nodes - 1)
        # pe_transform = AddLaplacianEigenvectorPE(k=k, attr_name='pe')
        # data = pe_transform(data)
        # data = pad_pe(data, 20)

        # MINE and Fusion
        if mine:
            fingerprint = smiles_to_fingerprint[smi]
            data.fg = torch.from_numpy(fingerprint).unsqueeze(0)

        if fusion:
            descriptor = smiles_to_descriptor[smi]
            data.rdkit = torch.from_numpy(descriptor).unsqueeze(0)  # shape (1, D)

        data_list.append(data)

    if mine and fusion:
        return RevIndexedDataset(data_list), len(fg_counter), descriptor_names
    elif mine:
        return RevIndexedDataset(data_list), len(fg_counter)
    elif fusion:
        return RevIndexedDataset(data_list), descriptor_names

    return RevIndexedDataset(data_list)


def get_dataloaders_from_molnet(dataset, batch_size=64, val_ratio=0.2, seed=42, generator=None):
    # Split into train_val & test
    train_val_indices, test_indices, filtered_dataset = scaffold_balanced_split(dataset, val_ratio=val_ratio, seed=42)
    train_val_dataset = [filtered_dataset[i] for i in train_val_indices]
    test_dataset = [filtered_dataset[i] for i in test_indices]

    train_val_dataset = list(train_val_dataset)
    test_dataset = list(test_dataset)

    # Scaffold split train_val into train/val
    train_indices, val_indices, filtered_dataset = scaffold_balanced_split(train_val_dataset, val_ratio=val_ratio, seed=seed)
    train_dataset = [filtered_dataset[i] for i in train_indices]
    val_dataset = [filtered_dataset[i] for i in val_indices]

    # Feature standardization
    feature_scaler = FeatureScaler(targets=["x", "edge_attr"])
    train_dataset = feature_scaler.fit_transform(train_dataset)
    val_dataset = feature_scaler.transform(val_dataset)
    test_dataset = feature_scaler.transform(test_dataset)

    # Target standardization
    target_scaler = TargetScaler()
    train_dataset = target_scaler.fit_transform(train_dataset)
    val_dataset = target_scaler.transform(val_dataset)
    test_dataset = target_scaler.transform(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Stage 1 Train size: {len(train_dataset)} | Val size: {len(val_dataset)} | Test size: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, target_scaler


def get_emb_dataloaders_from_molnet(dataset, val_ratio=0.2, seed=42):
    # Split into train_val & test
    train_val_indices, test_indices, filtered_dataset = scaffold_balanced_split(dataset, val_ratio=val_ratio, seed=42)
    train_val_dataset = [filtered_dataset[i] for i in train_val_indices]
    test_dataset = [filtered_dataset[i] for i in test_indices]

    train_val_dataset = list(train_val_dataset)
    test_dataset = list(test_dataset)

    # Scaffold split train_val into train/val
    train_indices, val_indices, filtered_dataset = scaffold_balanced_split(train_val_dataset, val_ratio=val_ratio, seed=seed)
    train_dataset = [filtered_dataset[i] for i in train_indices]
    val_dataset = [filtered_dataset[i] for i in val_indices]

    # Feature standardization
    feature_scaler = FeatureScaler(targets=["x", "edge_attr"])
    train_dataset = feature_scaler.fit_transform(train_dataset)
    val_dataset = feature_scaler.transform(val_dataset)
    test_dataset = feature_scaler.transform(test_dataset)

    # Target standardization
    target_scaler = TargetScaler()
    train_dataset = target_scaler.fit_transform(train_dataset)
    val_dataset = target_scaler.transform(val_dataset)
    test_dataset = target_scaler.transform(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    print(f"Stage 2 Train size: {len(train_dataset)} | Val size: {len(val_dataset)} | Test size: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, target_scaler


def get_dataset_from_mlsmr_mtb(smiles_list, descriptor_names=None):
    df = pd.read_excel("/work/pi_annagreen_umass_edu/shiyun/MycoPermeNet/data/Prep notebook (Seigrist).xlsx",
                       sheet_name="mlsmr")
    df = df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

    # Keep only rows in smiles_list
    df = df[df["SMILES"].isin(smiles_list)].reset_index(drop=True)

    # Compute descriptor if needed
    if descriptor_names:
        smiles_to_descriptor, valid_descriptor_names, _ = mtb_descriptors_from_names(smiles_list, descriptor_names)
        print("molecular descriptor dim:", len(valid_descriptor_names))

    data_list = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        smi = row["SMILES"]
        if descriptor_names and smi not in smiles_to_descriptor:
            continue  # skip if descriptor not available due to cleaning

        data = smiles2data(smi, explicit_h=False)
        data.smiles = smi
        if descriptor_names:
            desc = smiles_to_descriptor[smi]
            data.rdkit = torch.from_numpy(desc).unsqueeze(0)  # (1, D)

        data_list.append(data)

    return RevIndexedDataset(data_list)


def get_dataset_from_enamine(smiles_list, descriptor_names=None):
    df = pd.read_csv("./data/enamine.csv")
    df = df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

    # Keep only rows in smiles_list
    df = df[df["SMILES"].isin(smiles_list)].reset_index(drop=True)

    # Compute descriptor if needed
    if descriptor_names:
        smiles_to_descriptor, valid_descriptor_names, _ = descriptors_from_names(smiles_list, descriptor_names)
        print("molecular descriptor dim:", len(valid_descriptor_names))

    data_list = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        smi = row["SMILES"]
        if descriptor_names and smi not in smiles_to_descriptor:
            continue  # skip if descriptor not available due to cleaning

        data = smiles2data(smi, explicit_h=False)
        data.smiles = smi
        if descriptor_names:
            desc = smiles_to_descriptor[smi]
            data.rdkit = torch.from_numpy(desc).unsqueeze(0)  # (1, D)

        data_list.append(data)

    return RevIndexedDataset(data_list)


def get_dataset_from_mlsmr(smiles_list, descriptor_names=None):
    df = pd.read_excel("/work/pi_annagreen_umass_edu/Isha/MycoPermeNet-v2/data/Prep notebook (Seigrist).xlsx",
                       sheet_name="mlsmr")
    df = df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

    # Keep only rows in smiles_list
    df = df[df["SMILES"].isin(smiles_list)].reset_index(drop=True)

    # Compute descriptor if needed
    if descriptor_names:
        smiles_to_descriptor, valid_descriptor_names, _ = descriptors_from_names(smiles_list, descriptor_names)
        print("molecular descriptor dim:", len(valid_descriptor_names))

    data_list = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        smi = row["SMILES"]
        if descriptor_names and smi not in smiles_to_descriptor:
            continue  # skip if descriptor not available due to cleaning

        data = smiles2data(smi, explicit_h=False)
        data.smiles = smi
        if descriptor_names:
            desc = smiles_to_descriptor[smi]
            data.rdkit = torch.from_numpy(desc).unsqueeze(0)  # (1, D)

        data_list.append(data)

    return RevIndexedDataset(data_list)


def get_dataloaders_from_mlsmr(dataset):
    feature_scaler = FeatureScaler(targets=["x", "edge_attr"])
    dataset = feature_scaler.fit_transform(dataset)

    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    return dataloader


def get_dataset_from_qm9(target_idx=7, fusion=False):
    """
    Loads QM9 dataset and returns it in RevIndexedDataset format with optional RDKit fusion descriptors.

    Parameters:
    - target_idx (int): index of target property to predict (default: 7 = gap)
    - fusion (bool): whether to include RDKit descriptors

    Returns:
    - RevIndexedDataset, optionally descriptor_names
    """

    dataset = QM9(root='/scratch3/workspace/swa_umass_edu-qm9/QM9')
    smiles_list = [d.smiles for d in dataset]

    if fusion:
        smiles_to_descriptor, descriptor_names, _ = descriptors_from_smiles(smiles_list)
        print("molecular descriptor dim:", len(descriptor_names))

    data_list = []
    for mol_data in tqdm(dataset, desc="Processing QM9"):
        smi = mol_data.smiles
        try:
            y_val = mol_data.y[:, target_idx].item()
        except Exception as e:
            print(f"[Error] Failed to get target for {smi}: {e}")
            continue

        try:
            data = smiles2data(smi, explicit_h=False)
            if data.edge_index.size(1) == 0:
                print(f"[Warning] Skipping molecule with no edges: {smi}")
                continue
        except Exception as e:
            print(f"[Error] Failed to process {smi}: {e}")
            continue

        data.smiles = smi
        data.y = torch.tensor([[y_val]], dtype=torch.float)

        if fusion:
            descriptor = smiles_to_descriptor[smi]
            data.rdkit = torch.from_numpy(descriptor).unsqueeze(0)

        data_list.append(data)

    if fusion:
        return RevIndexedDataset(data_list), descriptors_from_names

    return RevIndexedDataset(data_list)


if __name__ == "__main__":
    train_df = pd.read_csv('./data/train_scaffold_split.csv')
    test_df = pd.read_csv('./data/test_scaffold_split.csv')

    fg_vector = pd.read_csv("./data/functional_group_onehot.csv")
    smiles_to_fingerprint = {row['Smiles']: row.iloc[1:].astype(float).values for _, row in fg_vector.iterrows()}

    train_val_dataset = get_dataset(train_df, smiles_to_fingerprint)
    test_dataset = get_dataset(test_df, smiles_to_fingerprint)

    print(f"Train_Val dataset size: {len(train_val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    print(train_val_dataset[0])
    print(train_val_dataset[0].smiles)
    print(train_val_dataset[0].y)
    print(train_val_dataset[0].fg)

    train_val_atom_cout = [train_val_dataset[i].num_nodes for i in range(len(train_val_dataset))]
    test_atom_count = [test_dataset[i].num_nodes for i in range(len(test_dataset))]

    all_atom_count = train_val_atom_cout + test_atom_count

    print(max(all_atom_count), min(all_atom_count), np.mean(all_atom_count))
    print(max(train_val_atom_cout), min(train_val_atom_cout), np.mean(train_val_atom_cout))
    print(max(test_atom_count), min(test_atom_count), np.mean(test_atom_count))

    train_loader, val_loader, test_loader, target_scaler = get_dataloaders(
        train_val_dataset, test_dataset, batch_size=64
        )

    for i, data in enumerate(train_loader):
        if i == 0:
            print(data)
            break
