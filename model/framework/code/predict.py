"""MycoPermeNet inference.

MycoPermeNet is a Chemprop directed message-passing neural network (D-MPNN)
trained on the Siegrist Mtb dataset to predict the standardized residual of
mycomembrane permeation in Mycobacterium tuberculosis. The deposited checkpoint
(model/checkpoints/mpn_model.pt) contains both the message-passing encoder and
the 2-layer feed-forward readout head, so a single Chemprop prediction call
returns the permeation score directly.

Note: standardized residuals are inversely proportional to permeation, i.e.
LOWER predicted values correspond to HIGHER mycomembrane permeability. Values
in the training set range from about -3 (most permeable) to +3 (least permeable).

Checkpoint provenance
---------------------
Source: https://github.com/Nevbarunegbe/Mycomembrane-permeability-project
(mpn_model.zip -> mpn_model/mpn_model.pt), the only public MycoPermeNet artifact.

Per its bundled args.json, this checkpoint was trained with Chemprop 1.6.1 on an
80/10/10 Bemis-Murcko `scaffold_balanced` split (seed 0) of the 1,558-compound
dataset, i.e. on ~1,246 compounds with a 157-compound held-out test set. It is
therefore the split-trained *evaluation* model, NOT the full-dataset "final"
model the paper retrains on all data for Fig. 3c onward (that checkpoint was not
released).

Reproduction on the reconstructed 157-compound scaffold test split (this
checkpoint's direct readout): Spearman rho 0.71 (paper 0.74), R2 0.55 (0.51),
RMSE 0.65 (0.75), MAE 0.53 (0.61). The paper's reported metrics come from a
downstream scikit-learn MLP on the MPNN embeddings; using the readout head
directly gives comparable ranking (rho/R2) and slightly lower error.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import argparse

import torch
from chemprop.args import PredictArgs
from chemprop.train import make_predictions

try:
    torch.serialization.add_safe_globals([argparse.Namespace])
except AttributeError:
    pass

_orig_torch_load = torch.load


def _torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load

ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.abspath(os.path.join(ROOT, "..", "..", "checkpoints", "mpn_model.pt"))


def predict(smiles_list):
    """Return the predicted MTB mycomembrane permeation residual for each SMILES.

    Invalid / unparseable SMILES yield np.nan while preserving input order.
    """
    with tempfile.TemporaryDirectory() as tmp:
        input_csv = os.path.join(tmp, "input.csv")
        preds_csv = os.path.join(tmp, "preds.csv")
        pd.DataFrame({"smiles": list(smiles_list)}).to_csv(input_csv, index=False)

        args = PredictArgs().parse_args([
            "--test_path", input_csv,
            "--preds_path", preds_csv,
            "--checkpoint_path", CHECKPOINT,
            "--smiles_columns", "smiles",
            "--num_workers", "0",
        ])
        make_predictions(args)
        preds = pd.read_csv(preds_csv)

    target_col = [c for c in preds.columns if c != "smiles"][0]
    values = []
    for v in preds[target_col]:
        try:
            values.append(float(v))
        except (ValueError, TypeError):
            values.append(np.nan)  # Chemprop writes "Invalid SMILES" for bad inputs
    return values
