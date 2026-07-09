# imports
import os
import sys
import numpy as np
from ersilia_pack_utils.core import read_smiles, write_out

from predict import predict

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# current file directory
root = os.path.dirname(os.path.abspath(__file__))

# read SMILES from .csv file, assuming one column with header
_, smiles_list = read_smiles(input_file)

# run model
outputs = predict(smiles_list)

# check input and output have the same length
assert len(smiles_list) == len(outputs)

# write output in a .csv file
outputs = np.array(outputs, dtype=np.float32).reshape(-1, 1)
header = ["mycomembrane_permeation"]
write_out(outputs, header, output_file, np.float32)
