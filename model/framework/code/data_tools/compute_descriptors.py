# python analyses
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, QED, FindMolChiralCenters
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit.Chem import rdDistGeom, rdFreeSASA
from rdkit.Chem.rdmolops import GetFormalCharge
from itertools import combinations


def fused_ring_count(m):
    """Count the number of fused rings in a molecule."""
    q = m.GetRingInfo()
    rings = [set(r) for r in q.AtomRings()]
    go_next = True
    while go_next:
        go_next = False
        for i, j in combinations(range(len(rings)), 2):
            if rings[i] & rings[j]:
                q = rings[i] | rings[j]
                del rings[j], rings[i]
                rings.append(q)
                go_next = True
                break
    return len(rings)


def count_hbd_hba_atoms(m):
    """Count the number of hydrogen bond donor and acceptor atoms."""
    HDonorSmarts = Chem.MolFromSmarts('[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]')
    HAcceptorSmarts = Chem.MolFromSmarts('[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),' +
                                         '$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),' +
                                         '$([nH0,o,s;+0])]')
    HDonor = m.GetSubstructMatches(HDonorSmarts)
    HAcceptor = m.GetSubstructMatches(HAcceptorSmarts)
    return len(set(HDonor + HAcceptor))


def confgen(smile, prunermsthresh=0.1, numconf=3):
    """Generate conformers for a given molecule."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smile), addCoords=True)
    param = rdDistGeom.ETKDGv2()
    param.pruneRmsThresh = prunermsthresh
    cids = rdDistGeom.EmbedMultipleConfs(mol, numconf, param)
    mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
    AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=4, mmffVariant='MMFF94s')
    return mol


def calc_globularity_pbf(mol):
    """Calculate globularity and PBF for a molecule."""
    glob_ls = []
    pbf_ls = []
    for i in range(len(mol.GetConformers())):
        radii1 = rdFreeSASA.classifyAtoms(mol)
        sasa = rdFreeSASA.CalcSASA(mol, radii1, confIdx=i)
        molv = AllChem.ComputeMolVolume(mol, confId=i)
        globularity = ((molv * 3 / (4 * np.pi)) ** (2 / 3)) * 4 * np.pi / sasa
        pbf = rdMolDescriptors.CalcPBF(mol, confId=i)
        glob_ls.append(globularity)
        pbf_ls.append(pbf)
    return np.mean(np.asarray(glob_ls)), np.mean(np.asarray(pbf_ls))


def calc_descriptors(smiles_series):
    """
    Calculate molecular descriptors for a pandas Series of Smiles strings.
    :param smiles_series: pandas Series of Smiles strings.
    :return: DataFrame of molecular descriptors.
    """
    descriptors = []
    columns = ['HBA', 'HBD', 'HBA+HBD', 'NumRings', 'RTB', 'NumAmideBonds','Globularity', 'PBF',
               'TPSA', 'logP', 'MR', 'MW', 'Csp3',
               'fmf', 'QED', 'HAC', 'NumRingsFused', 'unique_HBAD', 'max_ring_size',
               'n_chiral_centers', 'fcsp3_bm', 'formal_charge', 'abs_charge']

    # for smi in smiles_series:
    for smi in tqdm(smiles_series):
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            try:
                m = confgen(smi)
                hba = rdMolDescriptors.CalcNumHBA(m)
                hbd = rdMolDescriptors.CalcNumHBD(m)
                nrings = rdMolDescriptors.CalcNumRings(m)
                rtb = rdMolDescriptors.CalcNumRotatableBonds(m)
                glob, pbf = calc_globularity_pbf(m)
                psa = rdMolDescriptors.CalcTPSA(m)
                logp, mr = rdMolDescriptors.CalcCrippenDescriptors(m)
                mw = rdMolDescriptors._CalcMolWt(m)
                csp3 = rdMolDescriptors.CalcFractionCSP3(m)
                hac = m.GetNumHeavyAtoms()
                fmf = GetScaffoldForMol(m).GetNumHeavyAtoms() / hac if hac > 0 else 0
                qed = QED.qed(m)
                nrings_fused = fused_ring_count(m)
                n_unique_hba_hbd_atoms = count_hbd_hba_atoms(m)
                max_ring_size = len(max(m.GetRingInfo().AtomRings(), key=len, default=()))
                n_chiral_centers = len(FindMolChiralCenters(m, includeUnassigned=True))
                fcsp3_bm = rdMolDescriptors.CalcFractionCSP3(GetScaffoldForMol(m))
                n_amide_bond = rdMolDescriptors.CalcNumAmideBonds(m)
                f_charge = GetFormalCharge(m)
                abs_charge = abs(f_charge)
                descriptors.append([hba, hbd, hba + hbd, nrings, rtb, n_amide_bond, glob, pbf,
                                    psa, logp, mr, mw, csp3, fmf, qed, hac, nrings_fused,
                                    n_unique_hba_hbd_atoms, max_ring_size, n_chiral_centers,
                                    fcsp3_bm, f_charge, abs_charge])
            except:
                descriptors.append([None] * len(columns))
        else:
            descriptors.append([None] * len(columns))

    return pd.DataFrame(descriptors, columns=columns)


# parser = argparse.ArgumentParser(description='Preprocess descriptors')
# parser.add_argument('--sample_number', type=int, default=1, help='Number of samples for each NST iteration')
# args = parser.parse_args()

# sample_number = args.sample_number

# # mlsmr = pd.read_excel('./data/Prep notebook (Seigrist).xlsx', sheet_name='mlsmr')
# mlsmr = pd.read_csv(f'./data/scaling/5k_samp{sample_number}.csv')
# unlabeled = mlsmr.Smiles
# unlabeled_df = pd.DataFrame(unlabeled, columns=['Smiles'])

# print(unlabeled_df.shape)
# desc_for_mlsmr = calc_descriptors(unlabeled_df.Smiles)
# desc_for_mlsmr['Smiles'] = unlabeled_df['Smiles']

# desc_for_mlsmr.to_csv(f'./data/scaling/5k_samp{sample_number}_26.csv')
# print(desc_for_mlsmr.shape)
