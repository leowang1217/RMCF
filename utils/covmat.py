from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign
import math
from copy import deepcopy

def get_best_rmsd(probe, ref):
    probe = Chem.RemoveHs(probe)
    ref = Chem.RemoveHs(ref)
    rmsd = rdMolAlign.GetBestRMS(probe, ref)
    return rmsd


def covmat(pred_mols,ref_mols,thres = 1.25):
    rmsd_mat = -1 * np.ones([len(ref_mols), len(pred_mols)],dtype=np.float)
    valid_flag = True
    for i,p in enumerate(pred_mols):
        for j,r in enumerate(ref_mols):
            try:
                rmsd  = get_best_rmsd(deepcopy(p),deepcopy(r))
            except Exception as e:
                print(e)
                valid_flag = False
                break
            if not math.isnan(rmsd) and rmsd < 100:
                rmsd_mat[j,i] = rmsd
            else:
                valid_flag = False
                break
        if not valid_flag:
            break
    if valid_flag:
        rmsd_ref_min = rmsd_mat.min(-1)    # np (num_ref, )
        rmsd_gen_min = rmsd_mat.min(0)     # np (num_gen, )
        rmsd_cov_thres = rmsd_ref_min <= thres  # np (num_ref, num_thres)
        rmsd_jnk_thres = rmsd_gen_min <= thres # np (num_gen, num_thres)
        return {
            'MAT-R':rmsd_ref_min.mean(),
            'COV-R':rmsd_cov_thres.mean(),
            'MAT-P':rmsd_gen_min.mean(),
            'COV-P':rmsd_jnk_thres.mean()
        }
    else:
        return {
            'MAT-R':2.0,
            'COV-R':-1.0,
            'MAT-P':2.0,
            'COV-P':-1.0
        }