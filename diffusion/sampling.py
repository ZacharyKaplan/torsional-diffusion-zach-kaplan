from utils.featurization import featurize_mol_from_smiles
from utils.torsion import *
from diffusion.likelihood import *
import torch, copy
import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem

device = torch.device('cpu')

def try_mmff(mol):
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
        return True
    except Exception as e:
        return False


def get_seed(smi, seed_confs=None, dataset='drugs'):
    mol, data = featurize_mol_from_smiles(smi, dataset=dataset)
    if not mol:
        return None, None
    data.edge_mask, data.mask_rotate = get_transformation_mask(data)
    data.edge_mask = torch.tensor(data.edge_mask)
    return mol, data


def embed_seeds(mol, data, n_confs, single_conf=False, smi=None, embed_func=None, seed_confs=None):
    if not seed_confs:
        embed_num_confs = n_confs if not single_conf else 1
        try:
            mol = embed_func(mol, embed_num_confs)
        except Exception as e:
            print(e.output)
            pass
        if len(mol.GetConformers())!=embed_num_confs:
            print(len(mol.GetConformers()), '!=', embed_num_confs)
            return None
    conformers = []
    for i in range(n_confs):
        data_conf = copy.deepcopy(data)
        seed_mol = copy.deepcopy(mol)
        [seed_mol.RemoveConformer(j) for j in range(n_confs) if j != i]
        data_conf.pos = torch.from_numpy(seed_mol.GetConformers()[0].GetPositions()).float()
        data_conf.seed_mol = copy.deepcopy(seed_mol)
        conformers.append(data_conf)
    if mol.GetNumConformers() > 1:
        [mol.RemoveConformer(j) for j in range(n_confs) if j != 0]
    return conformers


def perturb_seeds(data):
    for data_conf in data:
        torsion_updates = np.random.uniform(low=-np.pi,high=np.pi, size=data_conf.edge_mask.sum())
        data_conf.pos = modify_conformer(data_conf.pos, data_conf.edge_index.T[data_conf.edge_mask],
                                         data_conf.mask_rotate, torsion_updates)
        data_conf.total_perturb = torsion_updates
    return data


def sample(conformers, model, sigma_max=np.pi, sigma_min=0.01 * np.pi, steps=20, batch_size=32,
           ode=False, likelihood=None):
    conf_dataset = InferenceDataset(conformers)
    loader = DataLoader(conf_dataset, batch_size=batch_size, shuffle=False)

    sigma_schedule = 10 ** np.linspace(np.log10(sigma_max), np.log10(sigma_min), steps + 1)[:-1]
    eps = 1 / steps
    for data in loader:
        dlogp = torch.zeros(data.num_graphs)
        data_gpu = copy.deepcopy(data).to(device)
        for sigma in sigma_schedule:
            data_gpu.node_sigma = sigma * torch.ones(data.num_nodes, device=device)
            with torch.no_grad():
                data_gpu = model(data_gpu)
            g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
            z = torch.normal(mean=0, std=1, size=data_gpu.edge_pred.shape)
            score = data_gpu.edge_pred.cpu()
            perturb = g ** 2 * eps * score + g * np.sqrt(eps) * z
            conf_dataset.apply_torsion_and_update_pos(data, perturb.numpy())
            data_gpu.pos = data.pos.to(device)
            for i, d in enumerate(dlogp):
                conformers[data.idx[i]].dlogp = d.item()
    return conformers


def pyg_to_mol(mol, data, rmsd=True):
    if not mol.GetNumConformers():
        conformer = Chem.Conformer(mol.GetNumAtoms())
        mol.AddConformer(conformer)
    coords = data.pos
    if type(coords) is not np.ndarray:
        coords = coords.double().numpy()
    for i in range(coords.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(coords[i, 0], coords[i, 1], coords[i, 2]))
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
    except Exception as e:
        pass
    try:
        if rmsd:
            mol.rmsd = AllChem.GetBestRMS(
                Chem.RemoveHs(data.seed_mol),
                Chem.RemoveHs(mol)
            )
        mol.total_perturb = data.total_perturb
    except:
        pass
    mol.n_rotable_bonds = data.edge_mask.sum()
    return mol


class InferenceDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        for i, d in enumerate(data_list):
            d.idx = i
        self.data = data_list

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

    def apply_torsion_and_update_pos(self, data, torsion_updates):
        pos_new, torsion_updates = perturb_batch(data, torsion_updates, split=True, return_updates=True)
        for i, idx in enumerate(data.idx):
            try:
                self.data[idx].total_perturb += torsion_updates[i]
            except:
                pass
            self.data[idx].pos = pos_new[i]
