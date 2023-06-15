import os
import csv
import json
import functools
import copy
import numpy as np
from pymatgen.core.structure import Structure
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from prefetch_generator import BackgroundGenerator


def get_data_loader(
    dataset,
    collate_fn=default_collate,
    data_size=100,
    batch_size=64,
    num_workers=1,
    pin_memory=False,
):
    train_sampler = SubsetRandomSampler(list(range(data_size)))
    data_loader = DataLoaderX(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    return data_loader


def collate_pool(dataset_list):
    (
        batch_atom_fea,
        batch_nbr_fea,
        batch_nbr_fea_idx,
        crystal_atom_idx,
        batch_adj,
        batch_cif_ids,
    ) = ([], [], [], [], [], [])
    base_idx = 0
    for _, ((atom_fea, nbr_fea, nbr_fea_idx), adj, cif_id) in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)
        batch_cif_ids.append(cif_id)
        batch_adj.append(adj)
        base_idx += n_i
    return (
        (
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx,
        ),
        batch_adj,
        batch_cif_ids,
    )


def get_molecular_adj(nbr_fea_idx, N):
    edge_list = nbr_fea_idx
    adj = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in edge_list[i]:
            if adj[i][j] <= 4:
                adj[i][j] = adj[i][j] + 1
    return adj


def disc_edge_feature(edge, data):
    N = np.shape(edge)[0]
    nbr_fea_idx = data
    edge_tensor = np.zeros((N, N, 5))
    for i in range(np.shape(edge)[0]):
        for j in range(np.shape(edge)[1]):
            edge_dist = int(round(edge[i][j]))
            edge_idx = nbr_fea_idx[i][j]
            if 4 <= edge_dist:
                edge_tensor[i][edge_idx][4] = 1
            else:
                edge_tensor[i][edge_idx][edge_dist - 1] = 1
    return edge_tensor


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())



class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(
            -((distances[..., np.newaxis] - self.filter) ** 2) / self.var**2
        )


class AtomInitializer(object):
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {
            idx: atom_type for atom_type, idx in self._embedding.items()
        }

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, "_decodedict"):
            self._decodedict = {
                idx: atom_type for atom_type, idx in self._embedding.items()
            }
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    def __init__(self, root_dir, max_num_nbr, radius, dmin=0, step=0.2):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius, self.dmin, self.step = max_num_nbr, radius, dmin, step
        id_prop_file = os.path.join(self.root_dir, "id_prop.csv")
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            # headings = next(reader)
            self.id_prop_data = [row for row in reader]
        atom_init_file = os.path.join(self.root_dir, "atom_init.json")
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=self.dmin, dmax=self.radius, step=self.step)
        self.record_all_path = os.path.join(self.root_dir, "record_all_path.json")
        with open(self.record_all_path, "r") as f:
            record_hash = str(f'VE_{self.max_num_nbr}_{self.radius}_{dmin}_{self.step}')
            self.record_path = os.path.join(self.root_dir, record_hash)
            record_all_path = json.load(f)
            if not record_hash in record_all_path.keys():
                record_all_path[record_hash] = {"type" : "VE", "max_num_nbr" : self.max_num_nbr, "radius" : self.radius, "dmin" : self.dmin, "step" : self.step}
                os.makedirs(self.record_path)
        with open(self.record_all_path, "w") as f:
            f.write(json.dumps(record_all_path, indent=1))

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, _ = copy.deepcopy(self.id_prop_data[idx])
        if os.path.exists(os.path.join(self.record_path, cif_id)):
            data = torch.load(os.path.join(self.record_path, cif_id))
            atom_fea = data['atom_fea']
            nbr_fea = data['nbr_fea']
            nbr_fea_idx = data['nbr_fea_idx']
            adj = data['adj']
        else:
            crystal = Structure.from_file(os.path.join(self.root_dir, cif_id + ".cif"))
            atom_fea = np.vstack(
                [
                    self.ari.get_atom_fea(crystal[i].specie.number)
                    for i in range(len(crystal))
                ]
            )
            all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            nbr_fea_idx, nbr_fea = [], []
            for nbr in all_nbrs:
                if len(nbr) < self.max_num_nbr:
                    nbr_fea_idx.append(
                        list(map(lambda x: x[2], nbr)) + [0] * (self.max_num_nbr - len(nbr))
                    )
                    nbr_fea.append(
                        list(map(lambda x: x[1], nbr))
                        + [self.radius + 1.0] * (self.max_num_nbr - len(nbr))
                    )
                else:
                    nbr_fea_idx.append(list(map(lambda x: x[2], nbr[: self.max_num_nbr])))
                    nbr_fea.append(list(map(lambda x: x[1], nbr[: self.max_num_nbr])))
            nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
            adj = get_molecular_adj(nbr_fea_idx, nbr_fea.shape[0])
            adj = adj.flatten()
            nbr_fea = self.gdf.expand(nbr_fea)
            atom_fea = torch.Tensor(atom_fea)
            nbr_fea = torch.Tensor(nbr_fea)
            nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
            adj = torch.LongTensor(adj)
            torch.save({'atom_fea':atom_fea, 'nbr_fea':nbr_fea, 'nbr_fea_idx':nbr_fea_idx, 'adj':adj}, os.path.join(self.record_path, cif_id))
        return (atom_fea, nbr_fea, nbr_fea_idx), adj, cif_id
