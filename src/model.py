import torch
import torch.nn as nn


# 单个晶体图卷积层
class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer

        Parameters
        ----------
        atom_fea_len: int
            Number of atom hidden features.
        nbr_fea_len: int
            Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(
            2 * self.atom_fea_len + self.nbr_fea_len, 2 * self.atom_fea_len
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus12 = nn.Softplus()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------
        atom_fea: torch.Tensor shape (N, atom_fea_len)
            Atom hidden features before convolution
        nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
            Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
            indices of M neighbors of each atom

        Returns
        ---------
        atom_out_fea: torch.Tensor shape (N, atom_fea_len)
            Atom hidden features after convolution
        """
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [
                atom_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
                atom_nbr_fea,
                nbr_fea,
            ],
            dim=2,
        )
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(
            total_gated_fea.view(-1, self.atom_fea_len * 2)
        ).view(N, M, self.atom_fea_len * 2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus12(atom_fea + nbr_sumed)
        return out


# 预测器
class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """

    def __init__(
        self,
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=1,
        classification=False,
    ):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------
        orig_atom_fea_len: int
            Number of atom_features in the input.
        nbr_fea_len: int
            Number of bond features
        atom_fea_len: int
            Number of hidden atom features in the convolutional layers
        n_conv: int
            Number of convolutional layers
        h_fea_len: int
            Number of hidden features after pooling
        n_h: int
            Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len, bias=False)
        self.convs = nn.ModuleList(
            [
                ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
                for _ in range(n_conv)
            ]
        )
        self.fc0 = nn.Linear(atom_fea_len, h_fea_len)
        self.softplus0 = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList(
                [nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)]
            )
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------
        atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
            Atom features from atom type
        nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
            Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
            indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
            Mapping from the crystal idx to atom idx

        """
        # 嵌入，卷积
        atom_fea = self.embedding(atom_fea)
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)
        # 池化
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        # 输出
        crys_fea = self.softplus0(self.fc0(self.softplus0(crys_fea)))
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, "fcs") and hasattr(self, "softpluses"):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------
        atom_fea: torch.Tensor shape (N, atom_fea_len)
            Atom feature vectors of the batch
        crystal_atom_idx: List of torch.LongTensor of length N0
            Mapping from the crystal idx to atom idx
        """
        assert (
            sum([len(idx_map) for idx_map in crystal_atom_idx])
            == atom_fea.data.shape[0]
        )
        summed_fea = [
            torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
            for idx_map in crystal_atom_idx
        ]
        return torch.cat(summed_fea, dim=0)


# 自编码器
class CrystalAE(nn.Module):
    """
    Create a crystal graph auto encoder to learn node representations through unsupervised training
    """

    def __init__(self, orig_atom_fea_len, nbr_fea_len, atom_fea_len=64, n_conv=3):
        super(CrystalAE, self).__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len, bias=False)
        self.convs = nn.ModuleList(
            [
                ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
                for _ in range(n_conv)
            ]
        )
        self.fc_adj = nn.Bilinear(atom_fea_len, atom_fea_len, 6)
        self.fc1 = nn.Linear(6, 6)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc_atom_feature = nn.Linear(atom_fea_len, orig_atom_fea_len)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        # Encoder part
        atom_fea = self.embedding(atom_fea)
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)
        # Pooling
        bt_atom_fea = [atom_fea[idx_map] for idx_map in crystal_atom_idx]
        # Decoder part
        """
        Architecture: Node emb is N*64. We decode it back to adjacency tensor N*N*4.
        Entry (i,j) is a 4 dim one hot vector,
        where 0th vector = 1 -> No edges
        1st vector = 1 -> 1/2 Edges
        2nd vector = 1 -> 3/4 Edges
        3rd vector = 1 -> more than 5 Edges
        """
        adj_list = []
        atom_feature_list = []
        for atom_fea in bt_atom_fea:
            # adj reconst
            N, dim = atom_fea.shape
            atom_nbr_fea = atom_fea.repeat(N, 1, 1)
            atom_nbr_fea = atom_nbr_fea.contiguous().view(-1, dim)
            atom_adj_fea = torch.unsqueeze(atom_fea, 1).expand(N, N, dim)
            atom_adj_fea = atom_nbr_fea.contiguous().view(-1, dim)
            adj_p = self.fc_adj(atom_adj_fea, atom_nbr_fea)
            adj_p = self.fc1(adj_p)
            adj_p = self.logsoftmax(adj_p)
            adj_list.append(adj_p)
            # Atom Feature Reconstruction
            atom_feature_list.append(self.fc_atom_feature(atom_fea))
        return adj_list, atom_feature_list
