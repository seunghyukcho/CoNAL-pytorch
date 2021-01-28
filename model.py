import torch
import torch.nn as nn


class AuxiliaryNetwork(nn.Module):
    def __init__(self, x_dim, e_dim, w_dim):
        super().__init__()

        self.weight_v = nn.Linear(x_dim, w_dim)
        self.weight_u = nn.Linear(e_dim, w_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x, e):
        v = self.weight_v(x)
        u = self.weight_u(e)
        u = torch.transpose(u, 0, 1)
        w = torch.matmul(v, u)
        w = self.activation(w)

        return w


class NoiseAdaptationLayer(nn.Module):
    def __init__(self, n_class, n_annotator):
        super().__init__()

        self.global_confusion_matrix = nn.Linear(n_class, n_class)
        self.local_confusion_matrices = nn.ModuleList([nn.Linear(n_class, n_class) for _ in range(n_annotator)])

    def forward(self, f, w):
        global_confuse = self.global_confusion_matrix(f)
        local_confuses = [confusion_matrix(f) for confusion_matrix in self.local_confusion_matrices]

        h = [local_confuse * w[:, i] + global_confuse * (1 - w[:, i]) for i, local_confuse in enumerate(local_confuses)]
        return h
