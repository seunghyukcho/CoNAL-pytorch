import torch
import torch.nn as nn
import torch.nn.functional as f


class AuxiliaryNetwork(nn.Module):
    def __init__(self, x_dim, e_dim, w_dim):
        super().__init__()

        self.weight_v1 = nn.Linear(x_dim, 128)
        self.weight_v2 = nn.Linear(128, w_dim)
        self.weight_u = nn.Linear(e_dim, w_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x, e):
        v = self.weight_v1(x)
        v = self.weight_v2(v)
        v = f.normalize(v)
        u = self.weight_u(e)
        u = f.normalize(u)
        u = torch.transpose(u, 0, 1)
        w = torch.matmul(v, u)
        w = self.activation(w)

        return w


class NoiseAdaptationLayer(nn.Module):
    def __init__(self, n_class, n_annotator):
        super().__init__()

        self.global_confusion_matrix = nn.Parameter(torch.eye(n_class, n_class), requires_grad=True)
        self.local_confusion_matrices = nn.Parameter(
            torch.stack([torch.eye(n_class, n_class) for _ in range(n_annotator)]),
            requires_grad=True
        )

    def forward(self, f, w):
        global_prob = torch.einsum('ij,jk->ik', f, self.global_confusion_matrix)
        local_probs = torch.einsum('ik,jkl->ijl', f, self.local_confusion_matrices)

        h = w[:, :, None] * global_prob[:, None, :] + (1 - w[:, :, None]) * local_probs

        return h


class CoNAL(nn.Module):
    def __init__(self, input_dim, n_class, n_annotator, classifier, annotator_dim=77, embedding_dim=20):
        super().__init__()

        self.auxiliary_network = AuxiliaryNetwork(input_dim, annotator_dim, embedding_dim)
        self.classifier = classifier
        self.noise_adaptation_layer = NoiseAdaptationLayer(n_class, n_annotator)

    def forward(self, x, annotator):
        x_flatten = torch.flatten(x, start_dim=1)

        f = self.classifier(x)
        w = self.auxiliary_network(x_flatten, annotator)
        h = self.noise_adaptation_layer(f, w)

        return h, f
