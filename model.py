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

        h = [local_confuse * w[:, i:i + 1] + global_confuse * (1 - w[:, i:i + 1]) for i, local_confuse in enumerate(local_confuses)]
        h = torch.stack(h)
        h = torch.transpose(h, 0, 1)

        return h


class CoNAL(nn.Module):
    def __init__(self, input_dim, n_class, n_annotator, classifier, annotator_dim=77, embedding_dim=20):
        super().__init__()

        self.auxiliary_network = AuxiliaryNetwork(input_dim, annotator_dim, embedding_dim)
        self.classifier = classifier
        self.noise_adaptation_layer = NoiseAdaptationLayer(n_class, n_annotator)

    def forward(self, x, annotator):
        batch_size = x.size(0)
        x_flatten = torch.reshape(x, (batch_size, -1))

        f = self.classifier(x)
        w = self.auxiliary_network(x_flatten, annotator)
        h = self.noise_adaptation_layer(f, w)

        return h
