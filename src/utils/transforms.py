import torch

class TorchStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        # Calcola media e deviazione standard sul tensore
        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, keepdim=True)
        # Evita divisioni per zero se una feature è costante
        self.std[self.std == 0] = 1e-8 

    def transform(self, x):
        return (x - self.mean) / self.std

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        # Fondamentale se decidi di scalare anche le y
        return (x * self.std) + self.mean