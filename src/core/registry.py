class Registry:
    def __init__(self, name):
        self._module_dict = {}
        self.name = name

    def register(self, name=None):
        """Decoratore per registrare una classe o funzione."""
        def _register(cls):
            key = name if name is not None else cls.__name__
            if key in self._module_dict:
                raise KeyError(f"{key} è già registrato nel registry {self.name}")
            self._module_dict[key] = cls
            return cls
        return _register

    def get(self, name):
        """Recupera la classe registrata tramite il nome."""
        if name not in self._module_dict:
            raise KeyError(f"{name} non trovato nel registry {self.name}. "
                           f"Disponibili: {list(self._module_dict.keys())}")
        return self._module_dict[name]

DATASETS = Registry("Datasets")
MODELS = Registry("Models")
SOLVERS = Registry("Solvers")
TRAINERS = Registry("Trainers")