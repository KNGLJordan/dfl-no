# Running the project

- Requires Python and pip. Install dependencies if applicable:
```bash
pip install -r requirements.txt
```

- Run from the repository root (same folder that contains `src/`):
```bash
python -m src.main
```

You can select the model, dataset, and hyperparameters via command-line arguments. Common patterns:
- Model: `--model <name>`
- Dataset: `--dataset <name>`
- Hyperparameters (examples): `--lr <float> --batch-size <int> --epochs <int> --seed <int>`

See all available options with:
```bash
python -m src.main --help
```

