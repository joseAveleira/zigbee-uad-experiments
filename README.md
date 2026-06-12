# zigbee-uad-experiments

Experiment code for the article:

> J. Aveleira-Mata, A. Merayo, I. Fern&aacute;ndez-Mart&iacute;nez, D. Narciandi-Rodr&iacute;guez,
> I. Garc&iacute;a-Rodr&iacute;guez, *"Zigbee_UAD: a labelled IEEE 802.15.4 attack dataset and a
> leakage-aware machine learning baseline for intrusion detection"*, Ad Hoc Networks.

These scripts train and evaluate the intrusion-detection models reported in the article on the
**Zigbee_UAD dataset**, available on figshare
(DOI: [10.6084/m9.figshare.32657583](https://doi.org/10.6084/m9.figshare.32657583)).

## Files

- `run_experiments.py` &mdash; runs every experiment reported in the article.
- `preprocess.py` &mdash; loads the dataset CSVs, encodes the protocol fields and builds the
  causal traffic-context features. It is imported by `run_experiments.py` and is not run on its own.

## Requirements

Python &ge; 3.10:

```
pip install -r requirements.txt
```

## Usage

Download the three CSV files of the dataset from figshare and pass their folder with `--data`.
Run the stages in order:

```
python run_experiments.py importance --data PATH/TO/dataset
python run_experiments.py models     --data PATH/TO/dataset
python run_experiments.py svm --set protocol --data PATH/TO/dataset
python run_experiments.py svm --set context  --data PATH/TO/dataset
python run_experiments.py svm --set reduced  --data PATH/TO/dataset
python run_experiments.py scenarios  --data PATH/TO/dataset
python run_experiments.py leakage    --data PATH/TO/dataset
```

The run is deterministic (fixed seed and stratified train/test split), so it reproduces the
results reported in the article. Metrics are written to `results/results.json`.
