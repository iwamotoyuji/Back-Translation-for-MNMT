# Back-Translation for Unsupervised Multimodal Neural Machine Translation Using Text-to-Image Generation

Pytorch implementation for "Back-Translation for Unsupervised Multimodal Neural Machine
Translation Using Text-to-Image Generation"


### Dependencies
- python 3.8.5
- Pytorch 1.7.0

In addition, please add the project folder to PYTHONPATH and `pip install` the following packages:
- `accimage`
- `apex`
- `nltk`
- `numpy`
- `tqdm`

**Data Preparation**
- For small dataset: `./prepare_small_dataset.sh`
- For large dataset: `./prepare_large_dataset.sh`

**Trainig**
- For small dataset: `./ex_small.sh`
- For large dataset: `./ex_large.sh`

**Evaluation**
- `python evaluate.py`
