# CS7650-project

## MentionPairDataset Preprocessing Script

This script prepares mention-pair classification data from the [OntoNotes v5.0 (CoNLL-2012)](https://huggingface.co/datasets/conll2012_ontonotesv5) dataset for cross-document coreference resolution tasks using BERT.

It constructs positive and negative mention pairs, encodes them with context windows and optional document-level quality scores, and saves them in a PyTorch-compatible format for fast training.

---

### Output

For each document in the dataset:
- Constructs up to 3 **positive mention pairs** from gold clusters
- Samples an equal number of **negative mention pairs**
- Adds special mention tags: `<m>` and `</m>`
- Optionally prefixes the sequence with document quality scores like `<S1=high>` (if `--score` is used)

Saved files:
- `data/mention_pairs_{split}.pt`
- `data/mention_pairs_{split}_scored.pt` (if `--score` is set)
- `data/tokenizer_train` or `data/tokenizer_train_scored`

---

### Usage

Build dataset with document quality encoded in samples:
```bash
python dataset.py --split train --score --output_dir data
```

Build dataset without document quality encoded in samples:
```bash
python dataset.py --split train --output_dir data
```
---

## Mention Pair Classifier Training with BERT + LoRA

This script fine-tunes a BERT model (with LoRA injected) for a binary classification task on mention pairs, aimed at solving coreference resolution using the CoNLL-2012 OntoNotes dataset. It supports both baseline and scored versions of the dataset, periodic validation, early stopping, and checkpointing.


### Example

Training a doc_quality aware classifier:
```bash
python train.py --lr 2e-5 --epochs 3 --score --save_dir models --load_checkpoint
```
Training a baseline classifier:
```bash
python train.py --lr 2e-5 --epochs 3 --save_dir models --load_checkpoint
```

Results are logged in a file under `logs/`
```bash
logs/training_log_scored.txt  # if --score is used
logs/training_log_plain.txt   # otherwise
```
