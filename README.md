## Math Word Problem Solver

This repository implements a goal-driven, tree-structured neural model for solving Math Word Problems (MWP), adapted from:

* **Improving Math Word Problems with Pre-trained Knowledge and Hierarchical Reasoning** (Weijiang Yu et al.)
* **A Goal-Driven Tree-Structured Neural Model for Math Word Problems**

Using a pre-trained RoBERTa encoder augmented with hierarchical reasoning and a tree-based decoder, the system translates natural language math problems into equations and computes the numeric answer.

---

### Repository Structure

```
├── src/
│   ├── masked_cross_entropy.py       # Masked loss functions
│   ├── expressions_transfer.py       # Infix/postfix/prefix conversions
│   ├── pre_data.py                   # Data loading & preprocessing (Math23K, MAWPS, etc.)
│   ├── updated.py                    # Encoder adapters, hierarchical reasoning, fusion
│   ├── models.py                     # Seq2seq and seq2tree decoder modules
│   └── train_and_evaluate.py         # Training & evaluation routines
├── run_seq2seqtree.py                # Main training script (5-fold CV)
├── requirements.txt                  # Python package dependencies
└── README.md                         # This file
```

### Dependencies

* Python 3.8+
* PyTorch
* Transformers (Hugging Face)
* tqdm

Install via:

```bash
pip install -r requirements.txt
```

### Data Preparation

The code expects the Math23K dataset in JSON format. Configure the path in `run_seq2seqtree.py`:

```python
DATA_PATH = "/path/to/Math_23K.json"
```

### Training

The main script `run_seq2seqtree.py` performs five-fold cross-validation:

```bash
python run_seq2seqtree.py \
  --data_path /path/to/Math_23K.json \
  --batch_size 64 \
  --hidden_size 512 \
  --epochs 10 \
  --device cuda
```

Key hyperparameters:

* `LR_ENCODER`, `LR_DECODER`: learning rates for RoBERTa and decoder
* `FREEZE_EPOCHS`: initial epochs to freeze RoBERTa
* `MAX_INPUT_LENGTH`: maximum token length
* `BEAM_SIZE`: beam size for decoding

Saved models are written to `models/encoder_fold{i}.pt`, etc.

### Inference

Use the evaluation functions in `train_and_evaluate.py`. Example:

```python
from src.train_and_evaluate import evaluate_tree, compute_prefix_tree_result
# load encoder, decoder modules and vocab...
res = evaluate_tree(input_seq, input_len, generate_nums,
                    encoder, predict, generate, merge,
                    output_lang, num_pos, sentence_spans)
equation, value = compute_prefix_tree_result(res, gold_tree, output_lang, num_list, num_stack)
```

### Extending or Adapting

* To swap the encoder, modify `updated.py` and adjust `setup_roberta_training`.
* For different datasets (MAWPS, English), use the corresponding loader in `pre_data.py`.

### License

MIT License.
