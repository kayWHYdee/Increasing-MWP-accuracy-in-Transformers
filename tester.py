import os
import time
import random
import torch
from tqdm import tqdm
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from src.train_and_evaluate import (
    load_raw_data, transfer_num,
    prepare_data, prepare_train_batch,
    train_tree, evaluate_tree,
    compute_prefix_tree_result,
    time_since, get_sentence_spans
)
from src.expressions_transfer import from_infix_to_prefix
from src.models import Prediction, GenerateNode, Merge
from src.updated import (
    RobertaEncoder,
    HierarchicalReasoningEncoder,
    RPKHSFusion,
    RPKHSEncoderAdapter
)

# ─── Hyperparameters ───────────────────────────────────────────────────────────
DATA_PATH    = "/kaggle/input/math-23/data/Math_23K.json"
SUBSET_RATIO = 0.2
BATCH_SIZE   = 32
EMBED_SIZE   = 128
HIDDEN_SIZE  = 512
LR_ENCODER   = 1e-5
LR_DECODER   = 1e-3
WD_ENCODER   = 1e-2
WD_DECODER   = 1e-5
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MAX_INPUT    = 120

# ─── Load & slice dataset ──────────────────────────────────────────────────────
raw = load_raw_data(DATA_PATH)
pairs, generate_nums, copy_nums = transfer_num(raw)
# convert gold infix→prefix
pairs = [(p[0], from_infix_to_prefix(p[1]), p[2], p[3]) for p in pairs]

# take 20%
n_subset = int(len(pairs) * SUBSET_RATIO)
subset  = pairs[:n_subset]
# split 80/20 train/test within subset
split   = int(len(subset) * 0.8)
train_pairs = subset[:split]
test_pairs  = subset[split:]

# ─── Build vocab on subset ────────────────────────────────────────────────────
input_lang, output_lang, train_pairs, test_pairs = prepare_data(
    train_pairs, test_pairs,
    trim_min_count=1,           # include all tokens in small set
    generate_nums=generate_nums,
    copy_nums=copy_nums,
    tree=True
)

# ─── Instantiate model ────────────────────────────────────────────────────────
roberta = RobertaEncoder(
    model_name="roberta-base",
    target_hidden_size=HIDDEN_SIZE,
    max_length=MAX_INPUT,
    device=DEVICE
).to(DEVICE)
hier   = HierarchicalReasoningEncoder(hidden_size=HIDDEN_SIZE).to(DEVICE)
fusion = RPKHSFusion(hidden_size=HIDDEN_SIZE).to(DEVICE)
encoder= RPKHSEncoderAdapter(input_lang, roberta, hier, fusion).to(DEVICE)

orig_ops = output_lang.n_words - copy_nums - 1 - len(generate_nums)
num_consts = len(generate_nums)
num_ops    = orig_ops - num_consts

predict  = Prediction(HIDDEN_SIZE, orig_ops, num_consts).to(DEVICE)
generate = GenerateNode(HIDDEN_SIZE, num_ops, num_consts, EMBED_SIZE).to(DEVICE)
merge    = Merge(HIDDEN_SIZE, EMBED_SIZE).to(DEVICE)

# ─── Optimizer & Scheduler ───────────────────────────────────────────────────
enc_params   = list(roberta.roberta.parameters())
other_params = (
    list(roberta.project.parameters()) +
    list(hier.parameters()) +
    list(fusion.parameters()) +
    list(predict.parameters()) +
    list(generate.parameters()) +
    list(merge.parameters())
)
optimizer = AdamW([
    {"params": enc_params,   "lr": LR_ENCODER, "weight_decay": WD_ENCODER},
    {"params": other_params, "lr": LR_DECODER, "weight_decay": WD_DECODER},
])
# no warmup here—just constant LR for this quick check
scheduler = None

# ─── Single‐epoch training on the small subset ────────────────────────────────
batches = prepare_train_batch(train_pairs, BATCH_SIZE)
loss_total = 0.0
t0 = time.time()
for i in tqdm(range(len(batches[0])), desc="Training Subset"):
    loss = train_tree(
        # data
        batches[0][i], batches[1][i],
        batches[2][i], batches[3][i],
        batches[5][i], batches[7][i],
        generate_nums,
        # models
        encoder, predict, generate, merge,
        # optimizers (all point to same AdamW)
        optimizer, optimizer, optimizer, optimizer,
        # lang & spans
        output_lang, batches[6][i], batches[8][i]
    )
    loss_total += loss
    if scheduler:
        scheduler.step()
print(f"\n→ [Subset train] loss={loss_total/len(batches[0]):.4f} time={time_since(time.time()-t0)}")

# ─── Evaluation on the subset test split ─────────────────────────────────────
val_acc = eq_acc = tot = 0
t1 = time.time()
for tb in tqdm(test_pairs, desc="Eval Subset"):
    inp, inp_len, gold, gold_len, nums, num_pos, num_stack = tb
    spans = [ get_sentence_spans(inp) ]
    pred = evaluate_tree(
        inp, inp_len, generate_nums,
        encoder, predict, generate, merge,
        output_lang, num_pos, spans,
        beam_size=3
    )
    vac, eqc, _, _ = compute_prefix_tree_result(
        pred, gold, output_lang, nums, num_stack
    )
    val_acc += vac
    eq_acc  += eqc
    tot     += 1

print(f"→ [Subset eval] eq_acc={eq_acc/tot:.4f}, val_acc={val_acc/tot:.4f} time={time_since(time.time()-t1)}")
