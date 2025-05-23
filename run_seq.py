# run_seq.py
# coding: utf-8

import os
import time
from tqdm import trange, tqdm

import torch
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from src.train_and_evaluate import (
    load_raw_data, transfer_num,
    prepare_data, prepare_train_batch,
    train_tree, evaluate_tree, compute_prefix_tree_result,
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
DATA_PATH        = "/kaggle/input/math-23/data/Math_23K.json"
BATCH_SIZE       = 64
EMBED_SIZE       = 128
HIDDEN_SIZE      = 512
N_EPOCHS         = 10
BEAM_SIZE        = 5
LR_ENCODER       = 1e-5
LR_DECODER       = 1e-3
WD_ENCODER       = 1e-2
WD_DECODER       = 1e-5
WARMUP_RATIO     = 0.0    # no warm-up → full LR from step 0
FREEZE_EPOCHS    = 0      # unfreeze RoBERTa immediately
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
MAX_INPUT_LENGTH = 120

os.makedirs("models", exist_ok=True)

# ─── Load & preprocess ────────────────────────────────────────────────────────
raw = load_raw_data(DATA_PATH)
pairs, generate_nums, copy_nums = transfer_num(raw)
# convert infix→prefix for the gold equation
pairs = [(p[0], from_infix_to_prefix(p[1]), p[2], p[3]) for p in pairs]

# 5-fold split
fold_size = len(pairs) // 5
folds = [pairs[i*fold_size:(i+1)*fold_size] for i in range(4)]
folds.append(pairs[4*fold_size:])
best_acc = []

# ─── Cross-validation ─────────────────────────────────────────────────────────
for fold in trange(5, desc="Folds", unit="fold"):
    test_pairs  = folds[fold]
    train_pairs = sum(folds[:fold] + folds[fold+1:], [])

    # build vocabs and re-index
    input_lang, output_lang, train_pairs, test_pairs = prepare_data(
        train_pairs, test_pairs,
        trim_min_count=5,
        generate_nums=generate_nums,
        copy_nums=copy_nums,
        tree=True
    )

    # ─── Model instantiation ───────────────────────────────────────────────────
    # Encoder: RoBERTa → hierarchical → fusion
    roberta = RobertaEncoder(
        model_name="roberta-base",
        target_hidden_size=HIDDEN_SIZE,
        max_length=MAX_INPUT_LENGTH,
        device=DEVICE
    ).to(DEVICE)
    hier   = HierarchicalReasoningEncoder(hidden_size=HIDDEN_SIZE).to(DEVICE)
    fusion = RPKHSFusion(hidden_size=HIDDEN_SIZE).to(DEVICE)
    encoder = RPKHSEncoderAdapter(input_lang, roberta, hier, fusion).to(DEVICE)

    # Decoder modules
    original_op_nums = output_lang.n_words - copy_nums - 1 - len(generate_nums)
    num_consts       = len(generate_nums)
    num_ops          = original_op_nums - num_consts

    predict  = Prediction(
        hidden_size=HIDDEN_SIZE,
        op_nums=original_op_nums,
        input_size=num_consts
    ).to(DEVICE)
    generate = GenerateNode(
        hidden_size=HIDDEN_SIZE,
        num_operators=num_ops,
        num_constants=num_consts,
        embedding_size=EMBED_SIZE
    ).to(DEVICE)
    merge    = Merge(
        hidden_size=HIDDEN_SIZE,
        embedding_size=EMBED_SIZE
    ).to(DEVICE)

    # ─── Optimizer & Scheduler ────────────────────────────────────────────────
    enc_params   = list(roberta.roberta.parameters())
    other_params = (
        list(roberta.project.parameters()) +
        list(hier.parameters())     +
        list(fusion.parameters())   +
        list(predict.parameters())  +
        list(generate.parameters()) +
        list(merge.parameters())
    )
    optimizer = AdamW([
        {"params": enc_params,   "lr": LR_ENCODER, "weight_decay": WD_ENCODER},
        {"params": other_params, "lr": LR_DECODER, "weight_decay": WD_DECODER},
    ])
    n_batches   = len(prepare_train_batch(train_pairs, BATCH_SIZE)[0])
    total_steps = N_EPOCHS * n_batches
    warmup_steps= int(WARMUP_RATIO * total_steps)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Optionally freeze RoBERTa for the first FREEZE_EPOCHS
    def set_roberta_grad(on: bool):
        for p in roberta.roberta.parameters():
            p.requires_grad = on
    set_roberta_grad(False)

    # ─── Overfit sanity check ─────────────────────────────────────────────────
    small_train = train_pairs[:10]
    for epoch in range(3):
        batches    = prepare_train_batch(small_train, BATCH_SIZE)
        loss_total = 0.0
        for i in range(len(batches[0])):
            loss = train_tree(
                batches[0][i], batches[1][i],
                batches[2][i], batches[3][i],
                batches[5][i], batches[7][i],
                generate_nums,
                encoder, predict, generate, merge,
                optimizer, optimizer, optimizer, optimizer,
                output_lang, batches[6][i], batches[8][i]
            )
            scheduler.step()
            loss_total += loss
        print(f"[OVERFIT] Epoch {epoch} loss {loss_total/len(batches[0]):.4f}")

    # quick prediction check on those 10 examples
    encoder.eval(); predict.eval(); generate.eval(); merge.eval()
    for inp, inp_len, gold, gold_len, nums, num_pos, num_stack in small_train:
        spans = [get_sentence_spans(inp)]
        pred = evaluate_tree(
            inp, inp_len, generate_nums,
            encoder, predict, generate, merge,
            output_lang, num_pos, spans,
            beam_size=1
        )
        print("Input:", inp)
        print("Gold :", gold)
        print("Pred :", pred)
        print("-"*60)

    # ─── Full training ────────────────────────────────────────────────────────
    set_roberta_grad(False)
    for epoch in trange(N_EPOCHS, desc=f"Fold {fold+1} Epochs", unit="ep"):
        if epoch == FREEZE_EPOCHS:
            set_roberta_grad(True)

        (in_b, in_l, out_b, out_l,
         nums_b, stack_b, pos_b, size_b, spans_b) = prepare_train_batch(train_pairs, BATCH_SIZE)

        loss_total = 0.0
        t0 = time.time()
        for idx in tqdm(range(len(in_l)), desc="Training", unit="batch", leave=False):
            loss = train_tree(
                in_b[idx], in_l[idx],
                out_b[idx], out_l[idx],
                stack_b[idx], size_b[idx],
                generate_nums,
                encoder, predict, generate, merge,
                optimizer, optimizer, optimizer, optimizer,
                output_lang, pos_b[idx], spans_b[idx]
            )
            scheduler.step()
            loss_total += loss

            if idx in (0, len(in_l)-1):
                lrs = [g["lr"] for g in optimizer.param_groups]
                print(f"[LR check] batch {idx} LRs = {lrs}")

        print(f"\nFold[{fold+1}] Epoch[{epoch+1}] "
              f"loss={loss_total/len(in_l):.4f} time={time_since(time.time()-t0)}")

        # ─── Periodic evaluation ───────────────────────────────────────────
        if epoch % 5 == 0 or epoch >= N_EPOCHS-2:
            print("\n>>> Sample predictions:")
            for i, tb in enumerate(test_pairs[:5]):
                inp, inp_len, gold, gold_len, nums, num_pos, spans = tb
                pred_ids = evaluate_tree(
                    inp, inp_len, generate_nums,
                    encoder, predict, generate, merge,
                    output_lang, num_pos, [spans],
                    beam_size=BEAM_SIZE
                )
                print(f"Example {i+1}")
                print("  Input IDs :", inp)
                print("  Gold expr :", gold)
                print("  Pred IDs  :", pred_ids)

            val_acc = eq_acc = n_tot = 0
            t1 = time.time()
            for tb in tqdm(test_pairs, desc="Testing", unit="ex", leave=False):
                res = evaluate_tree(
                    tb[0], tb[1], generate_nums,
                    encoder, predict, generate, merge,
                    output_lang, tb[5], [tb[6]],
                    beam_size=BEAM_SIZE
                )
                vac, eqc, _, _ = compute_prefix_tree_result(
                    res, tb[2], output_lang, tb[4], tb[6]
                )
                val_acc += vac; eq_acc += eqc; n_tot += 1

            print(f"→ Test eq_acc={eq_acc/n_tot:.4f}, val_acc={val_acc/n_tot:.4f} "
                  f"time={time_since(time.time()-t1)}")

            torch.save(encoder.state_dict(),  f"models/enc_f{fold}.pt")
            torch.save(predict.state_dict(),  f"models/pred_f{fold}.pt")
            torch.save(generate.state_dict(), f"models/gen_f{fold}.pt")
            torch.save(merge.state_dict(),    f"models/merg_f{fold}.pt")

            if epoch == N_EPOCHS - 1:
                best_acc.append((eq_acc, val_acc, n_tot))

# ─── Summary ─────────────────────────────────────────────────────────────────
sum_eq, sum_val, sum_tot = map(sum, zip(*best_acc))
print(f"\n>> Overall eq_acc = {sum_eq/sum_tot:.4f}, val_acc = {sum_val/sum_tot:.4f}")
