import os
import time
from tqdm import trange, tqdm

import torch
from torch.nn.utils import clip_grad_norm_
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
DATA_PATH        = "/content/drive/MyDrive/code/data/Math_23K.json"
BATCH_SIZE       = 64
EMBED_SIZE       = 128
HIDDEN_SIZE      = 512
N_EPOCHS         = 10
BEAM_SIZE        = 5
LR_ENCODER       = 1e-5
LR_DECODER       = 1e-3
WD_ENCODER       = 1e-2
WD_DECODER       = 1e-5
WARMUP_RATIO     = 0.0
FREEZE_EPOCHS    = 2
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
MAX_INPUT_LENGTH = 120

# ─── Prepare data & 5-fold splits ───────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
raw = load_raw_data(DATA_PATH)
pairs, generate_nums, copy_nums = transfer_num(raw)
pairs = [(p[0], from_infix_to_prefix(p[1]), p[2], p[3]) for p in pairs]

fold_size = len(pairs) // 5
folds = [pairs[i*fold_size:(i+1)*fold_size] for i in range(4)]
folds.append(pairs[4*fold_size:])

best_acc = []

# ─── Five-fold cross-validation ────────────────────────────────────────────────
for fold in trange(5, desc="Folds", unit="fold"):
    test_pairs  = folds[fold]
    train_pairs = sum(folds[:fold] + folds[fold+1:], [])

    input_lang, output_lang, train_pairs, test_pairs = prepare_data(
        train_pairs, test_pairs, 5, generate_nums, copy_nums, True
    )

    # ─── Encoder ────────────────────────────────────────────────────────────────
    roberta = RobertaEncoder(
        model_name="roberta-base",
        target_hidden_size=HIDDEN_SIZE,
        max_length=MAX_INPUT_LENGTH,
        device=DEVICE
    ).to(DEVICE)
    hier   = HierarchicalReasoningEncoder(hidden_size=HIDDEN_SIZE).to(DEVICE)
    fusion = RPKHSFusion(hidden_size=HIDDEN_SIZE).to(DEVICE)
    encoder = RPKHSEncoderAdapter(input_lang, roberta, hier, fusion).to(DEVICE)

    # ─── Decoder ───────────────────────────────────────────────────────────────
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
    merge    = Merge(hidden_size=HIDDEN_SIZE, embedding_size=EMBED_SIZE).to(DEVICE)

    # ─── Optimizer & Scheduler ───────────────────────────────────────────────
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
    encoder_optimizer  = optimizer
    predict_optimizer  = optimizer
    generate_optimizer = optimizer
    merge_optimizer    = optimizer

    n_batches    = len(prepare_train_batch(train_pairs, BATCH_SIZE)[0])
    total_steps  = N_EPOCHS * n_batches
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

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
                # input and target
                batches[0][i], batches[1][i],
                batches[2][i], batches[3][i],
                # number stacks and sizes
                batches[5][i], batches[7][i],
                generate_nums,
                # modules
                encoder, predict, generate, merge,
                # optimizers
                encoder_optimizer, predict_optimizer,
                generate_optimizer, merge_optimizer,
                # lang, num_pos, spans
                output_lang, batches[6][i], batches[8][i]
            )
            scheduler.step()
            loss_total += loss
        print(f"[OVERFIT] Epoch {epoch} loss {loss_total/len(batches[0]):.4f}")

    # quick prediction check
    encoder.eval(); predict.eval(); generate.eval(); merge.eval()
    for inp, gold, nums, num_pos in small_train:
        spans = [get_sentence_spans(inp)]
        pred = evaluate_tree(
            inp, len(inp), generate_nums,
            encoder, predict, generate, merge,
            output_lang, num_pos, spans,
            beam_size=1
        )
        print("Input:", inp)
        print("Gold :", gold)
        print("Pred :", pred)
        print("-"*40)

    # ─── Full training ────────────────────────────────────────────────────────
    for epoch in trange(N_EPOCHS, desc=f"Fold {fold+1} Epochs", unit="ep"):
        if epoch == FREEZE_EPOCHS:
            set_roberta_grad(True)

        (input_batches, input_lengths,
         output_batches, output_lengths,
         nums_batches, num_stack_batches,
         num_pos_batches, num_size_batches,
         sentence_spans_batches) = prepare_train_batch(train_pairs, BATCH_SIZE)

        loss_total = 0.0
        start      = time.time()

        for idx in tqdm(range(len(input_lengths)), desc="Training", unit="batch", leave=False):
            loss = train_tree(
                input_batches[idx], input_lengths[idx],
                output_batches[idx], output_lengths[idx],
                num_stack_batches[idx], num_size_batches[idx],
                generate_nums,
                encoder, predict, generate, merge,
                encoder_optimizer, predict_optimizer,
                generate_optimizer, merge_optimizer,
                output_lang,
                num_pos_batches[idx], sentence_spans_batches[idx]
            )
            scheduler.step()
            loss_total += loss

            if idx in {0, len(input_lengths)-1}:
                lrs = [g['lr'] for g in optimizer.param_groups]
                print(f"[LR check] batch {idx} LRs = {lrs}")

        print(f"\nFold[{fold+1}] Epoch[{epoch+1}] "
              f"loss={loss_total/len(input_lengths):.4f} "
              f"time={time_since(time.time()-start)}")

        if epoch % 5 == 0 or epoch > N_EPOCHS - 3:
            print("\n>>> Sample predictions:")
            for i, tb in enumerate(test_pairs[:5]):
                inp, inp_len, gold, _, _, num_pos, spans = tb
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

            val_ac, eq_ac, tot = 0, 0, 0
            eval_start = time.time()
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
                val_ac += vac; eq_ac += eqc; tot += 1

            print(f"→ Test eq_acc={eq_ac/tot:.4f}, val_acc={val_ac/tot:.4f} "
                  f"time={time_since(time.time()-eval_start)}")

            torch.save(encoder.state_dict(),  f"models/encoder_f{fold}.pt")
            torch.save(predict.state_dict(),  f"models/predict_f{fold}.pt")
            torch.save(generate.state_dict(), f"models/generate_f{fold}.pt")
            torch.save(merge.state_dict(),    f"models/merge_f{fold}.pt")

            if epoch == N_EPOCHS - 1:
                best_acc.append((eq_ac, val_ac, tot))

# ─── Aggregate folds ───────────────────────────────────────────────────────────
sum_eq, sum_val, sum_tot = map(sum, zip(*best_acc))
print(f"\n>> Overall eq_acc = {sum_eq/sum_tot:.4f}, val_acc = {sum_val/sum_tot:.4f}")
