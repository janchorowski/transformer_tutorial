"""
Train a tiny GPT2-like transformer model. Sample run:

$ python train.py --batch_size=32
"""

import math
import os
import time

import torch
import torch.utils.tensorboard

from model import GPT, GPTConfig
from utils import data_loader, load_model, prep_torch

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = "out"
eval_interval = 250  # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10  # don't print too too often

eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume'
resume_checkpoint_name = "ckpt"
# data
dataset = "shakespeare_char"
batch_size = 64
block_size = 256  # context of up to 256 previous characters
# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 2000
lr_decay_iters = 2000  # make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small
seed = 1337

grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 100  # how many steps to warm up for
# system
device = (
    "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
)
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

tokens_per_iter = batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

autocast_ctx = prep_torch(seed, dtype, device)

# poor man's data loader
data_dir = os.path.join("data", dataset)


# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    bias=bias,
    dropout=dropout,
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    model = load_model(resume_checkpoint_name, out_dir, device, model_args=None)
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device
)


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        eval_data_iter = data_loader(
            split, block_size, batch_size, device=device, data_dir=data_dir
        )
        for k in range(eval_iters):
            X, Y = next(eval_data_iter)
            with autocast_ctx:
                logits, unused_state = model(X, None)
                loss = model.loss(logits, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def save(model, optimizer, model_args, name="ckpt.pt"):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "config": config,
    }
    fname = os.path.join(out_dir, name)
    print(f"saving checkpoint to {fname}")
    torch.save(checkpoint, fname)


# training loop
tb_writer = torch.utils.tensorboard.writer.SummaryWriter(log_dir=out_dir)
data_iter = data_loader(
    "train", block_size, batch_size, device=device, data_dir=data_dir
)
X, Y = next(data_iter)  # fetch the very first batch
t0 = time.time()
# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

        tb_writer.add_scalar("train/loss", losses["train"], iter_num)
        tb_writer.add_scalar("val/loss", losses["val"], iter_num)
        tb_writer.add_scalar("lr", lr, iter_num)

        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                save(model, optimizer, model_args)
    if iter_num == 0 and eval_only:
        break

    # forward backward update, using the GradScaler if data type is float16
    with autocast_ctx:
        logits, unused_state = model(X)
        loss = model.loss(logits, Y)
    # immediately async prefetch next batch while model is doing the forward pass on the GPU
    X, Y = next(data_iter)
    # backward pass, with gradient scaling if training in fp16
    scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        lossf = loss.item()
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

        tb_writer.add_scalar("train/instant_loss", lossf, iter_num)
        save(model, optimizer, model_args, "last.pt")
    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
