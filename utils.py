import os
from contextlib import nullcontext

import numpy as np
import torch

from model import GPT, GPTConfig


def encode(s):
    """Encode string as utf8 bytes."""
    return torch.frombuffer(s.encode(), dtype=torch.uint8).to(torch.int64)


def decode(l):
    """Decode possibly invalid utf8 bytes"""
    return bytes(l.to(torch.uint8)).decode(errors="backslashreplace")


def data_loader(
    split,
    block_size,
    batch_size,
    device="cpu",
    reset_prob=0.1,
    data_dir="data/shakespeare_char/",
):
    data = torch.from_numpy(
        np.memmap(os.path.join(data_dir, f"{split}.txt"), dtype=np.uint8, mode="r")
    )
    # concat data with some separator and repeated data
    separator_length = 4
    separator = data.new_full((separator_length,), ord("#"))
    data = torch.cat((separator, data, separator, data))
    num_tokens = data.size(0)
    idx = torch.full((batch_size,), num_tokens, dtype=torch.long)
    while True:
        # reset a random fraction of indexes,
        reset_mask = (torch.rand(batch_size) < reset_prob) | (
            idx >= num_tokens - block_size
        )
        idx[reset_mask] = torch.randint(num_tokens - block_size - 1, (sum(reset_mask),))
        x = torch.stack([data[i : i + block_size + 1].to(torch.int64) for i in idx])
        x[reset_mask, :separator_length] = ord("#")
        idx += block_size
        y = x[:, 1:].contiguous()
        x = x[:, :-1].contiguous()

        if "cuda" in device:
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                device, non_blocking=True
            )
        else:
            x, y = x.to(device), y.to(device)
        yield (x, y)


def prep_torch(seed, dtype, device):
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in device else "cpu"
    )  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )
    return ctx


def load_model(checkpoint_name, out_dir, device, model_args=None):
    if not checkpoint_name.endswith(".pt"):
        checkpoint_name += ".pt"
    ckpt_path = os.path.join(out_dir, checkpoint_name)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not model_args:
        model_args = checkpoint["model_args"]
    else:
        for k in ["n_layer", "n_head", "n_embd", "bias"]:
            model_args[k] = checkpoint["model_args"][k]

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
