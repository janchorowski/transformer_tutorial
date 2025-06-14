"""
Sample from a trained model
"""

import torch

from utils import decode, encode, load_model, prep_torch

# -----------------------------------------------------------------------------
out_dir = "out"  # ignored if init_from is not 'resume'
checkpoint_name = "ckpt"
start = "####"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1  # number of samples to draw
max_new_tokens = 512  # number of tokens generated in each sample
temperature = (
    0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
)
top_k = (
    10  # retain only the top_k most likely tokens, clamp others to have 0 probability
)
seed = 1337
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32' or 'bfloat16' or 'float16'
exec(open("configurator.py").read())  # overrides from command line or config file
if start.startswith("FILE:"):
    with open(start[5:], "r", encoding="utf-8") as f:
        start = f.read()
# -----------------------------------------------------------------------------


autocast_ctx = prep_torch(seed, dtype, device)

# init from a model saved in a specific directory

model = load_model(checkpoint_name, out_dir, device)
model.eval()

# encode the beginning of the prompt
x = torch.tensor(encode(start), dtype=torch.long, device=device)[None, ...]

# run generation
with torch.no_grad():
    with autocast_ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0]))
            print("---------------")
