"""
Preview training data. The data loader makes sure that batches read subsequent parts of the text,
making it possible to train stateful models. Occasionally, the batch is reset, which is indicated to the model
with #### characters.
"""

import os

from utils import data_loader, decode

dataset = "shakespeare_char"
block_size = 64  # context of up to 256 previous characters
exec(open("configurator.py").read())  # overrides from command line or config file

data_iter = data_loader(
    "train",
    block_size,
    1,
    device="cpu",
    data_dir=os.path.join("data", dataset),
    reset_prob=0.3,
)
for _ in range(20):
    X, Y = next(data_iter)  # fetch the very first batch
    print(repr(decode(X[0])))
