# Transformer tutorial repository

(a fork of https://github.com/karpathy/nanoGPT)


<a target="_blank" href="https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2Fjanchorowski%2Ftransformer_tutorial">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open in Studio" />
</a>

## Quick Start

First, install the dependencies
```sh
pip install -U torch rotary_embedding_torch numpy tensorboard gdown
```

## Training a model

**I have a GPU**. Great, we can quickly train a baby GPT with the settings provided in the [config/train_shakespeare_char.py](config/train_shakespeare_char.py) config file:

```sh
python train.py config/train_shakespeare_char.py
```

You can pass arguments of the form `--name=value` to override a variable called `name` in the code.

If you peek inside it, you'll see that we're training a GPT with a context size of up to 256 characters, 384 feature channels, and it is a 6-layer Transformer with 6 heads in each layer. On one A100 GPU this training run takes about 3 minutes and the best validation loss is 1.4697. Based on the configuration, the model checkpoints are being written into the `--out_dir` directory `out-shakespeare-char`. So once the training finishes we can sample from the best model by pointing the sampling script at this directory:

```sh
python sample.py --out_dir=out-shakespeare-char
```

This generates one sample, for example:

```
Come hither, sir, let him challenge the fairest lady:
High happy times have not such a tidings ha'
A shown of me, so she was not good for revenges but
conclusion what to say the man: but so goes so,
the aeor suburning up for the friar away.
```

lol  `¯\_(ツ)_/¯`. Not bad for a character-level model after few minutes of training on a GPU.

There are two types of checkpoints written:
- `ckpt.pt`, written after evaluation if the model has the smallest validation loss so far.
- `last.pt`, written after every `log_interval` iterations
Sampling by default uses `ckpt.pt`, you can change it by passing `--checkpoint_name=last.pt`.

You can resume training a model instead of creating a new one by passing `--init_from=resume` to `train.py`.
By default it will load the `ckpt.pt` checkpoint, you can change it by using `--resume_checkpoint_name=...`
```sh
python train.py config/train_shakespeare_char.py --out_dir=out-shakespeare-char --init_from=resume --resume_checkpoint_name=ckpt.pt
```

Finally, to only calculate the validation loss, resume training passing additionally `--max_iters=0 --eval_interval=1`.

## Sampling / Inference

Use the script `sample.py` to sample from a model you trained yourself. Use the `--out_dir` to point the code appropriately. You can also prompt the model with some text from a file, e.g. ```python sample.py --start=FILE:prompt.txt```. You can pass `--checkpoint_name=last.pt` to use the latest checkpoint instead of the best one.

## Exercises

### Assignment 1: Warmup

Download a pre-trained model:

```bash
mkdir -p out-shakespeare-char
pushd out-shakespeare-char
gdown https://drive.google.com/uc?id=14OPzCVBiCo-z9F6liAGAe0cyhPCtXycu
popd
```

Then generate a sample:
```bash
python sample.py --out_dir=out-shakespeare-char
```

Notice how the model produces a few good outputs, then starts generating noise.

Let's apply two hotfixes:
1. change the sampling code to limit the length of historical kv0cache provided to the model at each inference step

2. Undo the last change. Instead apply a few iterations of trainig with a much larger `block_size`. How many are needed to tun the model to use a larger block_size?

Hint: use the options `resume`, `block_size`, and `max_iters`. When decoding use the `--checkpoint_name=last` to read the last checkpoint created every few iterations.

### Assignment 2: Implement a TransformerXL
Change the training and evaluation during training code to pass the model the last attention state. In this way, we can train the model on extended history, striking a tradeoff between context used by the model and the amount of gradient backpropagation through time done during training.

We can train the model in this manner because the data loader emits related data batches (please verify this using the `gen_sample_data.py` utility).

Don't forget to limit the size of the historical KV-cache!


### Assignment 3: Implement a linear transformer

Implement the [Linear Transformer](https://arxiv.org/abs/2006.16236). Use the naive implementation of the attention and remove SoftMax normalization. To make the model trainable, you may need to normalize attention outputs (e.g. using `torch.nn.functional.layer_norm`).

Please note that linear attention works like a [Hopfield](https://en.wikipedia.org/wiki/Hopfield_network) associative memory, the storage capacity depends on the attention head dimension. The default configuration uses a dimension of `64`, severly limiting the capacity of the associative memory. For best result in this quick exploration, compare with regular transformers trained with small `block_sizes`. 

Hint: to keep the historical KV-cache in the form of a fixed dimension matrix, you will need to either disable Rope positional embeddings, or rotate the historical data.

### Assignment 4: open-ended exploratoin of KV-cache size reduction

Implement as many tricks as you can think of to reduce the size of block_states.
Do not try to optimize the latency at all cost. Tip: pay attention to the offset
when applying rotary embeddings when the KV cache has been modified.

Training-free ideas include:
- Quantizing the KV cache tensors to 8 bits. Which 8-bit format would you use?
  What it would take to go lower than 8 bits?
- Evicting (removing) unused tokens based on their cumulative attention scores
  (tech hint: use the naive attnetion implementation to access attention scores)

Retrain / do a few gradient steps from a saved checkpoint to change how
the model accesses its context:
- Do a few gradient steps with a much longer context. How many are needed to
  make it work?
- Use grouped query attention (have fewer KV heads that query heads)
  as in https://arxiv.org/abs/1911.02150 (tech hint: use the `enable_gqa`
  argument of scaled_dot_product_attention)
- Use long attention only in a few layers, limit others to small windows
  and share attention between neighboring layers
  https://research.character.ai/optimizing-inference/

How low, in terms of KV size in bits can you go (theoretically, you can
use masking and aligned data structures to make implementation easier)
