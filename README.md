# MLinPL 2024 tutorial repository

(a fork of https://github.com/karpathy/nanoGPT)


<a target="_blank" href="https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2Fjanchorowski%2Fmlinpl2024tutorial">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open in Studio" />
</a>

## Quick Start

First, install the dependencies
```sh
pip install -U torch rotary_embedding_torch numpy tensorboard
```

To prepare the dataset, run
```sh
python data/shakespeare_char/prepare.py
```

This creates a `train.txt` and `val.txt` in that data directory. Now it is time to train your GPT. The size of it very much depends on the computational resources of your system:

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

