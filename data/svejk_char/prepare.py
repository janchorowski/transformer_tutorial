import os

input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
with open(input_file_path, "r") as f:
    data = f.read()
n = len(data)
print(f"length of dataset in characters: {n:,}")

train_data = data[: int(n * 0.9)]
with open(os.path.join(os.path.dirname(__file__), "train.txt"), "wt") as of:
    of.write(train_data)
val_data = data[int(n * 0.9) :]
with open(os.path.join(os.path.dirname(__file__), "val.txt"), "wt") as of:
    of.write(val_data)
