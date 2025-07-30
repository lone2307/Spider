# Spider

A simple transformer model train on Shakespeare writes

Tech used:
 - WordPiece tokenizer
 - Rotary Positional Embedding (RoPE), combined with NTK and YaRN for better long context scaling
 - GLU (Gated Linear Unit) for Feed Forward layer

Default settings:
 - 11000 vocabs
 - 30M parameters

## Installation


```bash
pip install numpy torch 
```

## Usage

To edit hyperparameters, change values in:

```bash
settings.py
```

To train model, run:
```bash
train.ipynb
```

To do inference, run:
```bash
inference.ipynb
```


## License

[MIT](https://choosealicense.com/licenses/mit/)
