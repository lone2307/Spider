# Spider

A simple transformer model train on Shakespeare writes

Tech used:
 - WordPiece tokenizer
 - RoPE positional embedding, combined with NTK and YaRN for better scaling

Default settings:
 - 11000 vocabs
 - 30M parameters

## Installation


```bash
pip install numpy torch 
```

## Usage

To change hyperparameters: edit settings.py

To train model, run:
```bash
train.ipynb
```

To do inference:
```bash
inference.ipynb
```


## License

[MIT](https://choosealicense.com/licenses/mit/)
