# Spider

A simple transformer model train on Shakespeare writes
Default settings:
 - 11000 vocabs
 - 30M parameters

## Installation


```bash
pip install numpy torch 
```

## Usage

To change hyperparameters: edit settings.py

To train tokenizer:
```bash
python tokenizer_train.py
```
To train model:
```bash
python train.py
```

To do inference:
```bash
python transformer_inference.py
```


## License

[MIT](https://choosealicense.com/licenses/mit/)
