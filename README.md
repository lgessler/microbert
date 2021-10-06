# Usage

## Setup
1. Ensure submodules are initialized:

```
git submodule update --init --recursive
```

2. Create a new environment:

```bash
conda create --name embur python=3.9
conda activate embur
```

3. Install PyTorch etc. 
```bash
conda install pytorch torchvision cudatoolkit -c pytorch
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Experiments

### Adding a language
For each new language to be added, you'll want to follow these conventions:

1. Put all data under `data/$NAME/`, with "raw" data going in some kind of subdirectory. 
   (If it is a UD corpus, the standard UD name would be good, e.g. `data/coptic/UD_Coptic-Scriptorium`)
2. Put a script at `embur/scripts/$NAME_data_prep.py` that will take the dataset's native format and 
   write it out into `data/$NAME/converted`. 
   (Note that this script is a submodule of the top-level package `embur`.
   To invoke it, you'll write `python embur.scripts.$NAME_data_prep`.)
3. Update `embur.language_configs` with the language's information.   

### Running experiments

1. Invoke your data preprocessing script: `python -m embur.scripts.coptic_data_prep`
2. Run pretraining: `python main.py pretrain` (invoke with `--help` to see options)
3. Run the evaluation: 
