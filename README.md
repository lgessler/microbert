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

This repo is exposed as a CLI with the following commands:

```
├── data                  # Data prep commands
│   ├── prepare-mlm
│   └── prepare-ner
├── word2vec              # Static embedding condition
│   ├── train
│   ├── evaluate-ner
│   └── evaluate-parser
├── mbert                 # Pretrained MBERT
│   ├── evaluate-ner
│   └── evaluate-parser
├── mbert-va              # Pretrained MBERT with Chau et al. (2020)'s VA method
│   ├── evaluate-ner
│   ├── evaluate-parser
│   └── train
├── bert                  # Monolingual BERT--main experimental condition
│   ├── evaluate-ner
│   ├── evaluate-parser
│   └── train
├── evaluate-ner-all      # Convenience to perform evals on all NER conditions
├── evaluate-parser-all   # Convenience to perform evals on all parser conditions
└── stats                 # Supporting commands for statistical summaries
    └── format-metrics
```

To see more information, add `--help` at the end of any partial subcommand, e.g. `python main.py --help`, 
`python main.py bert --help`, `python main.py word2vec train --help`.

### Adding a language
For each new language to be added, you'll want to follow these conventions:

1. Put all data under `data/$NAME/`, with "raw" data going in some kind of subdirectory. 
   (If it is a UD corpus, the standard UD name would be good, e.g. `data/coptic/UD_Coptic-Scriptorium`)
2. Ensure that it will be properly handled by the module `embur.commands.data`. 
   Put a script at `embur/scripts/$NAME_data_prep.py` that will take the dataset's native format and 
   write it out into `data/$NAME/converted`, if appropriate. 
3. Update `embur.language_configs` with the language's information.

If you'd like to add a language's Wikipedia dump, see [wiki-thresher](https://github.com/lgessler/wiki-thresher).
