⚠️ **NOTE**: If you want to train a MicroBERT for your language, please see [lgessler/microbert2](https://github.com/lgessler/microbert2).

# Introduction

[MicroBERT](https://aclanthology.org/2022.mrl-1.9/) is a BERT variant intended for training **monolingual** models for **low-resource** languages by 
**reducing model sizes** and using **multitask learning** on part of speech tagging and dependency parsing 
in addition to the usual masked language modeling.

For more information, please see [our paper](https://aclanthology.org/2022.mrl-1.9/).
If you'd like to cite our work, please use the following citation:

```
@inproceedings{gessler-zeldes-2022-microbert,
    title = "{M}icro{BERT}: Effective Training of Low-resource Monolingual {BERT}s through Parameter Reduction and Multitask Learning",
    author = "Gessler, Luke  and
      Zeldes, Amir",
    booktitle = "Proceedings of the The 2nd Workshop on Multi-lingual Representation Learning (MRL)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.mrl-1.9",
    pages = "86--99",
}
```

# Pretrained Models
The following pretrained models are available.
Note that each model's suffix indicates the tasks that were used to pretrain it: masked language modeling (`m`),
XPOS tagging (`x`), or dependency parsing (`p`).

 - [`microbert-ancient-greek-m`](https://huggingface.co/lgessler/microbert-ancient-greek-m)
 - [`microbert-ancient-greek-mx`](https://huggingface.co/lgessler/microbert-ancient-greek-mx)
 - [`microbert-ancient-greek-mxp`](https://huggingface.co/lgessler/microbert-ancient-greek-mxp)
 - [`microbert-coptic-m`](https://huggingface.co/lgessler/microbert-coptic-m)
 - [`microbert-coptic-mx`](https://huggingface.co/lgessler/microbert-coptic-mx)
 - [`microbert-coptic-mxp`](https://huggingface.co/lgessler/microbert-coptic-mxp)
 - [`microbert-indonesian-m`](https://huggingface.co/lgessler/microbert-indonesian-m)
 - [`microbert-indonesian-mx`](https://huggingface.co/lgessler/microbert-indonesian-mx)
 - [`microbert-indonesian-mxp`](https://huggingface.co/lgessler/microbert-indonesian-mxp)
 - [`microbert-maltese-m`](https://huggingface.co/lgessler/microbert-maltese-m)
 - [`microbert-maltese-mx`](https://huggingface.co/lgessler/microbert-maltese-mx)
 - [`microbert-maltese-mxp`](https://huggingface.co/lgessler/microbert-maltese-mxp)
 - [`microbert-uyghur-m`](https://huggingface.co/lgessler/microbert-uyghur-m)
 - [`microbert-uyghur-mx`](https://huggingface.co/lgessler/microbert-uyghur-mx)
 - [`microbert-uyghur-mxp`](https://huggingface.co/lgessler/microbert-uyghur-mxp)
 - [`microbert-tamil-m`](https://huggingface.co/lgessler/microbert-tamil-m)
 - [`microbert-tamil-mx`](https://huggingface.co/lgessler/microbert-tamil-mx)
 - [`microbert-tamil-mxp`](https://huggingface.co/lgessler/microbert-tamil-mxp)
 - [`microbert-wolof-m`](https://huggingface.co/lgessler/microbert-wolof-m)
 - [`microbert-wolof-mx`](https://huggingface.co/lgessler/microbert-wolof-mx)
 - [`microbert-wolof-mxp`](https://huggingface.co/lgessler/microbert-wolof-mxp)


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

Please don't hesitate to email me (`lukegessler@gmail.com`) if you have any questions.
