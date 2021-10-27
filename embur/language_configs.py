"""
Some functions to help build configurations for different parts of the experiments.
We don't put these into .libsonnet files in part because configuration overrides can
significantly modify e.g. the structure of the pretraining config (-x removes a key
from all dicts). Be smart in this file and do as much as possible to only write
data, not code!
"""

LANGUAGES = ["coptic"]

def get_pretrain_config(language, tokenizer_path, excluded_tasks):
    """
    Contains language-specific config for pretraining, which mostly has to do with dataset paths.

    To understand how these variables are used, see configs/bert_pretain.jsonnet.
    """
    mismatched_reader = {
        "type": "embur_conllu",
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer_mismatched",
                "model_name": tokenizer_path
            },
        }
    }

    language_config = {
        "coptic": {
            "train_data_paths": {
                "xpos": "data/coptic/converted/train",
                "mlm": "data/coptic/converted/train",
                "parser": "data/coptic/UD_Coptic-Scriptorium/cop_scriptorium-ud-train.conllu"
            },
            "dev_data_paths": {
                "xpos": "data/coptic/converted/dev",
                "mlm": "data/coptic/converted/dev",
                "parser": "data/coptic/UD_Coptic-Scriptorium/cop_scriptorium-ud-dev.conllu",
            },
            "readers": {
                "xpos": mismatched_reader,
                "mlm": mismatched_reader,
                "parser": mismatched_reader
            },
            "tokenizer_conllu_path": "data/coptic/converted/train"
        }
    }[language]

    for subconfig_name, subconfig in language_config.items():
        if subconfig_name in ["train_data_paths", "dev_data_paths", "readers"]:
            for excluded_key in excluded_tasks:
                language_config[subconfig_name] = {k: v for k, v in subconfig.items() if k != excluded_key}

    return language_config


def get_eval_config(language, model_name):
    """
    Contains language-specific config for evaluation, which mostly has to do with dataset paths.
    Note that each language has two keys under it: "training" for the training phase of the eval,
    and "testing" for the final evaluation.

    To understand how these variables are used, see configs/bert_eval.jsonnet.
    """
    return {
        "coptic": {
            "training": {
                "dataset_reader": {
                    "type": "embur_conllu",
                    "token_indexers": {
                        "tokens": {
                            "type": "pretrained_transformer_mismatched",
                            "model_name": model_name
                        }
                    }
                },
                "train_data_path": "data/coptic/UD_Coptic-Scriptorium/cop_scriptorium-ud-train.conllu",
                "validation_data_path": "data/coptic/UD_Coptic-Scriptorium/cop_scriptorium-ud-dev.conllu"
            },
            "testing": {
                "input_file": "data/coptic/UD_Coptic-Scriptorium/cop_scriptorium-ud-test.conllu"
            }
        }
    }[language]
