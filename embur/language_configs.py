"""
Some functions to help build configurations for different parts of the experiments.
We don't put these into .libsonnet files in part because configuration overrides can
significantly modify e.g. the structure of the pretraining config (-x removes a key
from all dicts). Be smart in this file and do as much as possible to only write
data, not code!
"""

LANGUAGES = ["coptic", "maltese", "wolof", "uyghur", "greek", "latin"]

def _std_pretrain_config(mismatched_reader, language, treebank_name, conllu_name, ssplit_type="_punct"):
    return {
        "train_data_paths": {
            "xpos": f"data/{language}/{treebank_name}/{conllu_name}-train.conllu",
            "mlm": f"data/{language}/converted{ssplit_type}/train",
            "parser": f"data/{language}/{treebank_name}/{conllu_name}-train.conllu"
        },
        "dev_data_paths": {
            "xpos": f"data/{language}/{treebank_name}/{conllu_name}-dev.conllu",
            "mlm": f"data/{language}/converted{ssplit_type}/dev",
            "parser": f"data/{language}/{treebank_name}/{conllu_name}-dev.conllu"
        },
        "readers": {
            "xpos": mismatched_reader,
            "mlm": mismatched_reader,
            "parser": mismatched_reader
        },
        "tokenizer_conllu_path": f"data/{language}/converted{ssplit_type}/train"
    }


def _std_eval_config(model_name, language, treebank_name, conllu_name):
    return {
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
            "train_data_path": f"data/{language}/{treebank_name}/{conllu_name}-train.conllu",
            "validation_data_path": f"data/{language}/{treebank_name}/{conllu_name}-dev.conllu"
        },
        "testing": {
            "input_file": f"data/{language}/{treebank_name}/{conllu_name}-dev.conllu"
        }
    }


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
        },
        "maltese": _std_pretrain_config(mismatched_reader, "maltese", "UD_Maltese-MUDT", "mt_mudt-ud", ssplit_type="_punct"),
        "wolof": _std_pretrain_config(mismatched_reader, "wolof", "UD_Wolof-WTB", "wo_wtb-ud", ssplit_type="_punct"),
        "uyghur": _std_pretrain_config(mismatched_reader, "uyghur", "UD_Uyghur-UDT", "ug_udt-ud", ssplit_type="_punct"),
        "greek": _std_pretrain_config(mismatched_reader, "greek", "UD_Ancient_Greek-PROIEL", "grc_proiel-ud", ssplit_type="")
    }[language]

    for subconfig_name, subconfig in language_config.items():
        if subconfig_name in ["train_data_paths", "dev_data_paths", "readers"]:
            language_config[subconfig_name] = {k: v for k, v in subconfig.items() if k not in excluded_tasks}

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
                "input_file": "data/coptic/UD_Coptic-Scriptorium/cop_scriptorium-ud-dev.conllu"
            }
        },
        "maltese": _std_eval_config(model_name, "maltese", "UD_Maltese-MUDT", "mt_mudt-ud"),
        "wolof": _std_eval_config(model_name, "wolof", "UD_Wolof-WTB", "wo_wtb-ud"),
        "uyghur": _std_eval_config(model_name, "uyghur", "UD_Uyghur-UDT", "ug_udt-ud"),
        "greek": _std_eval_config(model_name, "greek", "UD_Ancient_Greek-PROIEL", "grc_proiel-ud")
    }[language]
