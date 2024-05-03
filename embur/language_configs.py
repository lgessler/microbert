"""
Some functions to help build configurations for different parts of the experiments.
We don't put these into .libsonnet files in part because configuration overrides can
significantly modify e.g. the structure of the pretraining config (-x removes a key
from all dicts). Be smart in this file and do as much as possible to only write
data, not code!
"""

LANGUAGES = ["coptic", "maltese", "wolof", "uyghur", "greek", "indonesian", "tamil", "english", "nahuatl"]


def _std_pretrain_config(mismatched_reader, language, treebank_name, conllu_name, ssplit_type="_punct"):
    return {
        "train_data_paths": {
            "xpos": f"data/{language}/{treebank_name}/{conllu_name}-train.conllu",
            "mlm": f"data/{language}/converted{ssplit_type}/train",
            "parser": f"data/{language}/{treebank_name}/{conllu_name}-train.conllu",
        },
        "dev_data_paths": {
            "xpos": f"data/{language}/{treebank_name}/{conllu_name}-dev.conllu",
            "mlm": f"data/{language}/converted{ssplit_type}/dev",
            "parser": f"data/{language}/{treebank_name}/{conllu_name}-dev.conllu",
        },
        "readers": {"xpos": mismatched_reader, "mlm": mismatched_reader, "parser": mismatched_reader},
        "tokenizer_conllu_path": f"data/{language}/converted{ssplit_type}/train",
    }


def _std_eval_config(model_name, language, treebank_name, conllu_name):
    return {
        "training": {
            "dataset_reader": {
                "type": "embur_conllu",
                "token_indexers": {"tokens": {"type": "pretrained_transformer_mismatched", "model_name": model_name}},
            },
            "train_data_path": f"data/{language}/{treebank_name}/{conllu_name}-train.conllu",
            "validation_data_path": f"data/{language}/{treebank_name}/{conllu_name}-dev.conllu",
        },
        "testing": {"input_file": f"data/{language}/{treebank_name}/{conllu_name}-dev.conllu"},
    }


def get_pretrain_config(language, tokenizer_path, tasks):
    """
    Contains language-specific config for pretraining, which mostly has to do with dataset paths.

    To understand how these variables are used, see configs/bert_pretain.jsonnet.
    """
    mismatched_reader = {
        "type": "embur_conllu",
        "huggingface_tokenizer_path": tokenizer_path,
        "token_indexers": {
            "tokens": {"type": "pretrained_transformer_mismatched", "model_name": tokenizer_path},
        },
    }

    language_config = {
        "coptic": {
            "train_data_paths": {
                "xpos": "data/coptic/converted/train",
                "mlm": "data/coptic/converted/train",
                "parser": "data/coptic/UD_Coptic-Scriptorium/cop_scriptorium-ud-train.conllu",
            },
            "dev_data_paths": {
                "xpos": "data/coptic/converted/dev",
                "mlm": "data/coptic/converted/dev",
                "parser": "data/coptic/UD_Coptic-Scriptorium/cop_scriptorium-ud-dev.conllu",
            },
            "readers": {"xpos": mismatched_reader, "mlm": mismatched_reader, "parser": mismatched_reader},
            "tokenizer_conllu_path": "data/coptic/converted/train",
        },
        "english": _std_pretrain_config(
          mismatched_reader, "english", "UD_English-GUM", "en_ewt-ud", ssplit_type=""
        ),
        "maltese": _std_pretrain_config(
            mismatched_reader, "maltese", "UD_Maltese-MUDT", "mt_mudt-ud", ssplit_type="_punct"
        ),
        "wolof": _std_pretrain_config(mismatched_reader, "wolof", "UD_Wolof-WTB", "wo_wtb-ud", ssplit_type="_punct"),
        "uyghur": _std_pretrain_config(mismatched_reader, "uyghur", "UD_Uyghur-UDT", "ug_udt-ud", ssplit_type="_punct"),
        "greek": _std_pretrain_config(
            mismatched_reader, "greek", "UD_Ancient_Greek-PROIEL", "grc_proiel-ud", ssplit_type=""
        ),
        "indonesian": _std_pretrain_config(
            mismatched_reader, "indonesian", "UD_Indonesian-GSD", "id_gsd-ud", ssplit_type="_punct"
        ),
        "tamil": _std_pretrain_config(mismatched_reader, "tamil", "UD_Tamil-TTB", "ta_ttb-ud", ssplit_type="_punct"),
        "nahuatl": None,
    }[language]

    for subconfig_name, subconfig in language_config.items():
        if subconfig_name in ["train_data_paths", "dev_data_paths", "readers"]:
            language_config[subconfig_name] = {k: v for k, v in subconfig.items() if k in tasks}

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
                        "tokens": {"type": "pretrained_transformer_mismatched", "model_name": model_name}
                    },
                },
                "train_data_path": "data/coptic/UD_Coptic-Scriptorium/cop_scriptorium-ud-train.conllu",
                "validation_data_path": "data/coptic/UD_Coptic-Scriptorium/cop_scriptorium-ud-dev.conllu",
            },
            "testing": {"input_file": "data/coptic/UD_Coptic-Scriptorium/cop_scriptorium-ud-dev.conllu"},
        },
        "english": _std_eval_config(model_name, "english", "UD_English-EWT", "en_ewt-ud"),
        "maltese": _std_eval_config(model_name, "maltese", "UD_Maltese-MUDT", "mt_mudt-ud"),
        "wolof": _std_eval_config(model_name, "wolof", "UD_Wolof-WTB", "wo_wtb-ud"),
        "uyghur": _std_eval_config(model_name, "uyghur", "UD_Uyghur-UDT", "ug_udt-ud"),
        "greek": _std_eval_config(model_name, "greek", "UD_Ancient_Greek-PROIEL", "grc_proiel-ud"),
        "indonesian": _std_eval_config(model_name, "indonesian", "UD_Indonesian-GSD", "id_gsd-ud"),
        "tamil": _std_eval_config(model_name, "tamil", "UD_Tamil-TTB", "ta_ttb-ud"),
        "nahuatl": {
            "training": {
                "dataset_reader": {
                    "type": "embur_conllu",
                    "token_indexers": {"tokens": {"type": "pretrained_transformer_mismatched", "model_name": model_name}},
                },
                "train_data_path": f"data/nahuatl/nhi_itml-ud-test.conllu",
                "validation_data_path": f"data/nahuatl/nhi_itml-ud-test.conllu",
            },
            "testing": {"input_file": f"data/nahuatl/Book_05_-_The_Omens.conllu"},
        }

    }[language]


def get_wikiann_path(language):
    paths = {
        "maltese": "data/maltese/wikiann-mt.bio",
        "uyghur": "data/uyghur/wikiann-ug.bio",
        "wolof": "data/wolof/wikiann-wo.bio",
        "tamil": "data/tamil/wikiann-ta.bio",
        "indonesian": "data/indonesian/wikiann-id.bio",
    }
    if language not in paths:
        raise ValueError(f"Language not available for NER: {language}")
    return paths[language]


def get_formatted_wikiann_path(language, split):
    if split not in ["train", "dev", "test"]:
        raise ValueError("Split must be one of train, dev, test")
    paths = {
        "maltese": f"data/maltese/wikiann-mt_{split}.bio",
        "uyghur": f"data/uyghur/wikiann-ug_{split}.bio",
        "wolof": f"data/wolof/wikiann-wo_{split}.bio",
        "tamil": f"data/tamil/wikiann-ta_{split}.bio",
        "indonesian": f"data/indonesian/wikiann-id_{split}.bio",
    }
    if language not in paths:
        raise ValueError(f"Language not available for NER: {language}")
    return paths[language]
