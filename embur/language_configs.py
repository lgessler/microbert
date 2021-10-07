
def get_pretrain_config(language, tokenizer_path):
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
    return {
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
            }
        }
    }[language]


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
