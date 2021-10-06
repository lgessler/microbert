def get_config(language, tokenizer_path):
    mismatched_reader = {
        "type": "coptic_conllu",
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
