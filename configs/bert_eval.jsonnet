local embedding_dim = std.parseInt(std.extVar("BERT_DIMS"));
local model_name = std.extVar("BERT_PATH");
local trainable = if std.parseInt(std.extVar("TRAINABLE")) == 1 then true else false;

{
    "dataset_reader" : {
        "type": "coptic_conllu",
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer_mismatched",
                "model_name": model_name
            }
        },
        "read_entities": true
    },
    "model": {
        "type": "entity_crf",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer_mismatched",
                    "model_name": model_name,
                    "train_parameters": trainable,
                    "last_layer_only": trainable
                },
            }
        },
        "dropout": 0.5,
        "encoder": {
            "type": "lstm",
            "input_size": embedding_dim,
            "hidden_size": embedding_dim,
            "num_layers": 1,
            "bidirectional": true
        }
    },
    "train_data_path": "data/coptic/converted/train",
    "validation_data_path": "data/coptic/converted/dev",
    "data_loader": {
        "batch_size": 32,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 3e-3,
            "parameter_groups": [
                [[".*transformer.*"], {"lr": 2e-5}]
            ]
        },
        "patience": 5,
        "num_epochs": 40
    }
}
