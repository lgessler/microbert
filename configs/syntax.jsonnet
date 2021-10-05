local embedding_dim = 100;
local pos_embedding_dim = 100;

{
    "dataset_reader" : {
        "type": "multitask",
        "readers": {
            "ud": {
                "type": "coptic_conllu",
            },
        }
    },
    "data_loader": {
        "type": "multitask",
        "scheduler": {
            "batch_size": 64,
        },
        "shuffle": true
    },
    "train_data_path": {
        "ud": "data/coptic/UD_Coptic-Scriptorium/cop_scriptorium-ud-train.conllu"
    },
    "validation_data_path": {
        "ud": "data/coptic/UD_Coptic-Scriptorium/cop_scriptorium-ud-dev.conllu"
    },
    "model": {
        "type": "multitask",
        "backbone": {
            "type": "static_embedding",
            "embedder": {
                "token_embedders": {
                    "tokens": {
                        "embedding_dim": embedding_dim,
                        "type": "embedding",
                        "trainable": true,
                        //"pretrained_file": "embeddings/coptic_50d.vec"
                    }
                }
            }
        },
        "heads": {
            "ud": {
                "type": "biaffine_parser",
                "embedding_dim": embedding_dim,
                "pos_tag_embedding": {
                    "embedding_dim": pos_embedding_dim,
                    "vocab_namespace": "xpos_tags",
                    //"sparse": true
                },
                "encoder": {
                    //"type": "stacked_bidirectional_lstm",
                    //"input_size": embedding_dim + pos_embedding_dim,
                    //"hidden_size": 200,
                    //"num_layers": 1,
                    //"recurrent_dropout_probability": 0.3,
                    //"use_highway": true
                    "type": "gru",
                    "input_size": embedding_dim + pos_embedding_dim,
                    "hidden_size": 200,
                    "num_layers": 2,
                    "bias": true,
                    "bidirectional": true
                },
                "use_mst_decoding_for_validation": true,
                "arc_representation_dim": 250,
                "tag_representation_dim": pos_embedding_dim,
                "dropout": 0.3,
                "input_dropout": 0.3,
                "initializer": {
                    "regexes": [
                        [".*projection.*weight", {"type": "xavier_uniform"}],
                        [".*projection.*bias", {"type": "zero"}],
                        [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
                        [".*tag_bilinear.*bias", {"type": "zero"}],
                        [".*weight_ih.*", {"type": "xavier_uniform"}],
                        [".*weight_hh.*", {"type": "orthogonal"}],
                        [".*bias_ih.*", {"type": "zero"}],
                        [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
                    ]
                },
            }
        }
    },
    "trainer": {
        //"optimizer": {
        //    "type": "dense_sparse_adam",
        //    "betas": [0.9, 0.9],
        //    "lr": 3e-3
        //},
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 3e-3
        },
        "patience": 5,
        "num_epochs": 60
    }
}
