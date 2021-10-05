local num_layers = std.parseInt(std.extVar("NUM_LAYERS"));
// Note this invariant: token embedding dim must be divisible by number of attention heads
local token_embedding_dim = std.parseInt(std.extVar("EMBEDDING_DIM"));
local num_attention_heads = std.parseInt(std.extVar("NUM_ATTENTION_HEADS"));
local embedding_dim = token_embedding_dim; //+ char_embedding_dim;
local pos_embedding_dim = 10;
local tokenizer_path = std.extVar("TOKENIZER_PATH");


local mismatched_reader = {
    "type": "coptic_conllu",
    "token_indexers": {
        "tokens": {
            "type": "pretrained_transformer_mismatched",
            "model_name": tokenizer_path
        },
    }
};

{
    "dataset_reader" : {
        "type": "multitask",
        "readers": {
            "xpos": mismatched_reader,
            "mlm": {
                "type": "coptic_conllu",
                "token_indexers": {
                    "tokens": {
                        "type": "pretrained_transformer_mismatched",
                        "model_name": tokenizer_path
                    },
                },
                "seg_threshold": false
            },
            "biaffine_parser": mismatched_reader
        }
    },
    "data_loader": {
        "type": "multitask",
        "scheduler": {
            "type": "homogeneous_repeated_roundrobin",
            "batch_size": 32,
        },
        "shuffle": true
    },
    "train_data_path": {
        "xpos": "data/coptic/converted/train",
        "mlm": "data/coptic/converted/train",
        "biaffine_parser": "data/coptic/UD_Coptic-Scriptorium/cop_scriptorium-ud-train.conllu"
    },
    "validation_data_path": {
        "xpos": "data/coptic/converted/dev",
        "mlm": "data/coptic/converted/dev",
        "biaffine_parser": "data/coptic/UD_Coptic-Scriptorium/cop_scriptorium-ud-dev.conllu"
    },
    "model": {
        "type": "multitask",
        "backbone": {
            "type": "bert",
            "embedding_dim": embedding_dim,
            "feedforward_dim": embedding_dim * 4,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "position_embedding_dim": 24,
            "position_embedding_type": "sinusoidal",
            "tokenizer_path": tokenizer_path
        },
        "heads": {
            "xpos": {
                "type": "xpos",
                "embedding_dim": embedding_dim,
                "use_crf": true
            },
            "mlm": {
                "type": "mlm",
                "embedding_dim": embedding_dim
            },
            "biaffine_parser": {
                "type": "biaffine_parser",
                "embedding_dim": embedding_dim, // 100,
                "encoder": {
                    "type": "stacked_bidirectional_lstm",
                    "input_size": embedding_dim + pos_embedding_dim,
                    "hidden_size": (embedding_dim + pos_embedding_dim) / 2,
                    "num_layers": 1,
                    "recurrent_dropout_probability": 0.3,
                    "use_highway": true
                },
                "pos_tag_embedding": {
                    "embedding_dim": pos_embedding_dim,
                    "vocab_namespace": "xpos_tags",
                    "sparse": false
                },
                "use_mst_decoding_for_validation": true,
                "arc_representation_dim": embedding_dim * 5, // 500
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
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 3e-3,
            "betas": [0.9, 0.9],
            "weight_decay": 0.05
            //"type": "multi",
            //"optimizers": {
            //    "biaffine_parser": {"type": "dense_sparse_adam", "betas": [0.9, 0.9]},
            //    "default": {"type": "huggingface_adamw", "lr": 3e-3},
            //},
            //"parameter_groups": [
            //    [[".*biaffine_parser.*"], {"optimizer_name": "biaffine_parser"}]
            //]
        },
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 2,
            "verbose": true,
            "min_lr": 5e-6
        },
        "patience": 15,
        "num_epochs": 200,
        "validation_metric": "-mlm_perplexity"
    }
}
