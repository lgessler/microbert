local embedding_dim = std.parseInt(std.extVar("BERT_DIMS"));
local model_name = std.extVar("BERT_PATH");
local trainable = if std.parseInt(std.extVar("TRAINABLE")) == 1 then true else false;
local train_data_path = std.extVar("train_data_path");
local validation_data_path = std.extVar("validation_data_path");
local dataset_reader = std.parseJson(std.extVar("dataset_reader"));
local pos_embedding_dim = 10;

local parser_head = {
    "type": "biaffine_parser",
    "embedding_dim": embedding_dim,
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
    "arc_representation_dim": embedding_dim * 5,
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
    }
};

{
    "dataset_reader": {
        "type": "multitask",
        "readers": {"parser": dataset_reader}
    },
    "model": {
        "type": "multitask",
        "backbone": {
            "type": "pretrained_bert",
            "bert_model": model_name
        },
        "heads": {"parser": parser_head}
    },
    "train_data_path": {"parser": train_data_path},
    "validation_data_path": {"parser": validation_data_path},
    "data_loader": {
        "type": "multitask",
        "scheduler": {
            "batch_size": 16,
        },
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
