local embedding_dim = std.parseInt(std.extVar("BERT_DIMS"));
local num_layers = std.parseInt(std.extVar("BERT_LAYERS"));
local num_attention_heads = std.parseInt(std.extVar("BERT_HEADS"));
local model_name = std.extVar("BERT_PATH");
local train_data_path = std.extVar("train_data_path");
local validation_data_path = std.extVar("validation_data_path");
local dataset_reader = std.parseJson(std.extVar("dataset_reader"));
local tokenizer_path = std.extVar("TOKENIZER_PATH");
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
            "type": "bert",
            "embedding_dim": embedding_dim,
            "feedforward_dim": embedding_dim * 4,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            // Actually, we use ALiBi (https://github.com/ofirpress/attention_with_linear_biases)
            // (note that we are using a custom version of the transformers package with alibi hacked onto BERT)
            // So, the position embedding type is ignored and the position_embedding_dim is actually a max seq length
            "position_embedding_dim": 512,
            "position_embedding_type": "sinusoidal",
            "tokenizer_path": tokenizer_path
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
