local num_layers = std.parseInt(std.extVar("NUM_LAYERS"));
// Note this invariant: token embedding dim must be divisible by number of attention heads
local token_embedding_dim = std.parseInt(std.extVar("EMBEDDING_DIM"));
local num_attention_heads = std.parseInt(std.extVar("NUM_ATTENTION_HEADS"));
local embedding_dim = token_embedding_dim; //+ char_embedding_dim;
local pos_embedding_dim = 10;
local tokenizer_path = std.extVar("TOKENIZER_PATH");

// --------------------------------------------------------------------------------
// Reader setup
// --------------------------------------------------------------------------------
local readers = std.parseJson(std.extVar("readers"));

// --------------------------------------------------------------------------------
// Data path setup
// --------------------------------------------------------------------------------
local train_data_paths = std.parseJson(std.extVar("train_data_paths"));
local dev_data_paths = std.parseJson(std.extVar("dev_data_paths"));

// --------------------------------------------------------------------------------
// Head setup
// --------------------------------------------------------------------------------
local xpos_head = {
    "type": "xpos",
    "embedding_dim": embedding_dim,
    "use_crf": true
};
local mlm_head = {
  "type": "mlm",
  "embedding_dim": embedding_dim
};
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

local heads = (
  (if std.parseJson(std.extVar("XPOS")) then {"xpos": xpos_head} else {})
  + (if std.parseJson(std.extVar("MLM")) then {"mlm": mlm_head} else {})
  + (if std.parseJson(std.extVar("PARSER")) then {"parser": parser_head} else {})
);

local weights = (
  (if std.parseJson(std.extVar("XPOS")) then {"xpos": 0.1} else {})
  + (if std.parseJson(std.extVar("MLM")) then {"mlm": 0.8} else {})
  + (if std.parseJson(std.extVar("PARSER")) then {"parser": 0.1} else {})
);

// scheduling
local batch_size = 8;
local num_epochs = 120;
local instances_per_epoch = 10000;
local steps_per_epoch = instances_per_epoch / batch_size;
local plateau = {
    "type": "reduce_on_plateau",
    "factor": 0.5,
    "mode": "min",
    "patience": 2,
    "verbose": true,
    "min_lr": 5e-6
};

local slanted_triangular = {
    "type": "slanted_triangular",
    "num_epochs": num_epochs,
    "num_steps_per_epoch": steps_per_epoch
};


{
    "dataset_reader" : {
        "type": "multitask",
        "readers": readers
    },
    "data_loader": {
        "type": "multitask",
        "scheduler": {
            "type": "homogeneous_roundrobin",
            "batch_size": batch_size,
        },
        "shuffle": true,
        "instances_per_epoch": instances_per_epoch,
        "sampler": {
            "type": "weighted",
            "weights": weights
        }
    },
    "train_data_path": train_data_paths,
    "validation_data_path": dev_data_paths,
    "model": {
        "type": "multitask",
        "backbone": {
            "type": "bert",
            "embedding_dim": embedding_dim,
            "feedforward_dim": embedding_dim * 4,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "position_embedding_type": "relative_key",
            "tokenizer_path": tokenizer_path
        },
        "heads": heads
    },
    "trainer": {
        "type": "mtl",
        "optimizer": {
            "type": "adamw",
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
        "learning_rate_scheduler": slanted_triangular, //plateau,
        "patience": 100,
        "num_epochs": num_epochs,
        "validation_metric": "-mlm_perplexity",
        "callbacks": [
            {
                "type": "console_logger"
            },
            {
                "type": "tensorboard"
            },
            //{
            //    "type": "should_validate_callback",
            //    "validation_start": 100,
            //    "validation_interval": 5
            //}
        ]
    }
}
