local embedding_dim = std.parseInt(std.extVar("EMBEDDING_DIMS"));
local model_name = std.extVar("EMBEDDING_PATH");
local trainable = if std.parseInt(std.extVar("TRAINABLE")) == 1 then true else false;
local train_data_path = std.extVar("train_data_path");
local validation_data_path = std.extVar("validation_data_path");
local dataset_reader = std.parseJson(std.extVar("dataset_reader"));
local pos_embedding_dim = 0;

local validation_data_loader = {
  "type": "multiprocess",
  "batch_sampler": {
    "type": "bucket",
    "batch_size": 16
  }
};
local data_loader = validation_data_loader + {"batches_per_epoch": 200};

{
    "dataset_reader": {
      "type": "universal_dependencies",
      "use_language_specific_pos": true,
      "token_indexers": {
        "tokens": {
          "type": "single_id",
          "lowercase_tokens": false
        }
      }
    },
    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,
    "model": {
      "type": "biaffine_parser",
      "text_field_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "embedding",
            "embedding_dim": embedding_dim,
            "pretrained_file": model_name,
            "trainable": trainable
          }
        }
      },
      //"pos_tag_embedding":{
      //  "embedding_dim": pos_embedding_dim,
      //  "vocab_namespace": "pos",
      //  "sparse": true
      //},
      "encoder": {
        "type": "stacked_bidirectional_lstm",
        "input_size": embedding_dim + pos_embedding_dim,
        "hidden_size": 400,
        "num_layers": 3,
        "recurrent_dropout_probability": 0.3,
        "use_highway": true
      },
      "use_mst_decoding_for_validation": true,
      "arc_representation_dim": 100,
      "tag_representation_dim": 100,
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
    },
    "data_loader": data_loader,
    "validation_data_loader": validation_data_loader,
    "trainer": {
      "num_epochs": 1000,
      "grad_clipping": 5.0,
      "patience": 100,
      "validation_metric": "+LAS",
      "optimizer": {
        "type": "adamw",
        "betas": [0.9, 0.999],
        "lr": 1e-3
      }
    }
}
