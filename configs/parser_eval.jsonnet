local embedding_dim = std.parseInt(std.extVar("BERT_DIMS"));
local model_name = std.extVar("BERT_PATH");
local trainable = if std.parseInt(std.extVar("TRAINABLE")) == 1 then true else false;
local train_data_path = std.extVar("train_data_path");
local validation_data_path = std.extVar("validation_data_path");
local dataset_reader = std.parseJson(std.extVar("dataset_reader"));
local pos_embedding_dim = 0;

{
    "dataset_reader": {
      "type": "universal_dependencies",
      "use_language_specific_pos": true,
      "token_indexers": {
        "tokens": {
          "type": "pretrained_transformer_mismatched",
          "model_name": model_name,
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
            "type": "pretrained_transformer_mismatched",
            "model_name": model_name,
            "train_parameters": trainable,
            "last_layer_only": false
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
      "arc_representation_dim": 500,
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
    "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "batch_size": 16
      },
      "batches_per_epoch": 400
    },
    "trainer": {
      "num_epochs": 500,
      "grad_clipping": 5.0,
      "patience": 40,
      "validation_metric": "+LAS",
      "optimizer": {
        "type": "adamw",
        "betas": [0.9, 0.999],
        "lr": 1e-3,
        "parameter_groups": [
          [[".*transformer_model.*"], {"lr": 1e-5}]
        ]
      }
    }
}
