local embedding_dim = std.parseInt(std.extVar("BERT_DIMS"));
local model_name = std.extVar("BERT_PATH");
local trainable = if std.parseInt(std.extVar("TRAINABLE")) == 1 then true else false;
local train_data_path = std.extVar("train_data_path");
local validation_data_path = std.extVar("validation_data_path");

// Adapted from https://github.com/allenai/allennlp-models/blob/main/training_config/tagging/ner.jsonnet
{
  "dataset_reader": {
    "type": "conll2003",
    "tag_label": "ner",
    "convert_to_coding_scheme": null,
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer_mismatched",
        "model_name": model_name,
        "max_length": 512,
        "tokenizer_kwargs": {"max_length": 512}
      }
    }
  },
  "train_data_path": train_data_path,
  "validation_data_path": validation_data_path,
  "model": {
    "type": "crf_tagger",
    "label_encoding": "BIOUL",
    "constrain_crf_decoding": true,
    "calculate_span_f1": true,
    "dropout": 0.5,
    "include_start_end_transitions": false,
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
    "encoder": {
      "type": "lstm",
      "input_size": embedding_dim,
      "hidden_size": 200,
      "num_layers": 2,
      "dropout": 0.5,
      "bidirectional": true
    }
  },
  "data_loader": {
    "batch_size": 32
  },
  "trainer": {
    "optimizer": {
      "type": "adamw",
      "lr": 0.001,
      "betas": [0.9, 0.999]
    },
    "validation_metric": "+f1-measure-overall",
    "num_epochs": 1000,
    "grad_norm": 5.0,
    "patience": 100,
    "parameter_groups": [
      [[".*transformer_model.*"], {"lr": 5e-5}]
    ]
  }
}
