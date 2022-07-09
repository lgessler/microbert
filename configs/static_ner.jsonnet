local embedding_dim = std.parseInt(std.extVar("EMBEDDING_DIMS"));
local model_name = std.extVar("EMBEDDING_PATH");
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
        "type": "single_id",
        "lowercase_tokens": false
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
          "type": "embedding",
          "embedding_dim": embedding_dim,
          "pretrained_file": model_name,
          "trainable": trainable
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
    "batch_size": 16
  },
  "trainer": {
    "optimizer": {
      "type": "adamw",
      "lr": 0.001,
      "betas": [0.9, 0.999]
    },
    "validation_metric": "+f1-measure-overall",
    "num_epochs": 300,
    "grad_norm": 5.0,
    "patience": 50,
  }
}
