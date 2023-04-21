local language = std.extVar("LANGUAGE");
local embedding_dim = std.parseInt(std.extVar("BERT_DIMS"));
local model_name = std.extVar("BERT_PATH");
local trainable = if std.parseInt(std.extVar("TRAINABLE")) == 1 then true else false;
local train_data_path = std.extVar("train_data_path");
local validation_data_path = std.extVar("validation_data_path");
local language_code_index = import 'lib/language_code.libsonnet';
local stanza_do_not_retokenize = import 'lib/stanza_do_not_retokenize.libsonnet';
local stanza_no_mwt = import 'lib/stanza_no_mwt.libsonnet';

local do_sla = if std.parseInt(std.extVar("SLA")) == 1 then true else false;
local embedder_type = if do_sla then "pretrained_transformer_mismatched_with_dep_att_mask" else "pretrained_transformer_mismatched";
local indexer = if do_sla then {
          "type": "pretrained_transformer_mismatched_with_dep_att_mask",
          "stanza_language": language_code_index[language],
          "stanza_use_mwt": if std.member(stanza_no_mwt, language) then false else true,
          "allow_retokenization": false,
          "model_name": model_name,
          "max_distance": 4,
          "max_length": 512,
          "tokenizer_kwargs": {"max_length": 512}
        } else {
          "type": "pretrained_transformer_mismatched",
          "model_name": model_name,
          "max_length": 512,
          "tokenizer_kwargs": {"max_length": 512}
        };


// Adapted from https://github.com/allenai/allennlp-models/blob/main/training_config/tagging/ner.jsonnet
{
  "dataset_reader": {
    "type": "conll2003",
    "tag_label": "ner",
    "convert_to_coding_scheme": null,
    "token_indexers": {
      "tokens": indexer,
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
          "type": embedder_type,
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
    "parameter_groups": [
      [[".*transformer_model.*"], {"lr": 1e-5}]
    ]
  }
}
