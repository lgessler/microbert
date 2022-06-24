import argparse
import json
import os
from logging import getLogger
from torch.serialization import mkdtemp

from allennlp.commands.evaluate import evaluate_from_args
from allennlp.commands.train import train_model_from_file
from transformers import BertModel

logger = getLogger(__name__)


def eval_args(serialization_dir, input_file):
    args = argparse.Namespace()
    args.file_friendly_logging = False
    args.weights_file = None
    args.overrides = None
    args.embedding_sources_mapping = None
    args.extend_vocab = False
    args.batch_weight_key = None
    args.batch_size = None
    args.archive_file = f"{serialization_dir}/model.tar.gz"
    args.input_file = input_file
    args.output_file = serialization_dir + "/metrics"
    args.predictions_output_file = serialization_dir + "/predictions"
    args.cuda_device = 0
    args.auto_names = "NONE"
    return args


def evaluate_allennlp(config):
    """
    Given a language and a pretrained BERT model (or something API compatible with it),
    """
    with mkdtemp() as eval_dir:
        logger.info("Training for evaluation")
        bert_model = BertModel.from_pretrained(config.bert_model_name)
        language_config = config.parser_eval_language_config

        os.environ["BERT_DIMS"] = str(bert_model.config.hidden_size)
        os.environ["BERT_PATH"] = config.bert_model_name
        for k, v in language_config["training"].items():
            os.environ[k] = json.dumps(v) if isinstance(v, dict) else v

        overrides = []
        if config.finetune:
            overrides.append('"model.text_field_embedder.token_embedders.tokens.train_parameters": true')
        if config.debug:
            overrides.append('"trainer.num_epochs": 1')
        if len(overrides) > 0:
            overrides = "{" + ", ".join(overrides) + "}"
        else:
            overrides = ""

        train_model_from_file(config.parser_eval_jsonnet, eval_dir, overrides=overrides)

        logger.info("Evaluating")
        args = eval_args(eval_dir, language_config["testing"]["input_file"])
        metrics = evaluate_from_args(args)
    logger.info(metrics)
    return None, metrics


def evaluate_allennlp_static(config):
    """
    Given a language and a pretrained BERT model (or something API compatible with it),
    """
    with mkdtemp() as eval_dir:
        logger.info("Training for evaluation")
        os.environ["EMBEDDING_DIMS"] = str(config.embedding_dim)
        os.environ["EMBEDDING_PATH"] = config.word2vec_file
        for k, v in config.language_config.language_config["training"].items():
            os.environ[k] = json.dumps(v) if isinstance(v, dict) else v

        overrides = []
        if config.finetune:
            overrides.append('"model.text_field_embedder.token_embedders.tokens.train_parameters": true')
        if config.debug:
            overrides.append('"trainer.num_epochs": 1')
        if len(overrides) > 0:
            overrides = "{" + ", ".join(overrides) + "}"
        else:
            overrides = ""

        train_model_from_file(config.parser_eval_jsonnet, eval_dir, overrides=overrides)

        logger.info("Evaluating")
        args = eval_args(eval_dir, config.language_config["testing"]["input_file"])
        metrics = evaluate_from_args(args)
    logger.info(metrics)
    return None, metrics
