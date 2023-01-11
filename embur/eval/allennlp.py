import argparse
import json
import logging
import os
from logging import getLogger
from shutil import rmtree
from typing import Dict, Any

from torch.serialization import mkdtemp

from allennlp.commands.train import train_model_from_file
from allennlp.common import logging as common_logging
from allennlp.common.util import prepare_environment
from allennlp.data import DataLoader
from allennlp.models import load_archive
from allennlp.training.util import evaluate

from transformers import AutoModel

from embur.language_configs import get_formatted_wikiann_path

logger = getLogger(__name__)

# This is an older version of the function that'll tolerate dicts for input files
# function is from (from https://github.com/allenai/allennlp/commits/main/allennlp/commands/evaluate.py)
def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    common_logging.FILE_FRIENDLY_LOGGING = args.file_friendly_logging

    # Disable some of the more verbose logging statements
    logging.getLogger("allennlp.common.params").disabled = True
    logging.getLogger("allennlp.nn.initializers").disabled = True
    logging.getLogger("allennlp.modules.token_embedders.embedding").setLevel(logging.INFO)

    # Load from archive
    archive = load_archive(
        args.archive_file,
        weights_file=args.weights_file,
        cuda_device=args.cuda_device,
        overrides=args.overrides,
    )
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    # Load the evaluation data

    dataset_reader = archive.validation_dataset_reader

    evaluation_data_path = args.input_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)

    data_loader_params = config.pop("validation_data_loader", None)
    if data_loader_params is None:
        data_loader_params = config.pop("data_loader")
    if args.batch_size:
        data_loader_params["batch_size"] = args.batch_size
    data_loader = DataLoader.from_params(
        params=data_loader_params, reader=dataset_reader, data_path=evaluation_data_path
    )

    embedding_sources = json.loads(args.embedding_sources_mapping) if args.embedding_sources_mapping else {}

    if args.extend_vocab:
        logger.info("Vocabulary is being extended with test instances.")
        model.vocab.extend_from_instances(instances=data_loader.iter_instances())
        model.extend_embedder_vocab(embedding_sources)

    data_loader.index_with(model.vocab)

    metrics = evaluate(
        model,
        data_loader,
        args.cuda_device,
        args.batch_weight_key,
        output_file=args.output_file,
        predictions_output_file=args.predictions_output_file,
    )

    logger.info("Finished evaluating.")

    return metrics


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


def _eval_dir(config, task, model_name):
    os.makedirs('evals', exist_ok=True)
    return f"evals/{config.language}/{task}_{model_name.replace('/', '__')}{'_ft' if config.finetune else ''}"


def evaluate_parser(config, bert_model_path):
    """
    Given a language and a pretrained BERT model (or something API compatible with it),
    """
    eval_dir = _eval_dir(config, "parse", bert_model_path)
    if os.path.exists(eval_dir):
        rmtree(eval_dir)

    logger.info("Training for evaluation")
    bert_model = AutoModel.from_pretrained(bert_model_path)
    language_config = config.parser_eval_language_config

    os.environ["BERT_DIMS"] = str(bert_model.config.hidden_size)
    logger.info(f"Loaded BERT model from '{bert_model_path}': {bert_model}")
    os.environ["BERT_PATH"] = bert_model_path
    os.environ["TRAINABLE"] = "1" if config.finetune else "0"
    for k, v in language_config["training"].items():
        os.environ[k] = json.dumps(v) if isinstance(v, dict) else v

    overrides = []
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


def evaluate_parser_static(config):
    """
    Given a language and a pretrained BERT model (or something API compatible with it),
    """
    eval_dir = _eval_dir(config, "parse", config.word2vec_file)
    if os.path.exists(eval_dir):
        rmtree(eval_dir)

    logger.info("Training for evaluation")
    os.environ["EMBEDDING_DIMS"] = str(config.embedding_dim)
    os.environ["EMBEDDING_PATH"] = config.word2vec_file
    os.environ["TRAINABLE"] = "1" if config.finetune else "0"
    language_config = config.parser_eval_language_config
    for k, v in language_config["training"].items():
        os.environ[k] = json.dumps(v) if isinstance(v, dict) else v

    overrides = []
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


def evaluate_ner(config, bert_model_path):
    """
    Given a language and a pretrained BERT model (or something API compatible with it),
    """
    eval_dir = _eval_dir(config, "ner", bert_model_path)
    if os.path.exists(eval_dir):
        rmtree(eval_dir)

    logger.info("Training for evaluation")
    os.environ["BERT_DIMS"] = str(config.embedding_dim)
    os.environ["BERT_PATH"] = bert_model_path
    os.environ["TRAINABLE"] = "1" if config.finetune else "0"
    os.environ["train_data_path"] = get_formatted_wikiann_path(config.language, "train")
    os.environ["validation_data_path"] = get_formatted_wikiann_path(config.language, "dev")

    overrides = []
    if config.debug:
        overrides.append('"trainer.num_epochs": 1')
    if len(overrides) > 0:
        overrides = "{" + ", ".join(overrides) + "}"
    else:
        overrides = ""

    train_model_from_file(config.ner_eval_jsonnet, eval_dir, overrides=overrides)

    logger.info("Evaluating")
    args = eval_args(eval_dir, get_formatted_wikiann_path(config.language, "test"))
    metrics = evaluate_from_args(args)
    logger.info(metrics)
    return None, metrics


def evaluate_ner_static(config):
    """
    Given a language and a pretrained BERT model (or something API compatible with it),
    """
    eval_dir = _eval_dir(config, "ner", config.word2vec_file)
    if os.path.exists(eval_dir):
        rmtree(eval_dir)

    logger.info("Training for evaluation")
    os.environ["EMBEDDING_DIMS"] = str(config.embedding_dim)
    os.environ["EMBEDDING_PATH"] = config.word2vec_file
    os.environ["TRAINABLE"] = "1" if config.finetune else "0"
    os.environ["train_data_path"] = get_formatted_wikiann_path(config.language, "train")
    os.environ["validation_data_path"] = get_formatted_wikiann_path(config.language, "dev")

    overrides = []
    if config.debug:
        overrides.append('"trainer.num_epochs": 1')
    if len(overrides) > 0:
        overrides = "{" + ", ".join(overrides) + "}"
    else:
        overrides = ""

    train_model_from_file(config.ner_eval_jsonnet, eval_dir, overrides=overrides)

    logger.info("Evaluating")
    args = eval_args(eval_dir, get_formatted_wikiann_path(config.language, "test"))
    metrics = evaluate_from_args(args)
    logger.info(metrics)
    return None, metrics
