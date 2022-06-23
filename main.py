import argparse
import json
import os
import shutil
from logging import getLogger

import click
from allennlp.commands.evaluate import evaluate_from_args
from allennlp.commands.train import train_model_from_file
from allennlp.common.util import import_module_and_submodules
from filelock import FileLock
from rich import print
from torch.serialization import mkdtemp
from transformers import BertModel

import embur
from embur.config import Config, TASKS, TOKENIZATION_TYPES
from embur.dataset_reader import read_conllu_files
from embur.diaparser import evaluate as dp_evaluate
from embur.diaparser import train as dp_train
from embur.language_configs import LANGUAGES, get_eval_config, get_pretrain_config
from embur.tokenizers import train_tokenizer

import_module_and_submodules("allennlp_models")


logger = getLogger(__name__)


def _prepare_bert_pretrain_env_vars(config):
    os.environ["TOKENIZER_PATH"] = config.bert_dir
    os.environ["NUM_LAYERS"] = str(config.num_layers)
    os.environ["NUM_ATTENTION_HEADS"] = str(config.num_attention_heads)
    os.environ["EMBEDDING_DIM"] = str(config.embedding_dim)
    # Discard any pretraining paths we don't need
    xpos = mlm = parser = False
    for k, v in config.pretrain_language_config.items():
        os.environ[k] = json.dumps(v)
        if k == "train_data_paths":
            xpos = "xpos" in v
            mlm = "mlm" in v
            parser = "parser" in v
    os.environ["XPOS"] = json.dumps(xpos)
    os.environ["MLM"] = json.dumps(mlm)
    os.environ["PARSER"] = json.dumps(parser)


@click.group()
@click.option(
    "--language",
    "-l",
    type=click.Choice(LANGUAGES),
    default="coptic",
    help="A language to train on. Must correspond to an entry in main.py's LANGUAGES"
)
@click.option("--bert-model-name", "-m", default="bert-base-multilingual-cased")
@click.option("--training-config", default="configs/bert_pretrain.jsonnet",
              help="Multitask training config. You probably want to leave this as the default.")
@click.option("--parser-eval-config", default="configs/parser_eval.jsonnet",
              help="Parser evaluation config. You probably want to leave this as the default.")
@click.option("--task", "-t", default=TASKS, multiple=True, type=click.Choice(TASKS),
              help="Specify task(s) to exclude from a run. Possible values: mlm, parser, xpos")
@click.option("--tokenization-type", "-t", type=click.Choice(TOKENIZATION_TYPES), default="wordpiece")
@click.option("--num-layers", default=3, type=int, help="Number of BERT encoder block layers")
@click.option("--num-attention-heads", default=5, type=int, help="Number of BERT attention heads")
@click.option("--embedding-dim", default=100, type=int, help="BERT hidden dimension")
@click.option("--finetune/--no-finetune", default=False, help="Whether to make BERT params trainable during evaluation")
@click.option("--debug/--no-debug", default=False, help="When set, use toy data and small iteration counts to debug")
@click.pass_context
def top(ctx, **kwargs):
    print(kwargs)
    ctx.obj = Config(**kwargs)
    logger.info(f"Parsed experimental config: {ctx.obj.__dict__}")


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Evaluation
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
def evaluate_diaparser(language, bert):
    """
    Given a language and a pretrained BERT model (or something API compatible with it),
    """
    language_config = get_eval_config(language, bert)
    logger.info("#" * 40)
    logger.info("# Training for evaluation")
    logger.info("#" * 40)
    with mkdtemp() as eval_dir:
        eval_model_path = os.path.join(eval_dir, "parser.pt")
        dp_train(
            eval_model_path,
            language_config["training"]["train_data_path"],
            language_config["training"]["validation_data_path"],
            build=True,
            feat="bert",
            bert=bert,
            epochs=5000,
            max_len=512
        )
        metrics = dp_evaluate(
            eval_model_path,
            language_config["testing"]["input_file"]
        )
    logger.info(metrics)
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


def evaluate_allennlp(config):
    """
    Given a language and a pretrained BERT model (or something API compatible with it),
    """
    with mkdtemp() as eval_dir:
        logger.info("#" * 40)
        logger.info("# Training for evaluation")
        logger.info("#" * 40)
        bert_model = BertModel.from_pretrained(config.bert_model_name)
        language_config = config.parser_eval_language_config

        os.environ["BERT_DIMS"] = str(bert_model.config.hidden_size)
        os.environ["BERT_PATH"] = config.bert_model_name
        os.environ["TRAINABLE"] = "0"
        for k, v in language_config['training'].items():
            os.environ[k] = json.dumps(v) if isinstance(v, dict) else v
        overrides = ""
        if config.finetune or config.debug:
            overrides += "{"
            if config.debug:
                overrides += '"trainer.num_epochs": 1'
                if config.finetune:
                    overrides += ","
            if config.finetune:
                overrides += '"model.text_field_embedder.token_embedders.tokens.train_parameters": true'
            overrides += "}"
        train_model_from_file(config.parser_eval_jsonnet, eval_dir, overrides=overrides)

        logger.info("#" * 40)
        logger.info("# Evaluating")
        logger.info("#" * 40)
        args = eval_args(eval_dir, language_config['testing']['input_file'])
        metrics = evaluate_from_args(args)
    logger.info(metrics)
    return None, metrics


@click.command(help="pretrained BERT (bert-base-multilingual-cased) baseline evaluate on UD parsing task")
@click.pass_obj
def pretrained_baseline_evaluate(config):
    return evaluate_allennlp(config)


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Training
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
def _train_tokenizer(language_config, bert_dir, model_type):
    # Prepare tokenizer and save to dir
    # Read data, flatten List[List[TokenList]] into List[List[str]] for tokenizer training
    logger.info("Training tokenizer...")
    documents = read_conllu_files(language_config["tokenizer_conllu_path"])
    sentences = [" ".join([t['form'] for t in sentence]) for document in documents for sentence in document]
    train_tokenizer(sentences, serialize_path=bert_dir, model_type=model_type)


@click.command(help="Run the pretraining phase of an experiment where a BERT model is trained")
@click.pass_obj
def pretrain_evaluate(config):
    # Prepare directories that will be used for the AllenNLP model and the extracted BERT model
    config.prepare_dirs(delete=True)

    # Get config and train tokenizer
    _train_tokenizer(config.pretrain_language_config, config.bert_dir, model_type=config.tokenization_type)

    # these are needed by bert_pretrain.jsonnet
    _prepare_bert_pretrain_env_vars(config)

    # Train the LM
    logger.info("Beginning pretraining...")
    logger.info("#################################")
    logger.info("Config:\n", config.pretrain_language_config)
    logger.info("#################################")
    logger.info("Env:\n", os.environ)
    logger.info("#################################")
    overrides = '{"trainer.num_epochs": 1}' if config.debug else ""
    model = train_model_from_file(config.pretrain_jsonnet, config.experiment_dir, overrides=overrides)
    bert_serialization: BertModel = model._backbone.bert
    bert_serialization.save_pretrained(config.bert_dir)
    with open(os.path.join(config.experiment_dir, "metrics.json"), 'r') as f:
        train_metrics = json.load(f)

    eval_metrics = evaluate_allennlp(config)
    return train_metrics, eval_metrics


def _locked_write(filepath, s):
    lock = FileLock(filepath + ".lock")
    with lock, open(filepath, 'a') as f:
        f.write(s)


@click.command(help="Do only the evaluation half of pretrain_evaluate. "
                    "Assumes the BERT required has already been trained.")
@click.pass_obj
def evaluate(config):
    # Get dir paths without deleting them--we're assuming they're already there
    _, eval_metrics = evaluate_allennlp(config)
    name = "mlm_only" if "mlm" in config.tasks and len(config.tasks) == 1 else "mtl"
    name += "_ft" if config.finetune else ""
    output = "\t".join([
        config.language,
        name,
        "_",
        "_",
        str(eval_metrics["LAS"]),
    ])
    _locked_write("metrics.tsv", output + "\n")
    return None, eval_metrics


@click.command(help="Run a baseline eval for a given language")
@click.pass_context
def baseline_evaluate(ctx):
    config = ctx.obj
    _, metrics = ctx.invoke(pretrained_baseline_evaluate)
    name = "pretrained_baseline"
    name += "_ft" if config.finetune else ""
    output = "\t".join([config.language, name, "", "", str(metrics["LAS"])])
    _locked_write("metrics.tsv", output + "\n")


@click.command(help="Run a full eval for a given language")
@click.pass_context
def language_trial(ctx):
    config = ctx.obj

    # MLM only
    ctx.obj.tasks = ["mlm"]
    mlm_only_metrics_train, mlm_only_metrics_eval = ctx.invoke(pretrain_evaluate)
    _, metrics = mlm_only_metrics_eval
    output = "\t".join([
        config.language,
        "mlm_only",
        "",
        "",
        str(metrics["LAS"]),
    ])
    _locked_write("metrics.tsv", output + "\n")

    # All MTL tasks
    ctx.obj.tasks = ["mlm", "xpos", "parser"]
    mtl_metrics_train, mtl_metrics_eval = ctx.invoke(pretrain_evaluate,)
    _, metrics = mtl_metrics_eval
    output = "\t".join([
        config.language,
        "mtl",
        str(mtl_metrics_train["training_parser_LAS"]),
        "_",
        str(metrics["LAS"]),
    ])
    _locked_write("metrics.tsv", output + "\n")


top.add_command(pretrain_evaluate)
top.add_command(pretrained_baseline_evaluate)
top.add_command(evaluate)
top.add_command(baseline_evaluate)
top.add_command(language_trial)


if __name__ == "__main__":
    top()
