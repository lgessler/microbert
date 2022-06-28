import csv
import json
import os

from rich import print
import click
from allennlp.commands.train import train_model_from_file
from allennlp.common.util import import_module_and_submodules
from filelock import FileLock
from transformers import BertModel
from gensim.models import Word2Vec

from embur.config import Config, TASKS, TOKENIZATION_TYPES
from embur.dataset_reader import read_conllu_files
from embur.eval.allennlp import evaluate_allennlp, evaluate_allennlp_static
from embur.language_configs import LANGUAGES
from embur.tokenizers import train_tokenizer

import_module_and_submodules("allennlp_models")


@click.group()
@click.option(
    "--language",
    "-l",
    type=click.Choice(LANGUAGES),
    default="coptic",
    help="A language to train on. Must correspond to an entry in main.py's LANGUAGES",
)
@click.option("--bert-model-name", "-m", default="bert-base-multilingual-cased")
@click.option(
    "--training-config",
    default="configs/bert_pretrain.jsonnet",
    help="Multitask training config. You probably want to leave this as the default.",
)
@click.option(
    "--parser-eval-config",
    default="configs/parser_eval.jsonnet",
    help="Parser evaluation config. You probably want to leave this as the default.",
)
@click.option(
    "--task",
    "-t",
    default=TASKS,
    multiple=True,
    type=click.Choice(TASKS),
    help="Specify task(s) to exclude from a run. Possible values: mlm, parser, xpos",
)
@click.option(
    "--tokenization-type", type=click.Choice(TOKENIZATION_TYPES), default="wordpiece"
)
@click.option(
    "--num-layers", default=3, type=int, help="Number of BERT encoder block layers"
)
@click.option(
    "--num-attention-heads", default=5, type=int, help="Number of BERT attention heads"
)
@click.option("--embedding-dim", default=100, type=int, help="BERT hidden dimension")
@click.option(
    "--finetune/--no-finetune",
    default=False,
    help="Whether to make BERT params trainable during evaluation",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="When set, use toy data and small iteration counts to debug",
)
@click.pass_context
def top(ctx, **kwargs):
    ctx.obj = Config(**kwargs)
    print(f"Parsed experimental config: {ctx.obj.__dict__}")


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Training
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
@click.command(help="Train word2vec embeddings and use to evaluate.")
@click.pass_obj
def word2vec_evaluate(config):
    config.finetune = True  # always finetune
    config.parser_eval_jsonnet = "configs/static_parser_eval.jsonnet"

    documents = read_conllu_files(
        config.pretrain_language_config["tokenizer_conllu_path"]
    )
    sentences = [
        [t["form"] for t in sentence] for document in documents for sentence in document
    ]
    model = Word2Vec(
        sentences=sentences,
        sg=1,
        negative=5,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
    )

    os.makedirs("word2vec", exist_ok=True)
    with open(config.word2vec_file, "w") as f:
        f.write(f"{len(model.wv.key_to_index)} {config.embedding_dim}\n")
        for word in model.wv.key_to_index:
            vector = model.wv.get_vector(word).tolist()
            row = [word] + [str(x) for x in vector]
            f.write(" ".join(row) + "\n")

    train_metrics, eval_metrics = evaluate_allennlp_static(config)
    _write_to_tsv(config, "word2vec" + ("_ft" if config.finetune else ""), eval_metrics)


@click.command(
    help="Run the pretraining phase of an experiment where a BERT model is trained"
)
@click.pass_context
def pretrain_evaluate(ctx):
    # Prepare directories that will be used for the AllenNLP model and the extracted BERT model
    config = ctx.obj
    config.prepare_dirs(delete=True)

    # Get config and train tokenizer
    print("Training tokenizer...")
    documents = read_conllu_files(
        config.pretrain_language_config["tokenizer_conllu_path"]
    )
    sentences = [
        " ".join([t["form"] for t in sentence])
        for document in documents
        for sentence in document
    ]
    train_tokenizer(
        sentences, serialize_path=config.bert_dir, model_type=config.tokenization_type
    )

    # these are needed by bert_pretrain.jsonnet
    config.prepare_bert_pretrain_env_vars()

    # Train the LM
    print("Beginning pretraining...")
    print("Config:\n", config.pretrain_language_config)
    print("Env:\n", os.environ)
    overrides = (
        '{"trainer.num_epochs": 1, "data_loader.instances_per_epoch": 256}'
        if config.debug
        else ""
    )
    model = train_model_from_file(
        config.pretrain_jsonnet, config.experiment_dir, overrides=overrides
    )
    bert_serialization: BertModel = model._backbone.bert
    bert_serialization.save_pretrained(config.bert_dir)
    with open(os.path.join(config.experiment_dir, "metrics.json"), "r") as f:
        train_metrics = json.load(f)

    # Evaluation
    ctx.invoke(evaluate)


def _write_to_tsv(config, name, metrics):
    output = "\t".join(
        [
            config.language,
            name,
            "_",
            "_",
            str(metrics["LAS"]),
        ]
    )
    _locked_write("metrics.tsv", output + "\n")


def _locked_write(filepath, s):
    lock = FileLock(filepath + ".lock")
    with lock, open(filepath, "a") as f:
        f.write(s)


@click.command(
    help="Do only the evaluation half of pretrain_evaluate. Assumes the BERT required has already been trained."
)
@click.pass_obj
def evaluate(config):
    # Get dir paths without deleting them--we're assuming they're already there
    config.bert_model_name = None
    _, eval_metrics = evaluate_allennlp(config)
    _write_to_tsv(config, "-".join(config.tasks) + ("_ft" if config.finetune else ""), eval_metrics)
    return None, eval_metrics


@click.command(help="Run a baseline eval for a given language")
@click.pass_context
def baseline_evaluate(ctx):
    config = ctx.obj
    config.bert_model_name = "bert-base-multilingual-cased"
    _, metrics = evaluate_allennlp(config)
    name = "pretrained_baseline"
    name += "_ft" if config.finetune else ""
    output = "\t".join([config.language, name, "", "", str(metrics["LAS"])])
    _locked_write("metrics.tsv", output + "\n")


@click.command(help="Run a full eval for a given language")
@click.pass_context
def language_trial(ctx):
    config = ctx.obj

    # MLM only
    ctx.obj.set_tasks(["mlm"])
    ctx.invoke(pretrain_evaluate)

    # All MTL tasks
    ctx.obj.set_tasks(["mlm", "xpos", "parser"])
    ctx.invoke(pretrain_evaluate)


top.add_command(pretrain_evaluate)
top.add_command(word2vec_evaluate)
top.add_command(baseline_evaluate)
top.add_command(evaluate)
top.add_command(language_trial)


if __name__ == "__main__":
    top()
