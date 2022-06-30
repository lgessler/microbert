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
@click.option("--tokenization-type", type=click.Choice(TOKENIZATION_TYPES), default="wordpiece")
@click.option("--num-layers", default=3, type=int, help="Number of BERT encoder block layers")
@click.option("--num-attention-heads", default=5, type=int, help="Number of BERT attention heads")
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


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Training
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
@click.command(help="Train word2vec embeddings")
@click.pass_obj
def word2vec_train(config):
    documents = read_conllu_files(config.pretrain_language_config["tokenizer_conllu_path"])
    sentences = [[t["form"] for t in sentence] for document in documents for sentence in document]
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
    print(f"Wrote to {config.word2vec_file}")


@click.command(help="Pretrain a monolingual BERT")
@click.pass_context
def pretrain(ctx):
    # Prepare directories that will be used for the AllenNLP model and the extracted BERT model
    config = ctx.obj
    config.prepare_dirs(delete=True)

    # Get config and train tokenizer
    print("Training tokenizer...")
    documents = read_conllu_files(config.pretrain_language_config["tokenizer_conllu_path"])
    sentences = [" ".join([t["form"] for t in sentence]) for document in documents for sentence in document]
    train_tokenizer(sentences, serialize_path=config.bert_dir, model_type=config.tokenization_type)

    # these are needed by bert_pretrain.jsonnet
    config.prepare_bert_pretrain_env_vars()

    # Train the LM
    print("Beginning pretraining...")
    print("Config:\n", config.pretrain_language_config)
    print("Env:\n", os.environ)
    overrides = '{"trainer.num_epochs": 1, "data_loader.instances_per_epoch": 256}' if config.debug else ""
    model = train_model_from_file(config.pretrain_jsonnet, config.experiment_dir, overrides=overrides)
    bert_serialization: BertModel = model._backbone.bert
    bert_serialization.save_pretrained(config.bert_dir)
    with open(os.path.join(config.experiment_dir, "metrics.json"), "r") as f:
        train_metrics = json.load(f)


################################################################################
# Evaluation
################################################################################
@click.command(help="Evaluate a monolingual BERT")
@click.pass_obj
def evaluate(config):
    tmp = config.bert_model_name

    config.bert_model_name = None
    _, eval_metrics = evaluate_allennlp(config)
    _write_to_tsv(
        config,
        "-".join(config.tasks) + ("_ft" if config.finetune else ""),
        eval_metrics,
    )

    config.bert_model_name = tmp


@click.command(help="Run a baseline eval for a given language")
@click.pass_context
def baseline_evaluate(ctx):
    config = ctx.obj
    tmp = config.bert_model_name

    config.bert_model_name = "bert-base-multilingual-cased"
    _, metrics = evaluate_allennlp(config)
    name = "pretrained_baseline"
    name += "_ft" if config.finetune else ""
    _write_to_tsv(config, name, metrics)

    config.bert_model_name = tmp


@click.command(help="Evaluate word2vec embeddings")
@click.pass_obj
def word2vec_evaluate(config):
    tmp = config.parser_eval_jsonnet

    config.parser_eval_jsonnet = "configs/static_parser_eval.jsonnet"
    train_metrics, eval_metrics = evaluate_allennlp_static(config)
    _write_to_tsv(config, "word2vec" + ("_ft" if config.finetune else ""), eval_metrics)

    config.parser_eval_jsonnet = tmp


@click.command(help="Prepare data for a given language")
@click.pass_obj
def prepare_data(config):
    lang = config.language
    if lang == "coptic":
        from embur.scripts.coptic_data_prep import main

        main()
    elif lang == "greek":
        from embur.scripts.greek_data_prep import main

        main()
    elif lang in ["wolof", "uyghur", "maltese"]:
        from embur.scripts.wiki_prep import punct_inner

        punct_inner(f"data/{lang}/corpora", f"data/{lang}/converted_punct")
    else:
        raise Exception(f"Unknown language: {lang}")


@click.command(help="Run all evals for a given language")
@click.pass_context
def evaluate_all(ctx):
    config = ctx.obj

    # word2vec baseline
    config.finetune = False
    ctx.invoke(word2vec_evaluate)
    config.finetune = True
    ctx.invoke(word2vec_evaluate)

    # mBERT baseline
    config.finetune = False
    ctx.invoke(baseline_evaluate)
    config.finetune = True
    ctx.invoke(baseline_evaluate)

    # MLM only
    config.set_tasks(["mlm"])
    config.finetune = False
    ctx.invoke(evaluate)
    config.finetune = True
    ctx.invoke(evaluate)

    # MLM, xpos
    config.set_tasks(["mlm", "xpos"])
    config.finetune = False
    ctx.invoke(evaluate)
    config.finetune = True
    ctx.invoke(evaluate)

    # MLM, xpos, parser
    config.set_tasks(["mlm", "xpos", "parser"])
    config.finetune = False
    ctx.invoke(evaluate)
    config.finetune = True
    ctx.invoke(evaluate)


top.add_command(prepare_data)

top.add_command(word2vec_train)
top.add_command(pretrain)
top.add_command(word2vec_evaluate)
top.add_command(baseline_evaluate)
top.add_command(evaluate)

top.add_command(evaluate_all)


if __name__ == "__main__":
    top()
