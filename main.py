import json
import os
import shutil

import click
from allennlp.commands.train import train_model_from_file
from allennlp.common.util import import_module_and_submodules
from filelock import FileLock
from rich import print
from torch.serialization import mkdtemp
from transformers import BertModel

import embur
from embur.dataset_reader import read_conllu_files
from embur.diaparser import evaluate as dp_evaluate
from embur.diaparser import train as dp_train
from embur.language_configs import LANGUAGES, get_eval_config, get_pretrain_config
from embur.tokenizers import train_tokenizer

import_module_and_submodules("allennlp_models")
TASKS = ['mlm', 'xpos', 'parser']


def _bert_dir(language, tasks, num_layers=None, num_heads=None, embedding_dim=None):
    return (
        f"berts/{language}/"
        + f"{'-'.join(tasks)}"
        + (f"_layers-{num_layers}" if num_layers is not None else "")
        + (f"_heads-{num_heads}" if num_heads is not None else "")
        + (f"_hidden-{embedding_dim}" if embedding_dim is not None else "")
    )


def _model_dir(step, language, tasks, num_layers=None, num_heads=None, embedding_dim=None, model_name=None):
    return (
        f"models/{language}/"
        + f"{'-'.join(tasks)}"
        + (f"_layers-{num_layers}" if num_layers is not None else "")
        + (f"_heads-{num_heads}" if num_heads is not None else "")
        + (f"_hidden-{embedding_dim}" if embedding_dim is not None else "")
        + (f"_{model_name}" if model_name is not None else "")
        + f"_{step}"
    )


@click.group()
def top():
    pass


def evaluate(language, bert):
    language_config = get_eval_config(language, bert)
    print("#" * 40)
    print("# Training for evaluation")
    print("#" * 40)
    with mkdtemp() as eval_dir:
        eval_model_path = os.path.join(eval_dir, "parser.pt")
        dp_train(
            eval_model_path,
            language_config["training"]["train_data_path"],
            language_config["training"]["validation_data_path"],
            build=True,
            feat="bert",
            bert=bert,
            epochs=5000
        )
        metrics = dp_evaluate(
            eval_model_path,
            language_config["testing"]["input_file"]
        )
    print(metrics)
    return metrics


@click.command(help="pretrained BERT (bert-base-multilingual-cased) baseline evaluate on UD parsing task")
@click.option("--language", "-l", type=click.Choice(LANGUAGES), default="coptic",
              help="A language to train on. Must correspond to an entry in main.py's LANGUAGES")
@click.option("--model-name", "-m", default="bert-base-multilingual-cased")
def pretrained_baseline_evaluate(language, model_name):
    return evaluate(language, model_name)


def _prepare_dirs(language, tasks, num_layers, num_attention_heads, embedding_dim):
    bert_dir = _bert_dir(
        language,
        tasks,
        num_layers,
        num_attention_heads,
        embedding_dim
    )
    serialization_dir = _model_dir("random_baseline", language, tasks, num_layers, num_attention_heads, embedding_dim)
    if os.path.exists(bert_dir):
        print(f"{bert_dir} exists, removing...")
        shutil.rmtree(bert_dir)
    if os.path.exists(serialization_dir):
        print(f"{serialization_dir} exists, removing...")
        shutil.rmtree(serialization_dir)
    os.makedirs(bert_dir, exist_ok=True)

    return bert_dir, serialization_dir


def _prepare_bert_pretrain_env_vars(bert_dir, language_config, num_layers, num_attention_heads, embedding_dim):
    os.environ["TOKENIZER_PATH"] = bert_dir
    os.environ["NUM_LAYERS"] = str(num_layers)
    os.environ["NUM_ATTENTION_HEADS"] = str(num_attention_heads)
    os.environ["EMBEDDING_DIM"] = str(embedding_dim)
    # Discard any pretraining paths we don't need
    xpos = mlm = parser = False
    for k, v in language_config.items():
        os.environ[k] = json.dumps(v)
        if k == "train_data_paths":
            xpos = "xpos" in v
            mlm = "mlm" in v
            parser = "parser" in v
    os.environ["XPOS"] = json.dumps(xpos)
    os.environ["MLM"] = json.dumps(mlm)
    os.environ["PARSER"] = json.dumps(parser)


def _train_tokenizer(language_config, bert_dir, model_type):
    # Prepare tokenizer and save to dir
    # Read data, flatten List[List[TokenList]] into List[List[str]] for tokenizer training
    print("Training tokenizer...")
    documents = read_conllu_files(language_config["tokenizer_conllu_path"])
    sentences = [" ".join([t['form'] for t in sentence]) for document in documents for sentence in document]
    train_tokenizer(sentences, serialize_path=bert_dir, model_type=model_type)


@click.command(help="Run the pretraining phase of an experiment where a BERT model is trained")
@click.option("--config", "-c", default="configs/bert_pretrain.jsonnet",
              help="Multitask training config. You probably want to leave this as the default.")
@click.option("--language", "-l", type=click.Choice(LANGUAGES), default="coptic",
              help="A language to train on. Must correspond to an entry in main.py's LANGUAGES")
@click.option("--exclude-task", "-x", default=[], multiple=True, type=click.Choice(["mlm", "parser", "xpos"]),
              help="Specify task(s) to exclude from a run. Possible values: mlm, parser, xpos")
@click.option("--tokenization-type", "-t", type=click.Choice(["wordpiece", "bpe"]), default="wordpiece")
@click.option("--num-layers", default=2, type=int, help="Number of BERT encoder block layers")
@click.option("--num-attention-heads", default=10, type=int, help="Number of BERT attention heads")
@click.option("--embedding-dim", default=50, type=int, help="BERT hidden dimension")
def pretrain_evaluate(config, language, exclude_task, tokenization_type, num_layers, num_attention_heads, embedding_dim):
    # Prepare directories that will be used for the AllenNLP model and the extracted BERT model
    tasks = [x for x in TASKS if x not in exclude_task]
    bert_dir, serialization_dir = _prepare_dirs(language, tasks, num_layers, num_attention_heads, embedding_dim)

    # Get config and train tokenizer
    language_config = get_pretrain_config(language, bert_dir, exclude_task)
    _train_tokenizer(language_config, bert_dir, model_type=tokenization_type)

    # these are needed by bert_pretrain.jsonnet
    _prepare_bert_pretrain_env_vars(bert_dir, language_config, num_layers, num_attention_heads, embedding_dim)

    # Train the LM
    print("Beginning pretraining...")
    print("#################################")
    print("Config:\n", config)
    print("#################################")
    print("Env:\n", os.environ)
    print("#################################")
    model = train_model_from_file(config, serialization_dir)
    bert_serialization: BertModel = model._backbone.bert
    bert_serialization.save_pretrained(bert_dir)
    with open(os.path.join(serialization_dir, "metrics.json"), 'r') as f:
        train_metrics = json.load(f)

    eval_metrics = evaluate(language, bert_dir)
    return train_metrics, eval_metrics


def _locked_write(filepath, s):
    lock = FileLock(filepath + ".lock")
    with lock, open(filepath, 'a') as f:
        f.write(s)


@click.command(help="Run a full eval for a given language")
@click.argument('language', type=click.Choice(LANGUAGES))
@click.option("--tokenization-type", "-t", type=click.Choice(["wordpiece", "bpe"]), default="wordpiece")
@click.pass_context
def language_trial(ctx, language, tokenization_type):
    # metrics is a `diaparser.utils.metric.AttachmentMetric` object
    #_, metrics = ctx.invoke(pretrained_baseline_evaluate, language=language)
    #output = "\t".join([language, "pretrained_baseline", "", "", str(metrics.las)])
    #_locked_write("metrics.tsv", output + "\n")

    # MLM only
    mlm_only_metrics_train, mlm_only_metrics_eval = ctx.invoke(
        pretrain_evaluate,
        language=language,
        exclude_task=['parser', 'xpos'],
        tokenization_type=tokenization_type
    )
    _, metrics = mlm_only_metrics_eval
    output = "\t".join([
        language,
        "mlm_only",
        "",
        "",
        str(metrics.las),
    ])
    _locked_write("metrics.tsv", output + "\n")

    # All MTL tasks
    mtl_metrics_train, mtl_metrics_eval = ctx.invoke(
        pretrain_evaluate,
        language=language,
        tokenization_type=tokenization_type
    )
    _, metrics = mtl_metrics_eval
    output = "\t".join([
        language,
        "mtl",
        str(mtl_metrics_train["training_parser_LAS"]),
        str(mtl_metrics_train["validation_parser_LAS"]),
        str(metrics.las),
    ])
    _locked_write("metrics.tsv", output + "\n")


top.add_command(pretrain_evaluate)
top.add_command(pretrained_baseline_evaluate)
top.add_command(language_trial)


if __name__ == "__main__":
    top()
