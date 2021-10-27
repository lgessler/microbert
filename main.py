import argparse
import json
import os
import shutil

import click
from allennlp.common.util import import_module_and_submodules
from transformers import BertModel
from allennlp.commands.train import train_model_from_file
from allennlp.commands.evaluate import evaluate_from_args

import embur
from embur.dataset_reader import read_conllu_files
from embur.tokenizers import train_bert_tokenizer
from embur.language_configs import get_pretrain_config, get_eval_config, LANGUAGES


import_module_and_submodules("allennlp_models")
TASKS = ['mlm', 'xpos', 'parser']


def _bert_dir(language, tasks, num_layers, num_heads, embedding_dim):
    return f"berts/{language}/" \
           f"{'-'.join(tasks)}" \
           f"_layers-{num_layers}" \
           f"_heads-{num_heads}" \
           f"_hidden-{embedding_dim}"


def _model_dir(step, language, tasks, num_layers, num_heads, embedding_dim):
    return f"models/{language}/" \
           f"{'-'.join(tasks)}" \
           f"_layers-{num_layers}" \
           f"_heads-{num_heads}" \
           f"_hidden-{embedding_dim}" \
           f"_{step}"


@click.group()
def top():
    pass


@click.command(help="Run the pretraining phase of an experiment where a BERT model is trained")
@click.option("--config", "-c", default="configs/bert_pretrain.jsonnet",
              help="Multitask training config. You probably want to leave this as the default.")
@click.option("--language", "-l", type=click.Choice(LANGUAGES), default="coptic",
              help="A language to train on. Must correspond to an entry in main.py's LANGUAGES")
@click.option("--exclude-task", "-x", default=[], multiple=True,
              help="Specify task(s) to exclude from a run. Possible values: mlm, parser, xpos")
@click.option("--num-layers", default=2, type=int, help="Number of BERT encoder block layers")
@click.option("--num-attention-heads", default=10, type=int, help="Number of BERT attention heads")
@click.option("--embedding-dim", default=50, type=int, help="BERT hidden dimension")
def pretrain(config, language, exclude_task, num_layers, num_attention_heads, embedding_dim):
    tasks = [x for x in TASKS if x not in exclude_task]
    bert_dir = _bert_dir(
        language,
        tasks,
        num_layers,
        num_attention_heads,
        embedding_dim
    )
    serialization_dir = _model_dir("pretrain", language, tasks, num_layers, num_attention_heads, embedding_dim)

    if os.path.exists(bert_dir):
        print(f"{bert_dir} exists, removing...")
        shutil.rmtree(bert_dir)
    if os.path.exists(serialization_dir):
        print(f"{serialization_dir} exists, removing...")
        shutil.rmtree(serialization_dir)

    os.makedirs(bert_dir, exist_ok=True)

    # Prepare tokenizer and save to dir
    language_config = get_pretrain_config(language, bert_dir, exclude_task)
    documents = read_conllu_files(language_config["tokenizer_conllu_path"])
    sentences = []
    for document in documents:
        for sentence in document:
            sentences.append(" ".join([t['form'] for t in sentence]))
    print("Training tokenizer...")
    os.environ["TOKENIZER_PATH"] = bert_dir
    os.environ["NUM_LAYERS"] = str(num_layers)
    os.environ["NUM_ATTENTION_HEADS"] = str(num_attention_heads)
    os.environ["EMBEDDING_DIM"] = str(embedding_dim)
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
    # TODO: check pretrained tokenizer for behavior
    train_bert_tokenizer(sentences, serialize_path=bert_dir, vocab_size=6000)

    # Train the LM
    print("Beginning pretraining...")
    model = train_model_from_file(config, serialization_dir)

    # Write out
    bert_serialization: BertModel = model._backbone.bert
    bert_serialization.save_pretrained(bert_dir)


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
    return args


@click.command(help="Evaluate on UD parsing task")
@click.option("--config", "-c", default="configs/bert_eval.jsonnet", help="Entity eval training config")
@click.option("--language", "-l", type=click.Choice(LANGUAGES), default="coptic",
              help="A language to train on. Must correspond to an entry in main.py's LANGUAGES")
@click.option("--exclude-task", "-x", default=[], multiple=True,
              help="Specify task(s) to exclude from a run. Possible values: mlm, parser, xpos")
@click.option("--num-layers", default=2, type=int, help="Number of BERT encoder block layers")
@click.option("--num-attention-heads", default=10, type=int, help="Number of BERT attention heads")
@click.option("--embedding-dim", default=50, type=int, help="BERT hidden dimension")
@click.option("--trainable/--no-trainable", default=True, help="If true, embedding layers are adjusted")
def evaluate(config, language, exclude_task, num_layers, num_attention_heads, embedding_dim, trainable):
    tasks = [x for x in TASKS if x not in exclude_task]
    bert_dir = _bert_dir(
        language,
        tasks,
        num_layers,
        num_attention_heads,
        embedding_dim
    )
    serialization_dir = _model_dir("pretrain", language, tasks, num_layers, num_attention_heads, embedding_dim)

    if os.path.exists(serialization_dir):
        print(f"{serialization_dir} exists, removing...")
        shutil.rmtree(serialization_dir)

    language_config = get_eval_config(language, bert_dir)

    print("#" * 40)
    print("# Training")
    print("#" * 40)
    bert_model = BertModel.from_pretrained(bert_dir)

    os.environ["BERT_DIMS"] = str(bert_model.config.hidden_size)
    os.environ["BERT_PATH"] = bert_dir
    os.environ["TRAINABLE"] = str(int(trainable))
    for k, v in language_config['training'].items():
        os.environ[k] = json.dumps(v) if isinstance(v, dict) else v
    train_model_from_file(config, serialization_dir)

    print("#" * 40)
    print("# Evaluating")
    print("#" * 40)
    args = eval_args(serialization_dir, {"parser": language_config['testing']['input_file']})
    evaluate_from_args(args)


@click.command(help="Baseline evaluate on UD parsing task")
@click.option("--config", "-c", default="configs/bert_baseline_eval.jsonnet")
@click.option("--language", "-l", type=click.Choice(LANGUAGES), default="coptic",
              help="A language to train on. Must correspond to an entry in main.py's LANGUAGES")
@click.option("--num-layers", default=2, type=int, help="Number of BERT encoder block layers")
@click.option("--num-attention-heads", default=10, type=int, help="Number of BERT attention heads")
@click.option("--embedding-dim", default=50, type=int, help="BERT hidden dimension")
def baseline_evaluate(config, language, num_layers, num_attention_heads, embedding_dim):
    tasks = ["baseline"]
    bert_dir = _bert_dir(
        language,
        tasks,
        num_layers,
        num_attention_heads,
        embedding_dim
    )
    serialization_dir = _model_dir("pretrain", language, tasks, num_layers, num_attention_heads, embedding_dim)
    if os.path.exists(bert_dir):
        print(f"{bert_dir} exists, removing...")
        shutil.rmtree(bert_dir)
    if os.path.exists(serialization_dir):
        print(f"{serialization_dir} exists, removing...")
        shutil.rmtree(serialization_dir)

    os.makedirs(bert_dir, exist_ok=True)

    training_config = get_pretrain_config(language, bert_dir, [])
    documents = read_conllu_files(training_config["tokenizer_conllu_path"])
    sentences = []
    for document in documents:
        for sentence in document:
            sentences.append(" ".join([t['form'] for t in sentence]))
    print("Training tokenizer...")
    os.environ["TOKENIZER_PATH"] = bert_dir
    train_bert_tokenizer(sentences, serialize_path=bert_dir, vocab_size=6000)

    print("#" * 40)
    print("# Baseline training")
    print("#" * 40)
    os.environ["BERT_DIMS"] = str(embedding_dim)
    os.environ["BERT_LAYERS"] = str(num_layers)
    os.environ["BERT_HEADS"] = str(num_attention_heads)
    os.environ["BERT_PATH"] = bert_dir
    eval_config = get_eval_config(language, bert_dir)
    for k, v in eval_config['training'].items():
        os.environ[k] = json.dumps(v) if isinstance(v, dict) else v
    model = train_model_from_file(config, serialization_dir)
    bert_serialization: BertModel = model._backbone.bert
    bert_serialization.save_pretrained(bert_dir)

    print("#" * 40)
    print("# Baseline evaluating")
    print("#" * 40)
    args = eval_args(serialization_dir, {"parser": eval_config['testing']['input_file']})
    evaluate_from_args(args)


@click.command(help="Run a full eval for a given language")
@click.argument('language', type=click.Choice(LANGUAGES))
@click.pass_context
def language_trial(ctx, language):
    ctx.forward(baseline_evaluate)
    ctx.invoke(baseline_evaluate, language=language)

    ctx.forward(pretrain)
    ctx.invoke(pretrain, exclude_task=['parser', 'xpos'], language=language)

    ctx.forward(pretrain)
    ctx.invoke(pretrain, language=language)


top.add_command(pretrain)
top.add_command(evaluate)
top.add_command(baseline_evaluate)
top.add_command(language_trial)


if __name__ == "__main__":
    top()
