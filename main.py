import argparse
import json
import os
import shutil

import click
from transformers import BertModel
from allennlp.commands.train import train_model_from_file
from allennlp.commands.evaluate import evaluate_from_args

import embur
from embur.dataset_reader import read_conllu_files
from embur.tokenizers import train_bert_tokenizer
from embur.language_configs import get_pretrain_config, get_eval_config


@click.group()
def top():
    pass


@click.command(help="Run the pretraining phase of an experiment where a BERT model is trained")
@click.option("--config", "-c", default="configs/bert_pretrain.jsonnet",
              help="Multitask training config. You probably want to leave this as the default.")
@click.option("--language", "-l", default="coptic",
              help="A language to train on. Must correspond to an entry in main.py's LANGUAGES")
@click.option("--exclude-tasks", "-x", default=[], multiple=True,
              help="Specify task(s) to exclude from a run. Possible values: mlm, parser, xpos")
@click.option("--serialization-dir", "-s", default="models", help="Serialization dir for pretraining")
@click.option("--bert-dir", "-o", default="bert_out", help="BERT artefacts go here")
@click.option("--num-layers", default=4, type=int, help="Number of BERT encoder block layers")
@click.option("--num-attention-heads", default=12, type=int, help="Number of BERT attention heads")
@click.option("--embedding-dim", default=60, type=int, help="BERT hidden dimension")
@click.option(
    "--tokenizer-conllu-path",
    default="data/coptic/converted/train",
    help="conllu path used to train the toknizer"
)
def pretrain(config, language, exclude_tasks, serialization_dir, bert_dir,
             num_layers, num_attention_heads, embedding_dim, tokenizer_conllu_path):
    if os.path.exists(serialization_dir):
        print(f"{serialization_dir} exists, removing...")
        shutil.rmtree(serialization_dir)
    if os.path.exists(bert_dir):
        print(f"{bert_dir} exists, removing...")
        shutil.rmtree(bert_dir)

    os.makedirs(bert_dir, exist_ok=True)

    # Prepare tokenizer and save to dir
    documents = read_conllu_files(tokenizer_conllu_path)
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
    for k, v in get_pretrain_config(language, bert_dir).items():
        if isinstance(v, dict):
            v = {k2: v2 for k2, v2 in v.items() if k2 not in exclude_tasks}
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
    args.batch_size = 64
    args.archive_file = f"{serialization_dir}/model.tar.gz"
    args.input_file = input_file
    args.output_file = serialization_dir + "/metrics"
    args.predictions_output_file = serialization_dir + "/predictions"
    args.cuda_device = 0
    return args


@click.command(help="Evaluate on UD parsing task")
@click.option("--config", "-c", default="configs/bert_eval.jsonnet", help="Entity eval training config")
@click.option("--serialization-dir", "-s", default="models_eval", help="Serialization dir")
@click.option("--bert-path", "-b", default="bert_out", help="Path to pretrained bert")
@click.option("--trainable/--no-trainable", default=True, help="If true, embedding layers are adjusted")
def evaluate(config, serialization_dir, bert_path, trainable):
    if os.path.exists(serialization_dir):
        print(f"{serialization_dir} exists, removing...")
        shutil.rmtree(serialization_dir)

    language_config = get_eval_config("coptic", bert_path)

    print("#" * 40)
    print("# Training")
    print("#" * 40)
    os.environ["BERT_DIMS"] = str(BertModel.from_pretrained(bert_path).config.hidden_size)
    os.environ["BERT_PATH"] = bert_path
    os.environ["TRAINABLE"] = str(int(trainable))
    for k, v in language_config['training'].items():
        os.environ[k] = json.dumps(v)
    train_model_from_file(config, serialization_dir, include_package="allennlp_models")

    print("#" * 40)
    print("# Evaluating")
    print("#" * 40)
    args = eval_args(serialization_dir, language_config['testing']['input_file'])
    evaluate_from_args(args)


top.add_command(pretrain)
top.add_command(evaluate)


if __name__ == "__main__":
    top()
