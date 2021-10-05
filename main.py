import argparse
import os
import shutil

import click
from transformers import BertModel
from allennlp.commands.train import train_model_from_file
from allennlp.commands.evaluate import evaluate_from_args

import embur
from embur.dataset_reader import read_conllu_files
from embur.tokenizers import train_bert_tokenizer


@click.group()
def top():
    pass


@click.command(help="Run an experiment from end to end")
@click.option("--config", "-c", default="configs/bert_pretrain.jsonnet", help="Multitask training config")
@click.option("--serialization-dir", "-s", default="models", help="Serialization dir for pretraining")
@click.option("--output-dir", "-o", default="bert_out", help="BERT artefacts go here")
@click.option("--num-layers", default=4, type=int, help="Number of BERT encoder block layers")
@click.option("--num-attention-heads", default=12, type=int, help="Number of BERT attention heads")
@click.option("--embedding-dim", default=60, type=int, help="BERT hidden dimension")
@click.option(
    "--tokenizer-conllu-path",
    default="data/coptic/converted/train",
    help="conllu path used to train the toknizer"
)
def pretrain(config, serialization_dir, output_dir, num_layers, num_attention_heads, embedding_dim, tokenizer_conllu_path):
    if os.path.exists(serialization_dir):
        print(f"{serialization_dir} exists, removing...")
        shutil.rmtree(serialization_dir)
    if os.path.exists(output_dir):
        print(f"{output_dir} exists, removing...")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Prepare tokenizer and save to dir
    documents = read_conllu_files(tokenizer_conllu_path)
    sentences = []
    for document in documents:
        for sentence in document:
            sentences.append(" ".join([t['form'] for t in sentence]))
    print("Training tokenizer...")
    os.environ["TOKENIZER_PATH"] = output_dir
    os.environ["NUM_LAYERS"] = str(num_layers)
    os.environ["NUM_ATTENTION_HEADS"] = str(num_attention_heads)
    os.environ["EMBEDDING_DIM"] = str(embedding_dim)
    # TODO: check pretrained tokenizer for behavior
    train_bert_tokenizer(sentences, serialize_path=output_dir, vocab_size=6000)

    # Train the LM
    print("Beginning pretraining...")
    model = train_model_from_file(config, serialization_dir)

    # Write out
    bert_serialization: BertModel = model._backbone.bert
    bert_serialization.save_pretrained(output_dir)


def eval_args(serialization_dir):
    args = argparse.Namespace()
    args.file_friendly_logging = False
    args.weights_file = None
    args.overrides = None
    args.embedding_sources_mapping = None
    args.extend_vocab = False
    args.batch_weight_key = None
    args.batch_size = 64
    args.archive_file = f"{serialization_dir}/model.tar.gz"
    args.input_file = "data/coptic/converted/dev"
    args.output_file = serialization_dir + "/metrics"
    args.predictions_output_file = serialization_dir + "/predictions"
    args.cuda_device = 0
    return args


@click.command(help="Evaluate on the double-nested entity recognition task")
@click.option("--config", "-c", default="configs/bert_eval.jsonnet", help="Entity eval training config")
@click.option("--serialization-dir", "-s", default="models_eval", help="Serialization dir")
@click.option("--bert-path", "-b", default="bert_out", help="Path to pretrained bert")
@click.option("--trainable/--no-trainable", default=True, help="If true, embedding layers are adjusted")
def evaluate(config, serialization_dir, bert_path, trainable):
    if os.path.exists(serialization_dir):
        print(f"{serialization_dir} exists, removing...")
        shutil.rmtree(serialization_dir)

    print("#" * 40)
    print("# Training")
    print("#" * 40)
    os.environ["BERT_DIMS"] = str(BertModel.from_pretrained(bert_path).config.hidden_size)
    os.environ["BERT_PATH"] = bert_path
    os.environ["TRAINABLE"] = str(int(trainable))
    train_model_from_file(config, serialization_dir)

    print("#" * 40)
    print("# Evaluating")
    print("#" * 40)
    args = eval_args(serialization_dir)
    evaluate_from_args(args)


@click.command(help="Evaluate on the double-nested entity recognition task using pretrained word2vec")
@click.option("--config", "-c", default="configs/w2v_eval.jsonnet", help="Entity eval training config")
@click.option("--serialization-dir", "-s", default="models_w2v_eval", help="Serialization dir")
@click.option("--trainable/--no-trainable", default=True, help="If true, embedding layers are adjusted")
def evaluate_w2v(config, serialization_dir, trainable):
    if os.path.exists(serialization_dir):
        print(f"{serialization_dir} exists, removing...")
        shutil.rmtree(serialization_dir)

    print("#" * 40)
    print("# Training")
    print("#" * 40)
    os.environ["TRAINABLE"] = str(int(trainable))
    train_model_from_file(config, serialization_dir)

    print("#" * 40)
    print("# Evaluating")
    print("#" * 40)
    args = eval_args(serialization_dir)
    evaluate_from_args(args)


top.add_command(pretrain)
top.add_command(evaluate)
top.add_command(evaluate_w2v)


if __name__ == "__main__":
    top()
