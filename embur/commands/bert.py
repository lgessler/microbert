import json
import logging
import os
import shutil

import click
from allennlp.commands.train import train_model_from_file
from transformers import BertModel

from embur.commands.common import write_to_tsv, default_options
from embur.dataset_reader import read_conllu_files
from embur.eval.allennlp import evaluate_allennlp
from embur.language_configs import get_pretrain_config, get_eval_config
from embur.tokenizers import train_tokenizer


TASKS = ("mlm", "xpos", "parser")
TOKENIZATION_TYPES = ("bpe", "wordpiece")

logger = logging.getLogger(__name__)


@click.group(help="Use a BERT we trained ourselves")
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
@click.pass_context
def bert(ctx, **kwargs):
    ctx.obj.experiment_config = BertExperimentConfig(language=ctx.obj.language, **kwargs)


class BertExperimentConfig:
    def __init__(self, language, **kwargs):
        self.language = language
        combined_kwargs = default_options(bert)
        combined_kwargs.update(kwargs)

        self.tasks = combined_kwargs.pop("task")
        self.tokenization_type = combined_kwargs.pop("tokenization_type")
        self.num_layers = combined_kwargs.pop("num_layers")
        self.num_attention_heads = combined_kwargs.pop("num_attention_heads")
        self.embedding_dim = combined_kwargs.pop("embedding_dim")

        self.pretrain_language_config = get_pretrain_config(self.language, self.bert_dir, self.tasks)
        self.pretrain_jsonnet = combined_kwargs.pop("training_config")
        self.parser_eval_language_config = get_eval_config(self.language, self.bert_dir)
        self.parser_eval_jsonnet = combined_kwargs.pop("parser_eval_config")

    def set_tasks(self, tasks):
        self.tasks = tasks
        self.pretrain_language_config = get_pretrain_config(self.language, self.bert_dir, self.tasks)

    @property
    def bert_dir(self):
        return (
            f"berts/{self.language}/"
            + f"{'-'.join(self.tasks)}"
            + (f"_layers-{self.num_layers}" if self.num_layers is not None else "")
            + (f"_heads-{self.num_attention_heads}" if self.num_attention_heads is not None else "")
            + (f"_hidden-{self.embedding_dim}" if self.embedding_dim is not None else "")
        )

    @property
    def experiment_dir(self):
        return (
            f"models/{self.language}/"
            + f"{'-'.join(self.tasks)}"
            + (f"_layers-{self.num_layers}" if self.num_layers is not None else "")
            + (f"_heads-{self.num_attention_heads}" if self.num_attention_heads is not None else "")
            + (f"_hidden-{self.embedding_dim}" if self.embedding_dim is not None else "")
        )

    def prepare_dirs(self, delete=False):
        if delete:
            if os.path.exists(self.bert_dir):
                logger.info(f"{self.bert_dir} exists, removing...")
                shutil.rmtree(self.bert_dir)
            if os.path.exists(self.experiment_dir):
                logger.info(f"{self.experiment_dir} exists, removing...")
                shutil.rmtree(self.experiment_dir)
        os.makedirs(self.bert_dir, exist_ok=True)

    def prepare_bert_pretrain_env_vars(self):
        os.environ["TOKENIZER_PATH"] = self.bert_dir
        os.environ["NUM_LAYERS"] = str(self.num_layers)
        os.environ["NUM_ATTENTION_HEADS"] = str(self.num_attention_heads)
        os.environ["EMBEDDING_DIM"] = str(self.embedding_dim)
        # Discard any pretraining paths we don't need
        xpos = mlm = parser = False
        for k, v in self.pretrain_language_config.items():
            os.environ[k] = json.dumps(v)
            if k == "train_data_paths":
                xpos = "xpos" in v
                mlm = "mlm" in v
                parser = "parser" in v
        os.environ["XPOS"] = json.dumps(xpos)
        os.environ["MLM"] = json.dumps(mlm)
        os.environ["PARSER"] = json.dumps(parser)


@click.command(help="Pretrain a monolingual BERT")
@click.pass_context
def train(ctx):
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


@click.command(help="Evaluate a monolingual BERT")
@click.pass_obj
def evaluate(config):
    _, eval_metrics = evaluate_allennlp(config, config.bert_dir)
    write_to_tsv(
        config,
        "-".join(config.tasks) + ("_ft" if config.finetune else ""),
        eval_metrics,
    )


bert.add_command(train)
bert.add_command(evaluate)
