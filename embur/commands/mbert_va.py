import json
import logging
import os
import shutil
from collections import OrderedDict

import click
from allennlp.commands.train import train_model_from_file
from transformers import BertModel, BertTokenizer

from embur.commands.common import write_to_tsv, default_options
from embur.dataset_reader import read_conllu_files
from embur.eval.allennlp import evaluate_allennlp
from embur.language_configs import get_eval_config, get_pretrain_config
from embur.tokenizers import train_tokenizer


logger = logging.getLogger(__name__)


class MbertVaExperimentConfig:
    def __init__(self, language, **kwargs):
        self.language = language
        combined_kwargs = default_options(mbert_va)
        combined_kwargs.update(kwargs)

        self.bert_model_name = combined_kwargs.pop("bert_model_name")
        self.embedding_dim = BertModel.from_pretrained(self.bert_model_name).config.hidden_size
        self.new_wordpiece_count = combined_kwargs.pop("new_wordpiece_count")

        self.pretrain_language_config = get_pretrain_config(self.language, self.initial_bert_dir, ["mlm"])
        self.pretrain_jsonnet = combined_kwargs.pop("training_config")
        self.parser_eval_language_config = get_eval_config(self.language, self.bert_dir)
        self.parser_eval_jsonnet = combined_kwargs.pop("parser_eval_config")

    @property
    def initial_bert_dir(self):
        return f"berts/{self.language}/{self.bert_model_name}_va_initial"

    @property
    def bert_dir(self):
        return f"berts/{self.language}/{self.bert_model_name}_va"

    @property
    def experiment_dir(self):
        return f"models/{self.language}/{self.bert_model_name}_va"

    def prepare_dirs(self, delete=False):
        if delete:
            if os.path.exists(self.bert_dir):
                logger.info(f"{self.bert_dir} exists, removing...")
                shutil.rmtree(self.bert_dir)
            if os.path.exists(self.initial_bert_dir):
                logger.info(f"{self.initial_bert_dir} exists, removing...")
                shutil.rmtree(self.initial_bert_dir)
            if os.path.exists(self.experiment_dir):
                logger.info(f"{self.experiment_dir} exists, removing...")
                shutil.rmtree(self.experiment_dir)
        os.makedirs(self.bert_dir, exist_ok=True)
        os.makedirs(self.initial_bert_dir, exist_ok=True)

    def prepare_bert_pretrain_env_vars(self):
        os.environ["embedding_dim"] = str(self.embedding_dim)
        os.environ["bert_model"] = self.initial_bert_dir
        for k, v in self.pretrain_language_config.items():
            os.environ[k] = json.dumps(v)


@click.group(
    help=(
        'Implementation of Chau et al. (2020)\'s "vocabulary augmentation" method, where a pretrained'
        " BERT has its vocabulary extended with a set of new pieces and is pretrained for a short time"
        " using this new vocabulary."
    )
)
@click.pass_context
@click.option("--bert-model-name", "-m", default="bert-base-multilingual-cased")
@click.option("--new-wordpiece-count", "-n", type=int, default=99, help="Number of new wordpieces to add")
@click.option(
    "--training-config",
    default="configs/mbert_va.jsonnet",
    help="Training config. You probably want to leave this as the default.",
)
@click.option(
    "--parser-eval-config",
    default="configs/parser_eval.jsonnet",
    help="Parser evaluation config. You probably want to leave this as the default.",
)
def mbert_va(ctx, **kwargs):
    ctx.obj.experiment_config = MbertVaExperimentConfig(language=ctx.obj.language, **kwargs)


def _augment_vocabulary(base_tokenizer, mono_tokenizer, n):
    if n > 99:
        raise ValueError("Augmenting with more than 99 new wordpieces is not supported.")
    base_vocab = base_tokenizer.vocab
    mono_vocab = mono_tokenizer.get_vocab()

    assert base_vocab["[PAD]"] == 0, "Expected [PAD] to be 0"
    assert base_vocab["[UNK]"] == 100, "Expected [UNK] to be 100"
    for i in range(1, 100):
        assert base_vocab[f"[unused{i}]"] == i, f"Expected [unused{i}] at {i}"

    # Loop over the mono WPs and try to meet the requested number of new wordpieces
    new_wps = []
    base_wps = list(base_vocab.keys())
    mono_wps = [x[0] for x in sorted(list(mono_vocab.items()), key=lambda x: x[1])]
    i = 0
    while len(new_wps) < n and i < len(mono_wps):
        mono_wp = mono_wps[i]
        if mono_wp[0] != "[" and mono_wp not in base_wps:
            new_wps.append(mono_wp)
        i += 1
    if len(new_wps) < n:
        logger.warning(
            f"{n} new wordpieces were requested for vocabulary augmentation, but only {len(new_wps)}"
            f" were able to be added. Proceeding anyway."
        )

    # Rewrite the base vocabulary with the new wordpieces replacing unused ones
    output_vocab = OrderedDict()
    output_vocab["[PAD]"] = 0
    logger.info("Beginning vocabulary augmentation...")
    for i, wp in enumerate(new_wps):
        print(f"Replacing [unused{i+1}] with {wp}")
        output_vocab[wp] = i + 1
    for i in range(len(new_wps) + 1, 100):
        print(f"Keeping [unused{i}]")
        output_vocab[f"[unused{i}]"] = i
    for wp, index in base_vocab.items():
        if index < 100:
            continue
        output_vocab[wp] = index

    base_tokenizer.vocab = output_vocab


def _prepare_modified_model(config):
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
    model = BertModel.from_pretrained(config.bert_model_name)

    # Get config and train tokenizer
    logger.info("Training tokenizer...")
    documents = read_conllu_files(config.pretrain_language_config["tokenizer_conllu_path"])
    sentences = [" ".join([t["form"] for t in sentence]) for document in documents for sentence in document]
    new_tokenizer = train_tokenizer(sentences, model_type="wordpiece", vocab_size=5_000)

    # Augment vocabulary and save a local copy of both
    _augment_vocabulary(tokenizer, new_tokenizer, config.new_wordpiece_count)
    logger.info(f"Writing augmented models to {config.initial_bert_dir}")
    model.save_pretrained(config.initial_bert_dir)
    tokenizer.save_pretrained(config.initial_bert_dir)
    # Tokenizer doesn't change, so also put it where the finetuned model will be
    tokenizer.save_pretrained(config.bert_dir)


@click.command(help="Take a pretrained model and pretrain it more with vocabulary augmentation.")
@click.pass_context
def train(ctx):
    config = ctx.obj
    config.prepare_dirs(delete=True)

    _prepare_modified_model(config)

    # these are needed by mbert_va.jsonnet
    config.prepare_bert_pretrain_env_vars()

    # Train the LM
    logger.info("Beginning pretraining...")
    logger.info("Config:\n", config.pretrain_language_config)
    logger.info("Env:\n", os.environ)
    overrides = '{"trainer.num_epochs": 1, "data_loader.instances_per_epoch": 256}' if config.debug else ""
    model = train_model_from_file(config.pretrain_jsonnet, config.experiment_dir, overrides=overrides)
    bert_serialization: BertModel = model._backbone.bert
    bert_serialization.save_pretrained(config.bert_dir)
    with open(os.path.join(config.experiment_dir, "metrics.json"), "r") as f:
        train_metrics = json.load(f)


@click.command(help="Run an mbert_va evaluation.")
@click.pass_context
def evaluate(ctx):
    config = ctx.obj
    _, metrics = evaluate_allennlp(config, config.bert_dir)
    name = config.bert_model_name + "_va"
    name += "_ft" if config.finetune else ""
    write_to_tsv(config, name, metrics)


mbert_va.add_command(train)
mbert_va.add_command(evaluate)
