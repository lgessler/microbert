import logging
import os

import click
from gensim.models import Word2Vec

from embur.commands.common import write_to_tsv, default_options
from embur.dataset_reader import read_conllu_files
from embur.eval.allennlp import evaluate_allennlp_static
from embur.language_configs import get_eval_config, get_pretrain_config

logger = logging.getLogger(__name__)


class Word2vecExperimentConfig:
    def __init__(self, language, **kwargs):
        self.language = language
        combined_kwargs = default_options(word2vec)
        combined_kwargs.update(kwargs)

        self.embedding_dim = combined_kwargs.pop("embedding_dim")
        self.parser_eval_language_config = get_eval_config(self.language, None)
        self.parser_eval_jsonnet = combined_kwargs.pop("parser_eval_config")

    @property
    def word2vec_file(self):
        return f"word2vec/{self.language}_{self.embedding_dim}.vec"


@click.group()
@click.option("--embedding-dim", default=100, type=int, help="word2vec hidden dimension")
@click.option(
    "--parser-eval-config",
    default="configs/parser_eval.jsonnet",
    help="Parser evaluation config. You probably want to leave this as the default.",
)
@click.pass_context
def word2vec(ctx, **kwargs):
    ctx.obj.experiment_config = Word2vecExperimentConfig(language=ctx.obj.language, **kwargs)


@click.command(help="Train word2vec embeddings")
@click.pass_obj
def train(config):
    documents = read_conllu_files(get_pretrain_config(config.language, None, [])["tokenizer_conllu_path"])
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


@click.command(help="Evaluate word2vec embeddings")
@click.pass_obj
def evaluate(config):
    config.parser_eval_jsonnet = "configs/static_parser_eval.jsonnet"
    train_metrics, eval_metrics = evaluate_allennlp_static(config)
    write_to_tsv(config, "word2vec" + ("_ft" if config.finetune else ""), eval_metrics)


word2vec.add_command(train)
word2vec.add_command(evaluate)
