from rich import print
import click
from allennlp.common.util import import_module_and_submodules

import_module_and_submodules("allennlp_models")

import embur.commands.mbert as mbert
from embur.commands.mbert import mbert
import embur.commands.word2vec as word2vec
from embur.commands.word2vec import word2vec
import embur.commands.bert as bert
from embur.commands.bert import bert
from embur.config import Config
from embur.language_configs import LANGUAGES


@click.group()
@click.option(
    "--language",
    "-l",
    type=click.Choice(LANGUAGES),
    default="coptic",
    help="A language to train on.",
)
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


@click.command(help="Run all evals for a given language. Does not allow customization of hyperparams.")
@click.pass_context
def evaluate_all(ctx):
    config = ctx.obj

    # word2vec baseline
    config.experiment_config = word2vec.Word2vecExperimentConfig(language=config.language)
    config.finetune = False
    ctx.invoke(word2vec.evaluate)
    config.finetune = True
    ctx.invoke(word2vec.evaluate)

    # mBERT baseline
    config.experiment_config = mbert.MbertExperimentConfig(language=config.language)
    config.finetune = False
    ctx.invoke(mbert.evaluate)
    config.finetune = True
    ctx.invoke(mbert.evaluate)

    config.experiment_config = bert.BertExperimentConfig(language=config.language)
    for task_set in [["mlm"], ["mlm", "xpos"], ["mlm", "xpos", "parser"]]:
        config.experiment_config.set_tasks(task_set)
        config.finetune = False
        ctx.invoke(bert.evaluate)
        config.finetune = True
        ctx.invoke(bert.evaluate)


top.add_command(word2vec)
top.add_command(mbert)
top.add_command(bert)
top.add_command(prepare_data)
top.add_command(evaluate_all)


if __name__ == "__main__":
    top()
