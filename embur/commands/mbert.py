import click

from embur.commands.common import write_to_tsv, default_options
from embur.eval.allennlp import evaluate_allennlp
from embur.language_configs import get_eval_config


class MbertExperimentConfig:
    def __init__(self, language, **kwargs):
        self.language = language
        combined_kwargs = default_options(mbert)
        combined_kwargs.update(kwargs)

        self.bert_model_name = combined_kwargs.pop("bert_model_name")
        self.parser_eval_language_config = get_eval_config(self.language, self.bert_model_name)
        self.parser_eval_jsonnet = combined_kwargs.pop("parser_eval_config")


@click.group(help="Use a pretrained BERT")
@click.pass_context
@click.option("--bert-model-name", "-m", default="bert-base-multilingual-cased")
@click.option(
    "--parser-eval-config",
    default="configs/parser_eval.jsonnet",
    help="Parser evaluation config. You probably want to leave this as the default.",
)
def mbert(ctx, **kwargs):
    ctx.obj.experiment_config = MbertExperimentConfig(language=ctx.obj.language, **kwargs)


@click.command(help="Run a baseline eval for a given language")
@click.pass_context
def evaluate(ctx):
    config = ctx.obj
    _, metrics = evaluate_allennlp(config)
    name = config.bert_model_name
    name += "_ft" if config.finetune else ""
    write_to_tsv(config, name, metrics)


mbert.add_command(evaluate)
