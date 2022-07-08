import click
from filelock import FileLock


def default_options(command: click.Command):
    params = {p.name: (p.default if hasattr(p, "default") else None) for p in command.params}
    return params


def write_to_tsv(config, name, metrics, filepath="metrics.tsv", key="LAS"):
    output = "\t".join(
        [
            config.language,
            name,
            "_",
            "_",
            str(metrics[key]),
        ]
    )
    locked_write(filepath, output + "\n")


def write_to_tsv2(config, name, metrics, filepath="ner_metrics.tsv", key="f1-measure-overall"):
    output = "\t".join(
        [
            config.language,
            name,
            "_",
            "_",
            str(metrics[key]),
        ]
    )
    locked_write(filepath, output + "\n")


def locked_write(filepath, s):
    lock = FileLock(filepath + ".lock")
    with lock, open(filepath, "a") as f:
        f.write(s)
