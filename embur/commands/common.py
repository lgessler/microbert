import click
from filelock import FileLock


def default_options(command: click.Command):
    params = {p.name: (p.default if hasattr(p, "default") else None) for p in command.params}
    return params


def write_to_tsv(config, name, metrics):
    output = "\t".join(
        [
            config.language,
            name,
            "_",
            "_",
            str(metrics["LAS"]),
        ]
    )
    locked_write("metrics.tsv", output + "\n")


def locked_write(filepath, s):
    lock = FileLock(filepath + ".lock")
    with lock, open(filepath, "a") as f:
        f.write(s)
