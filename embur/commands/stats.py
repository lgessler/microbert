from collections import defaultdict

import click

from pathlib import Path
import logging

from embur.dataset_reader import read_conllu_files
from embur.language_configs import LANGUAGES

logger = logging.getLogger(__name__)


@click.group(help="Helpers for calculating stats")
def stats(**kwargs):
    pass


@click.command
def unlabeled():
    print(
        "Language\tTrain sentences\tTrain tokens\tUnique train tokens\tDev sentences\tDev tokens\tUnique dev tokens\tUnique total tokens"
    )
    for language in LANGUAGES:
        base_path = f"data/{language}/converted_punct"
        if language in ["coptic", "greek"]:
            base_path = base_path[:-6]
        train_sents = [s for d in read_conllu_files(f"{base_path}/train") for s in d]
        dev_sents = [s for d in read_conllu_files(f"{base_path}/dev") for s in d]
        train_tokens = [str(t) for s in train_sents for t in s]
        dev_tokens = [str(t) for s in dev_sents for t in s]
        print(
            "\t".join(
                map(
                    str,
                    [
                        language,
                        len(train_sents),
                        len(train_tokens),
                        len(set(train_tokens)),
                        len(dev_sents),
                        len(dev_tokens),
                        len(set(dev_tokens)),
                        len(set(train_tokens) | set(dev_tokens)),
                    ],
                )
            )
        )


@click.command(help="Format a metrics.tsv and average language-condition pairs")
@click.option("--tsv-path", default="metrics.tsv")
@click.option("--expected-trials", type=int, default=5)
def format_metrics(tsv_path, expected_trials):
    if not Path(tsv_path).exists():
        raise ValueError(f"Tsv not found at {tsv_path}")
    with open(tsv_path, "r") as f:
        lines = f.read().strip().split("\n")

    scores = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    for i, line in enumerate(lines):
        pieces = line.split("\t")
        if len(pieces) != 5:
            raise ValueError(f"Malformed TSV line at line {i+1} in {tsv_path}: {pieces}")
        language, condition, _, _, test_las = pieces
        scores[language][condition] += float(test_las)
        counts[language][condition] += 1

    all_conditions = []
    for language, conditions in scores.items():
        for condition, las in conditions.items():
            if condition not in all_conditions:
                all_conditions.append(condition)
            count = counts[language][condition]
            if count != expected_trials:
                logger.warning(f"Expected {expected_trials} trials for {(language, condition)} but found {count}")

    condition_rows = defaultdict(list)
    for language in sorted(list(scores.keys())):
        for condition in all_conditions:
            count = counts[language][condition]
            las = "_" if count == 0 else scores[language][condition] / count
            condition_rows[condition].append(las)

    print()
    print("\t".join([""] + sorted(list(scores.keys()))))
    for condition, values in condition_rows.items():
        print("\t".join([condition] + [str(v) for v in values]))
    print()


stats.add_command(unlabeled)
stats.add_command(format_metrics)
