from math import ceil
from random import shuffle

import click

from embur.language_configs import get_wikiann_path, get_formatted_wikiann_path
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans, to_bioul


@click.group
@click.pass_obj
def data(ctx):
    pass


@click.command(help="Prepare MLM data for a given language")
@click.pass_obj
def prepare_mlm(config):
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


def _parse_ner(path):
    with open(path, "r") as f:
        s = f.read()
    all_sentence_lines = [sentence.split("\n") for sentence in s.strip().split("\n\n")]
    sentences = []

    i = 0
    for sentence_lines in all_sentence_lines:
        sentence = []
        i += 1
        for line in sentence_lines:
            i += 1
            pieces = line.split(" ")
            if len(pieces) == 3:
                form, _, tag = pieces
            elif len(pieces) == 7:
                form, slug, canonical_slug, entity_type, confidence, _, tag = pieces
            else:
                raise ValueError(f"Malformed line at {i}! {pieces}")
            sentence.append((form, tag))
        sentences.append(sentence)
    return sentences


def _format_conll2003(sentences):
    formatted_sentences = [
        "\n".join([" ".join([form, "O", "O", bio_tag]) for form, bio_tag in sentence]) for sentence in sentences
    ]
    return "\n\n".join(formatted_sentences)


def _bio_to_bioul(sentences):
    bioul_sentences = []
    for sentence in sentences:
        bioul_tags = to_bioul([tag for form, tag in sentence], encoding="BIO")
        bioul_sentence = list(zip([form for form, _ in sentence], bioul_tags))
        bioul_sentences.append(bioul_sentence)
        if len(bioul_sentence) > 100:
            print(len(bioul_sentence))
            print(bioul_sentence)
            print()
    return bioul_sentences


def _split_ner(sentences):
    shuffle(sentences)
    indexes = [ceil(len(sentences) * 0.8), ceil(len(sentences) * 0.9)]
    return sentences[: indexes[0]], sentences[indexes[0] : indexes[1]], sentences[indexes[1] :]


@click.command(help="Prepare wikiann data for a given language")
@click.pass_obj
def prepare_ner(config):
    ner_path = get_wikiann_path(config.language)
    sentences = _parse_ner(ner_path)
    sentences = _bio_to_bioul(sentences)
    train, dev, test = _split_ner(sentences)
    with open(get_formatted_wikiann_path(config.language, "train"), "w") as f:
        f.write(_format_conll2003(train))
    with open(get_formatted_wikiann_path(config.language, "dev"), "w") as f:
        f.write(_format_conll2003(dev))
    with open(get_formatted_wikiann_path(config.language, "test"), "w") as f:
        f.write(_format_conll2003(test))


data.add_command(prepare_mlm)
data.add_command(prepare_ner)
