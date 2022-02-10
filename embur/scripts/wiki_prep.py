"""
Generic
"""
import os
import re
from glob import glob
from unicodedata import category as unicode_category

import bleach
import click
from bs4 import BeautifulSoup
from conllu import TokenList
from rich import print
from rich.progress import track
from spacy.pipeline.sentencizer import Sentencizer

import embur.scripts.common as esc

FULL_STOPS = Sentencizer.default_punct_chars + ["\n", "\r", "\r\n", "\n\r"]


def read_file(filepath):
    filename = filepath.split(os.sep)[-1]
    with open(filepath, 'r') as f:
        html = f.read()
        # Remove all tags except those in the list, while keeping their contents
        html = bleach.clean(html, tags=["text", "figure", "caption", "p"], strip=True)
        # Prettify the HTML to get linebreaks between the remaining elements, which are all elements
        # which surely imply a sentence boundary
        prettified = BeautifulSoup(html, features="html.parser").prettify()
        # Prettify returns a string, so parse it again
        soup = BeautifulSoup(prettified, features="html.parser")
        # equiv to js .innerHtml
        text = soup.getText()
        text = text.replace("\r\n", "\n")
        text = text.replace("\n\r", "\n")
        text = text.replace("\r", "\n")
    return filename, text


def read_dir(input_dir):
    try:
        filepaths = sorted(glob(f"{input_dir}/*.html"), key=lambda x: int(x.split("__")[1].replace(".html", "")))
    except:
        print("Failed to sort documents numerically, falling back to lexicographic sort")
        filepaths = sorted(glob(f"{input_dir}/*.html"))
    return [read_file(fp) for fp in filepaths]


@click.group(help="Use the appropriate subcommand for the kind of tokenization you wish to use.")
def top():
    pass


def filter_sents(sents):
    return [s for s in sents if s.strip() != ""]


def ssplit_by_punct(text):
    if len(text) == 0:
        return []

    # modified form of https://github.com/explosion/spaCy/blob/3877f78ff9f406a148e27a16ee60a7778bc5a551/spacy/pipeline/sentencizer.pyx#L91L119
    # iterate char by char and break on punct
    splits = []
    seen_full_stop = False
    start = 0
    for i, c in enumerate(text):
        is_punct = unicode_category(c)[0] == "P"
        is_full_stop = c in FULL_STOPS
        # Always split on a linebreak
        if c == "\n":
            splits.append(start)
            start = i
            seen_full_stop = False
        # If we've seen a full stop, split on the first char that isn't punct
        elif seen_full_stop and not is_punct and not is_full_stop:
            splits.append(start)
            start = i
            seen_full_stop = False
        # Notice full stops but hold off for now
        elif is_full_stop:
            seen_full_stop = True

    # Construct strings from the break indices
    sents = []
    begin = splits[0]
    for end in (splits + [len(text)]):
        sents.append(text[begin:end].strip())
        begin = end

    return filter_sents(sents)


multispace_pattern = re.compile(r"\s+")


def whitespace_tokenize_sents(sents):
    return [[t for t in re.sub(multispace_pattern, " ", s).split(" ") if not t.strip().isspace()] for s in sents]


def sents_to_tokenlists(dname, sents):
    tls = []
    for sent_num, sentence in enumerate(sents):
        dname = dname.replace(".html", "")
        tokens = [esc.token() for _ in range(len(sentence))]
        meta = {"sent_id": dname + "-" + str(sent_num + 1)}
        if sent_num == 0:
            meta["newdoc id"] = dname
        for i, (word, token) in enumerate(zip(sentence, tokens)):
            token["form"] = word
        tls.append(TokenList(tokens, meta))
    return tls


@click.command(help="")
@click.argument("input_dir")
@click.argument("output_dir")
def punct(input_dir, output_dir):
    docs = read_dir(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    doc_tls = []
    for dname, dtext in track(docs):
        sents = ssplit_by_punct(dtext)
        sents = whitespace_tokenize_sents(sents)
        tls = sents_to_tokenlists(dname, sents)
        doc_tls.append(tls)
    
    train_tls, dev_tls = esc.get_splits(doc_tls, proportions=[0.9, 0.1])
    train_tc = sum(sum(len(tl) for tl in tls) for tls in train_tls)
    dev_tc = sum(sum(len(tl) for tl in tls) for tls in dev_tls)
    esc.number(train_tls)
    esc.number(dev_tls)

    print(f"Split: train {train_tc}, dev {dev_tc}")

    os.makedirs(output_dir + "/train", exist_ok=True)
    os.makedirs(output_dir + "/dev", exist_ok=True)
    with open(f"{output_dir}/train/train.conllu", 'w') as f:
        f.write("".join(tl.serialize() for doc in train_tls for tl in doc))
    with open(f"{output_dir}/dev/dev.conllu", 'w') as f:
        f.write("".join(tl.serialize() for doc in dev_tls for tl in doc))


@click.command(help="")
@click.argument("input_dir")
@click.argument("output_dir")
def neural(input_dir, output_dir):
    pass


@click.command(help="")
@click.argument("input_dir")
@click.argument("output_dir")
@click.option("-m", "--model-name", default="xlm-roberta-base")
def transformer(input_dir, output_dir, model_name):
    pass


top.add_command(punct)
top.add_command(neural)
top.add_command(transformer)


if __name__ == '__main__':
    top()