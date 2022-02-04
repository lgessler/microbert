import os
from glob import glob
from shutil import rmtree

from bs4 import BeautifulSoup
from conllu import TokenList
from betacode.conv import beta_to_uni
from rich.progress import track

import embur.scripts.common as eso


def file_to_tokenlists(filepath):
    with open(filepath, 'r') as f:
        contents = f.read()
    soup = BeautifulSoup(contents, features="html.parser")
    title = soup.find("titlestmt").find("title").text
    title_id = soup.find("tlgid").text

    sentences = soup.findAll("sentence")
    tls = []

    for sent_num, sentence in enumerate(sentences):
        sentence_id = sentence["id"]
        words = sentence.findAll("word")
        tokens = [eso.token() for _ in range(len(words))]
        meta = {"sent_id": sentence_id}
        if sent_num == 0:
            meta["newdoc id"] = f"{title_id} - {title}"
        for i, (word, token) in enumerate(zip(words, tokens)):
            token["form"] = beta_to_uni(word["form"])
            lemma = word.find("lemma")
            token["lemma"] = lemma["entry"] if "entry" in lemma else "_"
            token["xpos"] = lemma["pos"] if "pos" in lemma else "_"
        tls.append(TokenList(tokens, meta))

    return tls


def main():
    filepaths = sorted(glob(f"{CORPORA_DIR}/*.xml"))
    doc_tls = []
    for filepath in track(filepaths):
        tls = file_to_tokenlists(filepath)
        doc_tls.append(tls)

    train_tls, dev_tls = eso.get_splits(doc_tls, proportions=[0.9, 0.1])
    train_tc = sum(sum(len(tl) for tl in tls) for tls in train_tls)
    dev_tc = sum(sum(len(tl) for tl in tls) for tls in dev_tls)
    eso.number(train_tls)
    eso.number(dev_tls)

    print(f"Split: train {train_tc}, dev {dev_tc}")

    with open(f"{OUTPUT_DIR}/train/train.conllu", 'w') as f:
        f.write("".join(tl.serialize() for doc in train_tls for tl in doc))
    with open(f"{OUTPUT_DIR}/dev/dev.conllu", 'w') as f:
        f.write("".join(tl.serialize() for doc in dev_tls for tl in doc))


if __name__ == '__main__':
    CORPORA_DIR = "data/greek/corpora/"
    OUTPUT_DIR = "data/greek/converted/"
    rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR + "/train", exist_ok=True)
    os.makedirs(OUTPUT_DIR + "/dev", exist_ok=True)
    main()
