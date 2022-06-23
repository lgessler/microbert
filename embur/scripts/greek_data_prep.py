import os
from glob import glob
from shutil import rmtree

from betacode.conv import beta_to_uni
from bs4 import BeautifulSoup
from conllu import TokenList, parse
from joblib import Parallel, delayed

import embur.scripts.common as esc


def file_to_tokenlists(filepath):
    with open(filepath, "r") as f:
        contents = f.read()
    soup = BeautifulSoup(contents, features="html.parser")
    title = soup.find("titlestmt").find("title").text
    title_id = soup.find("tlgid").text

    sentences = soup.findAll("sentence")
    tls = []

    for sent_num, sentence in enumerate(sentences):
        sentence_id = sentence["id"]
        words = sentence.findAll("word")
        tokens = [esc.token() for _ in range(len(words))]
        meta = {"sent_id": title_id + " - " + title + " - " + sentence_id}
        if sent_num == 0:
            meta["newdoc id"] = f"{title_id} - {title}"
        for i, (word, token) in enumerate(zip(words, tokens)):
            token["form"] = beta_to_uni(word["form"])
            lemma = word.find("lemma")
            token["lemma"] = lemma["entry"] if "entry" in lemma else "_"
            token["xpos"] = lemma["pos"] if "pos" in lemma else "_"
        if len(tokens) > 0:
            tls.append(TokenList(tokens, meta))
        else:
            print(f"Skipping empty sentence at {title} - {sent_num}")

    return tls


# It's a long story--see PretrainedTransformerTokenizer._reverse_engineer_special_tokens
BOGUS_DOC = """# newdoc id = BOGUS
# sent_id = BOGUS - 1
1       a       _       _       _       _       _       _       _       _
2       b       _       _       _       _       _       _       _       _

"""


def main():
    filepaths = sorted(glob(f"{CORPORA_DIR}/*.xml"))

    def process_doc(filepath):
        print("Processed", filepath)
        return "".join(tl.serialize() for tl in file_to_tokenlists(filepath))

    doc_strs = Parallel(n_jobs=-1, prefer="processes", verbose=6)(
        delayed(process_doc)(filepath) for filepath in filepaths
    )
    doc_tls = [parse(s) for s in doc_strs]

    train_tls, dev_tls = esc.get_splits(doc_tls, proportions=[0.9, 0.1])
    train_tc = sum(sum(len(tl) for tl in tls) for tls in train_tls)
    dev_tc = sum(sum(len(tl) for tl in tls) for tls in dev_tls)
    esc.number(train_tls)
    esc.number(dev_tls)

    print(f"Split: train {train_tc}, dev {dev_tc}")

    with open(f"{OUTPUT_DIR}/train/train.conllu", "w") as f:
        f.write("".join(tl.serialize() for doc in train_tls for tl in doc) + BOGUS_DOC)
    with open(f"{OUTPUT_DIR}/dev/dev.conllu", "w") as f:
        f.write("".join(tl.serialize() for doc in dev_tls for tl in doc) + BOGUS_DOC)


if __name__ == "__main__":
    CORPORA_DIR = "data/greek/corpora/"
    OUTPUT_DIR = "data/greek/converted/"
    rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR + "/train", exist_ok=True)
    os.makedirs(OUTPUT_DIR + "/dev", exist_ok=True)
    main()
