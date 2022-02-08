import os
from glob import glob
from shutil import copyfile, rmtree

from conllu import TokenList
from nltk.tokenize import wordpunct_tokenize
from rich import print

import embur.scripts.common as eso


def postprocess_docs(docs):
    new_docs = []
    for meta, doc in docs:
        new_doc = []
        for line in doc:
            new_doc.append(wordpunct_tokenize(line))
        new_docs.append([meta, new_doc])
    return new_docs


def make_tokenlists(docs):
    doc_tls = []
    for meta, doc in docs:
        id = meta["id"]
        url = meta["url"]
        title = meta["title"]
        sent_id = lambda i: f"{id}_{title}-{str(i).zfill(3)}"
        tls = []
        for line_num, line in enumerate(doc):
            tokens = [eso.token() for _ in line]
            for i, t in enumerate(line):
                tokens[i]['form'] = t
            if line_num == 0:
                meta = {"sent_id": sent_id(line_num + 1), "newdoc id": f"{id}_{title}", "url": url}
            else:
                meta = {"sent_id": sent_id(line_num + 1)}
            tls.append(TokenList(tokens, meta))
        doc_tls.append(tls)
    return doc_tls


def parse_wowiki():
    with open(f'{CORPORA_DIR}/wowiki_plain.xml') as f:
        lines = [l.strip() for l in f.read().split("\n")]

    docs = []
    doc = None
    for line in lines:
        if eso.ttline_is_open_tag(line):
            _, attrs = eso.ttline_parse_open_tag(line)
            id = attrs["id"]
            url = attrs["url"]
            title = attrs["title"]
            doc = [{"id": id, "url": url, "title": title}, []]
        elif eso.ttline_is_close_tag(line):
            if doc is not None:
                docs.append(doc)
            doc = None
        elif doc is not None:
            if line != "" and not line.isspace():
                doc[1].append(eso.unescape_xml(line))
    docs = postprocess_docs(docs)
    doc_tls = make_tokenlists(docs)
    return doc_tls


def parse_tt(filepath):
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f.read().split("\n")]

    sentence = None
    doc = None
    docs = []
    for i, line in enumerate(lines):
        if line[:5] == "<?xml":
            continue
        elif eso.ttline_is_open_tag(line):
            ename, attrs = eso.ttline_parse_open_tag(line)
            if ename == "s":
                sentence = []
            elif ename == "article":
                doc = [{"title": attrs["article_id"] if "article_id" in attrs else attrs["title"]}, []]
        elif eso.ttline_is_close_tag(line):
            if line[:5] == "</art":
                docs.append(doc)
                sentence = None
                doc = None
            elif line[:4] == "</s>":
                doc[1].append(sentence)
                sentence = None
        elif line != "":
            form, xpos = line.split("\t")
            token = eso.token()
            token['form'] = form
            token['xpos'] = xpos
            sentence.append(token)

    doc_tls = []
    for dmeta, doc in docs:
        tls = []
        for i, line in enumerate(doc):
            meta = {"sent_id": dmeta["title"] + "-" + str(i + 1).zfill(3)}
            if i == 0:
                meta["newdoc id"] = dmeta["title"]
            tls.append(TokenList(line, meta))
        doc_tls.append(tls)

    return doc_tls


def parse_tt_bible(filepath):
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f.read().split("\n")]

    sentence = None
    doc = None
    docs = []
    for i, line in enumerate(lines):
        if line[:5] == "<?xml":
            continue
        elif eso.ttline_is_open_tag(line):
            ename, attrs = eso.ttline_parse_open_tag(line)
            if ename == "verse":
                sentence = []
            elif ename == "chapter":
                doc = [{"title": attrs["chapter_id"]}, []]
        elif eso.ttline_is_close_tag(line):
            if line[:5] == "</cha":
                docs.append(doc)
                sentence = None
                doc = None
            elif line[:8] == "</verse>":
                doc[1].append(sentence)
                sentence = None
        elif line != "":
            form, xpos = line.split("\t")
            token = eso.token()
            token['form'] = form
            token['xpos'] = xpos
            sentence.append(token)

    doc_tls = []
    for dmeta, doc in docs:
        tls = []
        for i, line in enumerate(doc):
            meta = {"sent_id": dmeta["title"] + "-" + str(i + 1).zfill(3)}
            if i == 0:
                meta["newdoc id"] = dmeta["title"]
            tls.append(TokenList(line, meta))
        doc_tls.append(tls)

    return doc_tls


def parse_tts():
    web_tls = parse_tt(f"{CORPORA_DIR}/wolof_web.tt")
    gospels_tls = parse_tt_bible(f"{CORPORA_DIR}/wolof_gospels.tt")
    wiki_tls = parse_tt(f"{CORPORA_DIR}/wolof_wiki.tt")
    return web_tls + gospels_tls + wiki_tls


def main():
    wowiki_tls = parse_wowiki()
    tt_tls = parse_tts()
    doc_tls = wowiki_tls + tt_tls
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
    CORPORA_DIR = "data/wolof/corpora/"
    OUTPUT_DIR = "data/wolof/converted/"
    rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR + "/train", exist_ok=True)
    os.makedirs(OUTPUT_DIR + "/dev", exist_ok=True)
    main()
