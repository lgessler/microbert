import os
from glob import glob
from shutil import rmtree, copyfile

from rich import print
from tqdm import tqdm
import embur.scripts.common as eso


def format_wowiki():
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
            doc = []
        elif eso.ttline_is_close_tag(line):
            if doc is not None:
                docs.append(doc)
                print(doc)
            doc = None
        elif doc is not None:
            if line != "" and not line.isspace():
                doc.append(eso.unescape_xml(line))
    #TODO: should the words be simply tokenized? (wordpiece tokenizer will do work anyway...)


if __name__ == '__main__':
    CORPORA_DIR = "data/wolof/corpora/"
    OUTPUT_DIR = "data/wolof/converted/"
    rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR + "/train", exist_ok=True)
    os.makedirs(OUTPUT_DIR + "/dev", exist_ok=True)
    os.makedirs(OUTPUT_DIR + "/test", exist_ok=True)

    format_wowiki()
