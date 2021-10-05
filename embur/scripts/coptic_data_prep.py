import os
import sys
import re
import zipfile
from collections import OrderedDict, defaultdict
from glob import glob
from shutil import rmtree, copyfile

import conllu
from allennlp_models.structured_prediction import UniversalDependenciesDatasetReader
from lxml import etree
from tqdm import tqdm

ELT_REGEX = re.compile(r'<([a-zA-Z][a-zA-Z0-9_]*)')
ATTR_REGEX = re.compile(r'(?:[a-zA-Z][a-zA-Z0-9_]*:)?([a-zA-Z][a-zA-Z0-9_]*)="([^"]*)"')
EMPTY_TOKEN_DICT = {field_name: "_" for field_name in conllu.parser.DEFAULT_FIELDS}


def is_token(ttsgml_line):
    return not (ttsgml_line.startswith("<") and ttsgml_line.endswith(">"))


def parse_open_tag(ttsgml_line):
    try:
        element_name = re.search(ELT_REGEX, ttsgml_line).groups()[0]
    except Exception as e:
        print("!!!")
        print(ttsgml_line)
        raise e
    attrs = re.findall(ATTR_REGEX, ttsgml_line)
    unescape = lambda s: s.replace('&lt;', "<").replace("&gt;", ">").replace('&quot;', '"').replace("&apos;", "'").replace("&amp;", "&")
    attrs = [(k, unescape(v)) for k, v in attrs]
    return element_name, OrderedDict(attrs)


def encode_entities(sentence, scheme="BIO"):
    entity_spans = defaultdict(list)
    etypes = {}
    for i, token in enumerate(sentence):
        if 'Entity' in token['misc']:
            entities = [e for e in token['misc']['Entity']]# if 'abstract' not in e]
            for e in entities:
                entity_id, entity_type = e.split("_")
                entity_spans[entity_id].append(i)
                etypes[entity_id] = entity_type
        token['misc'] = {"Entity": "O"}

    # data has nested entities, but we just want simple non-nested entities
    # iteratively take the shortest entities available until you've taken as
    # much as possible without overlapping spans
    covered = []
    accepted = []
    for entity_id, token_ids in sorted(list(entity_spans.items()), key=lambda x:len(x[1])):
        if not any(token_id in covered for token_id in token_ids):
            accepted.append(entity_id)
            covered += token_ids

    entity_spans = {k: v for k, v in entity_spans.items() if k in accepted}
    for eid, tids in entity_spans.items():
        for i, tid in enumerate(tids):
            if i == 0:
                tag = "B-" + etypes[eid]
            else:
                tag = "I-" + etypes[eid]
            sentence[tid]['misc'] = {"Entity": tag}


def conllize(ttsgml):
    sentences = []
    sentence = []
    entity_open = []
    meta = None

    def finalize_sentence(sentence):
        encode_entities(sentence)
        sent_meta = {
            "sent_id": meta['document_cts_urn'][18:] + "-" + str(len(sentences) + 1),
            "doc_id": meta['document_cts_urn'],
            "segmentation": meta['segmentation'],
            "tagging": meta['tagging'],
            "parsing": meta.get('parsing', 'none'),
            "entities": meta.get('entities', 'none'),
        }
        tl = conllu.TokenList(sentence, sent_meta)
        sentences.append(tl.serialize())

    for line_num, line in enumerate(ttsgml.replace("\r", "").split("\n")):
        # Read the <meta> element at the beginning of the document
        if meta is None:
            element_name, attrs = parse_open_tag(line)
            assert line_num == 0 and element_name == 'meta'
            meta = attrs
            continue

        # Ignore orig tokens
        if is_token(line):
            pass

        # If we're at a closing tag...
        elif not is_token(line) and line[:2] == '</':
            # ... and it's an entity that's being closed, close the entity
            if line.strip() == '</entity>':
                entity_open.pop()

        # If we're at an opening tag...
        elif not is_token(line):
            elt, attrs = parse_open_tag(line)
            # TODO: extract entities
            if " translation=" in line and len(sentence) > 0:
                finalize_sentence(sentence)
                sentence = []
            elif elt == 'norm':
                token = EMPTY_TOKEN_DICT.copy()
                token['id'] = len(sentence) + 1
                token['form'] = attrs['norm']
                token['xpos'] = attrs['pos']
                token['lemma'] = attrs['lemma']
                # if 'head' in attrs:
                #     token['deprel'] = attrs['func']
                #     token['head'] = int(attrs['head'][2:])
                if len(entity_open) > 0:
                    token['misc'] = {"Entity": entity_open.copy()}
                sentence.append(token)
            elif elt == 'entity':
                entity_open.append(str(line_num) + "_" + attrs['entity'])
        else:
            raise Exception('Not a tag or a token: "' + line + '"')

    # flush the last sentence
    if len(sentence) > 0:
        finalize_sentence(sentence)

    print(f"Found {sum(len(s) for s in sentences)} tokens in {meta['document_cts_urn']}")
    return meta, "\n".join(sentences)


def format_tt(tt_dir):
    tt_data = []
    if tt_dir.endswith('.zip'):
        with zipfile.ZipFile(tt_dir) as zipf:
            for tt_filepath in [filepath for filepath in zipf.namelist() if filepath.endswith('tt')]:
                with zipf.open(tt_filepath) as f:
                    tt_data.append((f"{tt_dir[:-4]}/{tt_filepath}", f.read().decode('utf8')))
    else:
        for tt_filepath in sorted(glob(f"{tt_dir}/*.tt")):
            with open(tt_filepath) as f:
                tt_data.append((f"{tt_dir}/{tt_filepath}", f.read()))
    return tt_data


if __name__ == '__main__':
    OUTPUT_DIR = "data/coptic/converted/"
    SYNTAX_DIR = OUTPUT_DIR + "syntax/"

    rmtree(OUTPUT_DIR, ignore_errors=True)
    rmtree(SYNTAX_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR + "/train", exist_ok=True)
    os.makedirs(OUTPUT_DIR + "/dev", exist_ok=True)
    os.makedirs(OUTPUT_DIR + "/test", exist_ok=True)
    os.makedirs(SYNTAX_DIR, exist_ok=True)

    SPLIT_MAP = {
        "test": [
            "ap.31.monbeg.conllu",
            "besa.aphthonia.monbba.conllu",
            "nt.mark.sahidica:10.conllu",
            "nt.mark.sahidica:11.conllu",
            "nt.mark.sahidica:12.conllu",
            "nt.mark.sahidica:13.conllu",
            "nt.mark.sahidica:14.conllu",
            "nt.mark.sahidica:15.conllu",
            "nt.mark.sahidica:16.conllu",
            "nt.mark.sahidica:1.conllu",
            "nt.mark.sahidica:2.conllu",
            "nt.mark.sahidica:3.conllu",
            "nt.mark.sahidica:4.conllu",
            "nt.mark.sahidica:5.conllu",
            "nt.mark.sahidica:6.conllu",
            "nt.mark.sahidica:7.conllu",
            "nt.mark.sahidica:8.conllu",
            "nt.mark.sahidica:9.conllu",
            "shenoute.fox.monbxh_204_216.conllu"
        ],
        "dev": [
            "besa.exhortations.monbba.conllu",
            "shenoute.eagerness.monbgf_17_18.conllu",
            "shenoute.eagerness.monbgf_31_32.conllu",
            "shenoute.eagerness.monbgl_29_30.conllu",
            "shenoute.eagerness.monbgl_29.conllu",
            "shenoute.eagerness.monbgl_45_53.conllu",
            "shenoute.eagerness.monbgl_53_54.conllu",
            "shenoute.eagerness.monbgl_54_60.conllu",
            "shenoute.eagerness.monbgl_61_63.conllu",
            "shenoute.eagerness.monbgl_63_70.conllu",
            "shenoute.eagerness.monbgl_70.conllu",
            "shenoute.eagerness.monbgl_9_10.conllu",
            "shenoute.eagerness.monbgl_d.conllu",
            "shenoute.eagerness.monbxj_33_34.conllu",
            "shenoute.eagerness.monbxj_49_52.conllu",
            "shenoute.eagerness.monbxj_52_62.conllu",
            "shenoute.eagerness.monbxj_65_76.conllu",
            "shenoute.eagerness.monbxj_77_86.conllu",
            "shenoute.seeks.monbcz:26-36.conllu",
            "shenoute.unknown5_1.monbgf:101-102.conllu",
            "shenoute.unknown5_1.monbgf:131-139.conllu"
        ]
    }

    tt_data = []
    # conllu_data = []
    for corpus_dir in sorted(glob('data/coptic/corpora/*')):
        tt_dir = list(glob(f"{corpus_dir}/*_TT*"))
        tt_dir = tt_dir[0] if len(tt_dir) > 0 else None
        if tt_dir:
            tt_data += format_tt(tt_dir)

        conllu_dir = list(glob(f"{corpus_dir}/*_CONLLU*"))
        conllu_dir = conllu_dir[0] if len(conllu_dir) > 0 else None
        if conllu_dir:
            for conllu_filepath in sorted(glob(f"{conllu_dir}/*.conllu")):
                conllu_filename = conllu_filepath.split(os.sep)[-1]
                new_filepath = SYNTAX_DIR + os.sep + conllu_filename
                copyfile(conllu_filepath, new_filepath)
                # with open(conllu_filepath, 'r') as f:
                #     conllu_data.append((f"{conllu_dir}/{conllu_filepath}", f.read()))

    for filepath, tt_str in tqdm(tt_data):
        meta, conllu_string = conllize(tt_str)
        document_name = meta['document_cts_urn'][18:] + ".conllu"
        if document_name in SPLIT_MAP['test']:
            path = os.path.join(OUTPUT_DIR, "test", document_name)
        elif document_name in SPLIT_MAP['dev']:
            path = os.path.join(OUTPUT_DIR, "dev", document_name)
        else:
            path = os.path.join(OUTPUT_DIR, "train", document_name)
        with open(path, 'w') as f:
            f.write(conllu_string)
