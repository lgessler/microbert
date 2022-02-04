import math
import re
from collections import OrderedDict

import conllu
import random

ELT_REGEX = re.compile(r'<([a-zA-Z][a-zA-Z0-9_]*)')
ATTR_REGEX = re.compile(r'(?:[a-zA-Z][a-zA-Z0-9_]*:)?([a-zA-Z][a-zA-Z0-9_]*)="([^"]*)"')
EMPTY_TOKEN_DICT = {field_name: "_" for field_name in conllu.parser.DEFAULT_FIELDS}


def unescape_xml(s):
    s = s.replace("&quot;", '"')
    s = s.replace("&lt;", '<')
    s = s.replace("&gt;", '>')
    s = s.replace("&amp;", '&')
    s = s.replace("&apos;", "'")
    return s


def ttline_is_token(ttsgml_line):
    return not (ttsgml_line.startswith("<") and ttsgml_line.endswith(">"))


def ttline_is_close_tag(ttsgml_line):
    return not ttline_is_token(ttsgml_line) and ttsgml_line[:2] == "</"


def ttline_is_open_tag(ttsgml_line):
    return not ttline_is_token(ttsgml_line) and not ttline_is_close_tag(ttsgml_line)


def ttline_parse_open_tag(ttsgml_line):
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


def number(doc_tls):
    for tls in doc_tls:
        for tl in tls:
            i = 1
            for token in tl:
                token['id'] = i
                i += 1


def get_splits(xs, proportions):
    """
    Split a sequence into deterministically randomized splits with size indicated in proportions.
    Deterministically randomized in the sense that for fixed inputs,
    the same randomized sequences will always be returned.
    """
    random.seed(1337)
    count = len(xs)
    indices = list(range(count))
    random.shuffle(indices)

    splits = []
    i = 0
    for p in proportions:
        split = []
        j = math.ceil(p*count)
        for index in indices[i:i+j]:
            split.append(xs[index])
        splits.append(split)
        i = j

    return splits


def token():
    return EMPTY_TOKEN_DICT.copy()
