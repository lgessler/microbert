import re
from collections import OrderedDict

import conllu

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


def token():
    return EMPTY_TOKEN_DICT.copy()
