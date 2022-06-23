"""
Some wrapper functions for Diaparser.
For an explanation of params, see diaparser.cmds.biaffine_dependency
"""
import torch
from diaparser.parsers.biaffine_dependency import BiaffineDependencyParser as Parser
from diaparser.utils import Config


def train(
    path,
    train,
    dev,
    test=None,
    feat=None,
    build=False,
    punct=False,
    partial=False,
    tree=False,
    proj=False,
    max_len=None,
    buckets=32,
    embed=None,
    unk="[UNK]",
    n_word_embed=None,
    bert=None,
    attention_head=None,
    attention_layer=None,
    batch_size=5000,
    epochs=5000,
):
    assert feat in [None, "tag", "char", "bert"]
    assert max_len is None or isinstance(max_len, int)
    assert n_word_embed is None or isinstance(n_word_embed, int)
    assert attention_head is None or isinstance(attention_head, int)
    assert attention_layer is None or isinstance(attention_layer, int)
    torch.manual_seed(1337)
    # needed?
    # torch.set_num_threads(-1)
    args = dict(**locals())
    for key in ["n_word_embed", "attention_head", "attention_layer"]:
        if args[key] is None:
            del args[key]
    args = Config(**args)
    parser = Parser.build(**args)
    return parser.train(**args)


def evaluate(path, data, punct=False, buckets=8, batch_size=5000, tree=False, proj=False, partial=False):
    torch.manual_seed(1337)
    args = dict(**locals())
    args = Config(**args)
    parser = Parser.load(args.path)
    return parser.evaluate(**args)
