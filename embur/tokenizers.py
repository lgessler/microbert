import os
from typing import List

from tokenizers import Tokenizer
from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, Lowercase, Sequence, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
# https://huggingface.co/docs/tokenizers/python/latest/pipeline.html
from tokenizers.trainers import WordPieceTrainer
from transformers import BertTokenizer, PreTrainedTokenizerFast


def write_vocab(tokenizer: Tokenizer, serialization_dir: str):
    vocab = [(w, i) for w, i in tokenizer.get_vocab().items()]
    vocab = sorted(vocab, key=lambda x: x[1])
    assert [i for _, i in vocab] == list(range(vocab[-1][1] + 1)), "Vocabulary is not monotonic!"
    words = "\n".join([w for w, _ in vocab]) + "\n"
    with open(os.path.join(serialization_dir, "vocab.txt"), 'w') as f:
        f.write(words)


def train_tokenizer(sentences: List[str], serialize_path: str = "", vocab_size: int = 6000) -> Tokenizer:
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    unk_token = "[UNK]"
    pad_token = "[PAD]"
    mask_token = "[MASK]"
    special_tokens = [pad_token, cls_token, sep_token, unk_token, mask_token]

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single=f"{cls_token} $A {sep_token}",
        pair=f"{cls_token} $A {sep_token} $B:1 {sep_token}:1",
        special_tokens=[
            (cls_token, 1),
            (sep_token, 2),
        ],
    )
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    tokenizer.train_from_iterator(sentences, trainer=trainer)
    if serialize_path:
        full_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            cls_token=cls_token,
            sep_token=sep_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token
        )
        full_tokenizer.save_pretrained(serialize_path)
        write_vocab(tokenizer, serialize_path)
    return tokenizer


def train_bert_tokenizer(sentences: List[str], serialize_path: str, vocab_size: int = 6000) -> BertWordPieceTokenizer:
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False,
    )
    tokenizer.train_from_iterator(
        sentences,
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        limit_alphabet=500,
        wordpieces_prefix="##",
    )

    # Save the files--first write out the vocab, then use BertTokenizer's save_pretrained
    tokenizer.save_model(serialize_path)
    bert_tokenizer = BertTokenizer.from_pretrained(serialize_path)
    bert_tokenizer.save_pretrained(serialize_path)
    #os.rename(
    #    serialize_path + os.sep + "tokenizer_config.json",
    #    serialize_path + os.sep + "config.json"
    #)
    return bert_tokenizer
