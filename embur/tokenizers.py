import os
from typing import List
from tokenizers import Tokenizer
from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import BertTokenizer


# https://huggingface.co/docs/tokenizers/python/latest/pipeline.html
from tokenizers.trainers import WordPieceTrainer


def train_tokenizer(sentences: List[str], serialize_path: str = "", vocab_size: int = 6000) -> Tokenizer:
    bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    bert_tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    bert_tokenizer.pre_tokenizer = Whitespace()
    bert_tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    bert_tokenizer.train_from_iterator(sentences, trainer=trainer)
    if serialize_path:
        bert_tokenizer.save_model(serialize_path)
    return bert_tokenizer


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
    bert_tokenizer = BertTokenizer.from_pretrained(serialize_path + os.sep + "vocab.txt")
    bert_tokenizer.save_pretrained(serialize_path)
    os.rename(
        serialize_path + os.sep + "tokenizer_config.json",
        serialize_path + os.sep + "config.json"
    )
    return bert_tokenizer
