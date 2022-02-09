from typing import List

from tokenizers.implementations import BertWordPieceTokenizer
from transformers import AutoTokenizer

from embur.dataset_reader import read_conllu_files

documents = read_conllu_files('../data/coptic/converted/train')

sentences = []
for document in documents:
    for sentence in document:
        sentences.append(" ".join([t['form'] for t in sentence]))


from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer


# https://huggingface.co/docs/tokenizers/python/latest/pipeline.html
def train_tokenizer(sentences: List[str], serialize_path: str = "", vocab_size: int = 8000) -> Tokenizer:
    bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    bert_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
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
        bert_tokenizer.save(serialize_path)
    return bert_tokenizer



ids = bert_tokenizer.encode(sentences[10]).ids
bert_tokenizer.decode(ids)


from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

tokenizer = Tokenizer(models.Unigram())
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoders = decoders.ByteLevel()

trainer = trainers.UnigramTrainer(
    vocab_size=20000,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["<PAD>", "<BOS>", "<EOS>"],
)

tokenizer.train_from_iterator(sentences, trainer=trainer)
tokenizer.encode(sentences[4]).ids
tokenizer.decode(tokenizer.encode(sentences[4]).ids)
tokenizer.save('bert_out/test2')

tokenizer.save_pretrained('bert_out/test')
