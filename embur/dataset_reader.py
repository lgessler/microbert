import logging
import os
from functools import reduce
from glob import glob
from typing import Dict, List
import unicodedata

import tokenizers as T
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, MetadataField, SequenceLabelField, TensorField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer, PretrainedTransformerMismatchedIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer, PretrainedTransformerTokenizer
from conllu import TokenList, parse
from transformers import BertTokenizer


logger = logging.getLogger(__name__)
MAX_TOKEN_LENGTH = 200
MAX_WORDPIECE_LENGTH = 350
MAX_PIECES_IN_TOKEN = 100


def remove_huge_tokens(document, tokenizer):
    for sentence in document:
        for i in range(len(sentence)):
            if len(tokenizer.tokenize(sentence[i]['form'])) > MAX_PIECES_IN_TOKEN:
                logger.info(f"[UNK]ing out huge token: `{sentence[i]['form']}`")
                sentence[i]['form'] = "[UNK]"


def read_conllu_file(file_path: str, tokenizer: T.Tokenizer = None) -> List[TokenList]:
    document = []
    with open(file_path, "r") as file:
        contents = file.read()
    sentences = parse(contents)
    if tokenizer is not None:
        remove_huge_tokens(sentences, tokenizer)
    if len(sentences) == 0:
        print(f"WARNING: {file_path} is empty--likely conversion error.")
        return []

    vocab = None
    indexer = None
    if tokenizer is not None:
        path = tokenizer.name_or_path
        indexer = PretrainedTransformerMismatchedIndexer(path, "tokens")
        vocab = Vocabulary.from_pretrained_transformer(path)

    for sentence in sentences:
        m = sentence.metadata

        # Need to check that sentences are not too long.
        # First use a subpiece tokenizer if we have one
        if indexer is not None and vocab is not None:
            get_chunks(document, sentence, indexer, vocab)
        # Fall
        elif len(sentence) > MAX_TOKEN_LENGTH:
            subannotation = TokenList(sentence[:MAX_TOKEN_LENGTH])
            subannotation.metadata = sentence.metadata.copy()
            logger.info(f"Breaking up huge sentence in {file_path} with length {len(sentence)} "
                        f"into chunks of {MAX_TOKEN_LENGTH} tokens")
            while len(subannotation) > 0:
                document.append(subannotation)
                subannotation = TokenList(subannotation[MAX_TOKEN_LENGTH:])
                subannotation.metadata = sentence.metadata.copy()
        else:
            document.append(sentence)
    return document


def get_chunks(document, sentence, indexer, vocab):
    metadata = sentence.metadata

    tokens = [Token(token['form']) for token in sentence]

    sentence_chunks = []
    d = indexer.tokens_to_indices(tokens, vocab)
    offsets = d["offsets"]
    # pieces = [vocab.get_token_from_index(tid) for tid in d["token_ids"]]
    start, i = 0, 0
    chunk = []
    for t_b, t_e in offsets:
        # note: these are 1-indexed, inclusive indices
        if t_e - start > MAX_WORDPIECE_LENGTH:
            sentence_chunks.append(chunk)
            start = t_b
            chunk = []
        chunk.append(sentence[i])
        i += 1
    if len(chunk) > 0:
        sentence_chunks.append(chunk)

    if len(sentence_chunks) > 1:
        msg = "Split sentence into chunks:\n"
        for chunk in sentence_chunks:
            msg += f"\t{len(chunk)}, {' '.join([t['form'] for t in chunk[:5]])} ...\n"
        logger.info(msg)

    for chunk in sentence_chunks:
        chunk = TokenList(chunk, metadata=metadata)
        document.append(chunk)


def read_conllu_files(file_path: str, tokenizer: T.Tokenizer = None) -> List[List[TokenList]]:
    if file_path.endswith('.conllu'):
        file_paths = [file_path]
    else:
        file_paths = sorted(glob(os.path.join(file_path, "*.conllu")))

    documents = []
    for conllu_file_path in file_paths:
        document = read_conllu_file(conllu_file_path, tokenizer=tokenizer)
        doclen = sum(len(sentence) for sentence in document)
        if doclen == 0:
            print(conllu_file_path, "has length", doclen)
        else:
            documents.append(document)
    return documents


@DatasetReader.register("embur_conllu", exist_ok=True)
class EmburConllu(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer: Tokenizer = None,
        huggingface_tokenizer_path: str = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.tokenizer = tokenizer
        self.whitespace_tokenizer = WhitespaceTokenizer()

        # Load the HF tokenizer we'll use later in the bert backbone so that we can use it to gauge sentence length
        self.allennlp_tokenizer = None
        self.huggingface_tokenizer = None
        if huggingface_tokenizer_path is not None:
            self.huggingface_tokenizer = BertTokenizer.from_pretrained(huggingface_tokenizer_path)
            self.allennlp_tokenizer = PretrainedTransformerTokenizer(huggingface_tokenizer_path)

    def _read(self, file_path: str):
        documents = read_conllu_files(file_path, tokenizer=self.huggingface_tokenizer)
        token_count = 0
        for document in documents:
            document_count = 0
            for sentence in document:
                document_count += len(sentence)
            token_count += document_count
        print(f"\n\nTotal token count for {file_path} split: {token_count}\n\n")

        sample_printed = False
        yielded = 0

        for document in documents:
            for sentence in document:
                if 'TOY_DATA' in os.environ and yielded >= 100:
                    continue
                m = sentence.metadata
                # Only accept plain tokens
                sentence = [a for a in sentence if isinstance(a['id'], int)]

                # read directly from conllu output
                # not used: feats, deps
                forms = [x["form"] for x in sentence]
                xpos_tags = [x["xpos"] for x in sentence]
                upos_tags = [x["upos"] for x in sentence]
                lemmas = [x["lemma"] for x in sentence]
                heads, deprels = None, None
                if all(x["head"] is not None for x in sentence) and all(
                    x["deprel"] is not None for x in sentence
                ):
                    heads = [int(x["head"]) for x in sentence]
                    deprels = [str(x["deprel"]) for x in sentence]


                wp_len = len(self.allennlp_tokenizer.intra_word_tokenize(forms))
                assert wp_len <= MAX_WORDPIECE_LENGTH, (wp_len, MAX_WORDPIECE_LENGTH, m)

                instance = self.text_to_instance(
                    forms=forms,
                    xpos_tags=xpos_tags,
                    upos_tags=upos_tags,
                    lemmas=lemmas,
                    heads=heads,
                    deprels=deprels,
                )
                if not sample_printed:
                    print(f"Sample instance from {file_path}:", instance)
                    sample_printed = True

                yielded += 1

                yield instance

    def text_to_instance(
        self,  # type: ignore
        forms: List[str],
        lemmas: List[str] = None,
        upos_tags: List[str] = None,
        xpos_tags: List[str] = None,
        heads: List[int] = None,
        deprels: List[str] = None,
    ) -> Instance:
        fields: Dict[str, Field] = {}

        if self.tokenizer is not None:
            tokens = self.tokenizer.tokenize(" ".join(forms))
        else:
            tokens = [Token(t) for t in forms]

        # standard conll
        text_field = TextField(tokens, self._token_indexers)
        fields["text"] = text_field
        metadata = {"text": forms}
        if lemmas is not None and not all(l is None or l == "_" for l in lemmas):
            fields["lemmas"] = SequenceLabelField(lemmas, text_field, label_namespace="lemmas")
        if xpos_tags is not None and not all(t is None or t == "_" for t in xpos_tags):
            fields["xpos_tags"] = SequenceLabelField(xpos_tags, text_field, label_namespace="xpos_tags")
            metadata["xpos"] = xpos_tags
        if upos_tags is not None and not all(t is None or t == "_" for t in upos_tags):
            fields["upos_tags"] = SequenceLabelField(upos_tags, text_field, label_namespace="upos_tags")
            metadata["upos"] = upos_tags
        if heads is not None and deprels is not None:
            # fields["dependencies"] = AdjacencyField(
            #     [(i, j) for j, i in enumerate(heads) if i >= 0], text_field, deprels, "deprel_labels"
            # )
            fields["heads"] = SequenceLabelField(heads, text_field)
            fields["deprels"] = SequenceLabelField(deprels, text_field, label_namespace="deprel_labels")

        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)
