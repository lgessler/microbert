import logging
import os
import time
from glob import glob
from random import randint, random
from typing import Dict, List

import torch
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, MetadataField, SequenceLabelField, TensorField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from conllu import TokenList, parse
from overrides import overrides

logger = logging.getLogger(__name__)


def read_conllu_file(file_path: str) -> List[TokenList]:
    document = []
    with open(file_path, "r") as file:
        contents = file.read()
        tokenlists = parse(contents)
        if len(tokenlists) == 0:
            print(f"WARNING: {file_path} is empty--likely conversion error.")
            return []
        for annotation in tokenlists:
            m = annotation.metadata
            if len(annotation) > 200:
                subannotation = TokenList(annotation[:200])
                subannotation.metadata = annotation.metadata.copy()
                logger.info(f"Breaking up huge sentence in {file_path} with length {len(annotation)} "
                            f"into chunks of 200 norms")
                while len(subannotation) > 0:
                    document.append(subannotation)
                    subannotation = TokenList(subannotation[200:])
                    subannotation.metadata = annotation.metadata.copy()
            else:
                document.append(annotation)
    return document


def read_conllu_files(file_path: str) -> List[List[TokenList]]:
    if file_path.endswith('.conllu'):
        file_paths = [file_path]
    else:
        file_paths = sorted(glob(os.path.join(file_path, "*.conllu")))

    documents = []
    for conllu_file_path in file_paths:
        document = read_conllu_file(conllu_file_path)
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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.tokenizer = tokenizer
        self.whitespace_tokenizer = WhitespaceTokenizer()

    @overrides
    def _read(self, file_path: str):
        documents = read_conllu_files(file_path)
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

    @overrides
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
        if lemmas is not None:
            fields["lemmas"] = SequenceLabelField(lemmas, text_field, label_namespace="lemmas")
        if xpos_tags is not None and not all(t is None for t in xpos_tags):
            fields["xpos_tags"] = SequenceLabelField(xpos_tags, text_field, label_namespace="xpos_tags")
            metadata["xpos"] = xpos_tags
        if upos_tags is not None and not all(t is None for t in upos_tags):
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
