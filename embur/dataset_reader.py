import os
import time
from glob import glob
from random import random, randint
from typing import Dict, List
import logging

import torch
from overrides import overrides
from conllu import parse, TokenList

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, TensorField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer


logger = logging.getLogger(__name__)


def read_conllu_file(file_path: str, seg_threshold: bool = True) -> List[TokenList]:
    document = []
    with open(file_path, "r") as file:
        contents = file.read()
        tokenlists = parse(contents)
        if len(tokenlists) == 0:
            print(f"WARNING: {file_path} is empty--likely conversion error.")
            return []
        for annotation in tokenlists:
            m = annotation.metadata
            if seg_threshold and "segmentation" in m and m["segmentation"] not in ["checked", "gold"]:
                print("Skipping " + file_path + " because its segmentation is not checked or gold.")
                return []
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


def read_conllu_files(file_path: str, seg_threshold: bool = True) -> List[List[TokenList]]:
    if file_path.endswith('.conllu'):
        file_paths = [file_path]
    else:
        file_paths = sorted(glob(os.path.join(file_path, "*.conllu")))

    documents = []
    for conllu_file_path in file_paths:
        document = read_conllu_file(conllu_file_path, seg_threshold=seg_threshold)
        doclen = sum(len(sentence) for sentence in document)
        if doclen == 0:
            print(conllu_file_path, "has length", doclen)
        else:
            documents.append(document)
    return documents


@DatasetReader.register("coptic_conllu", exist_ok=True)
class CopticConllu(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        seg_threshold: bool = False,
        tokenizer: Tokenizer = None,
        read_entities: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.seg_threshold = seg_threshold
        self.tokenizer = tokenizer
        self.whitespace_tokenizer = WhitespaceTokenizer()
        self.read_entities = read_entities

    @overrides
    def _read(self, file_path: str):
        documents = read_conllu_files(file_path, seg_threshold=self.seg_threshold)
        token_count = 0
        for document in documents:
            document_count = 0
            for sentence in document:
                document_count += len(sentence)
            token_count += document_count
        print(f"\n\nTotal token count for {file_path} split: {token_count}\n\n")
        for document in documents:
            for sentence in document:
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

                # non-conllu info: entities, which are hanging out in misc
                entity_tags = None
                if self.read_entities and "entities" in m and m['entities'] in ["checked", "gold"]:
                    entity_tags = [x["misc"]["Entity"] for x in sentence]
                    lemmas = None
                    xpos_tags = None
                    upos_tags = None
                    heads = None
                    deprels = None
                elif self.read_entities:
                    continue
                yield self.text_to_instance(
                    forms=forms,
                    xpos_tags=xpos_tags,
                    upos_tags=upos_tags,
                    lemmas=lemmas,
                    heads=heads,
                    deprels=deprels,
                    entity_tags=entity_tags
                )

    def add_masked_fields(
        self,
        tokens: List[Token],
        fields: Dict[str, Field]
    ):
        masked = [True if random() < 0.15 else False for _ in range(len(tokens))]
        if not any(x for x in masked):
            masked[randint(0, len(masked) - 1)] = True
        masked_tokens = [t if not masked[i] else Token("[MASK]") for i, t in enumerate(tokens)]
        masked_positions = [i for i in range(len(masked)) if masked[i]]
        fields["masked_text"] = TextField(masked_tokens, self._token_indexers)
        fields["masked_positions"] = TensorField(torch.tensor(masked_positions))
        fields["true_masked_ids"] = TextField([t for i, t in enumerate(tokens) if masked[i]], self._token_indexers)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        forms: List[str],
        lemmas: List[str] = None,
        upos_tags: List[str] = None,
        xpos_tags: List[str] = None,
        heads: List[int] = None,
        deprels: List[str] = None,
        entity_tags: List[str] = None,
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
        if xpos_tags is not None:
            fields["xpos_tags"] = SequenceLabelField(xpos_tags, text_field, label_namespace="xpos_tags")
            metadata["xpos"] = xpos_tags
        if upos_tags is not None:
            fields["upos_tags"] = SequenceLabelField(upos_tags, text_field, label_namespace="upos_tags")
            metadata["upos"] = upos_tags
        if heads is not None and deprels is not None:
            # fields["dependencies"] = AdjacencyField(
            #     [(i, j) for j, i in enumerate(heads) if i >= 0], text_field, deprels, "deprel_labels"
            # )
            fields["heads"] = SequenceLabelField(heads, text_field)
            fields["deprels"] = SequenceLabelField(deprels, text_field, label_namespace="deprel_labels")

        # extra goodness
        # self.add_masked_fields(tokens, fields)
        if entity_tags is not None:
            fields["entity_tags"] = SequenceLabelField(entity_tags, text_field, label_namespace="entity_tags")
            metadata["entities"] = entity_tags

        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)
