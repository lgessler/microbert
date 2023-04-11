import stanza
from allennlp.common.checks import ConfigurationError
from allennlp.nn import util

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn.util import batched_index_select
from transformers import XLNetConfig

logger = logging.getLogger(__name__)

from typing import Dict, List, Any, Optional
import logging


import torch

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import PretrainedTransformerIndexer, TokenIndexer
from allennlp.data.token_indexers.token_indexer import IndexedTokenList

from embur.sla.sla import generate_sla_mask_list

logger = logging.getLogger(__name__)


def extend_tree_with_subword_edges(token_spans, head):
    flattened_token_spans = []
    for x, y in token_spans:
        flattened_token_spans.append(x)
        flattened_token_spans.append(y)
    token_spans = flattened_token_spans
    orig_head = head.copy()

    # Map from old token IDs to new token IDs after subword expansions
    id_map = {}
    running_diff = 0
    assert len(token_spans) // 2 == len(head) + 2, (len(token_spans) // 2, len(head))
    for token_id in range(0, len(token_spans) // 2):
        id_map[token_id] = token_id + running_diff
        b, e = token_spans[token_id * 2 : (token_id + 1) * 2]
        running_diff += e - b

    # Note how many subwords have been added so far
    new_token_spans = []
    for token_id in range(0, len(token_spans) // 2):
        # Inclusive indices of the subwords that the original token corresponds to
        b, e = token_spans[token_id * 2 : (token_id + 1) * 2]

        # If not a special token (we're assuming there are 2 on either end of the sequence),
        # replace the head value of the current token with the mapped value
        if token_id != 0 and token_id != ((len(token_spans) // 2) - 1):
            head[id_map[token_id] - 1] = id_map[orig_head[token_id - 1]]
        if e == b:
            # If we have a token that corresponds to a single subword, just append the same token_spans values
            new_token_spans.append(b)
            new_token_spans.append(e)
        else:
            # Note how many expansion subwords we'll add
            diff = e - b
            # This is the first subword in the token's index into head and deprel. Remember token_id is 1-indexed
            first_subword_index = id_map[token_id] - 1
            new_token_spans.append(b)
            new_token_spans.append(b)
            # For each expansion subword, add a separate token_spans entry and expand head and deprel.
            # Head's value is the ID of the first subword in the token it belongs to
            for j in range(1, diff + 1):
                new_token_spans.append(b + j)
                new_token_spans.append(b + j)
                head.insert(first_subword_index + j, id_map[token_id])

    heads = [int(x) for x in head]
    return heads


@TokenIndexer.register("pretrained_transformer_with_dep_att_mask")
class PretrainedTransformerWithDepAttMaskIndexer(TokenIndexer):
    """
    This `TokenIndexer` assumes that Tokens already have their indexes in them (see `text_id` field).
    We still require `model_name` because we want to form allennlp vocabulary from pretrained one.
    This `Indexer` is only really appropriate to use if you've also used a
    corresponding :class:`PretrainedTransformerTokenizer` to tokenize your input.  Otherwise you'll
    have a mismatch between your tokens and your vocabulary, and you'll get a lot of UNK tokens.

    Registered as a `TokenIndexer` with name "pretrained_transformer".

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use.
    namespace : `str`, optional (default=`tags`)
        We will add the tokens in the pytorch_transformer vocabulary to this vocabulary namespace.
        We use a somewhat confusing default value of `tags` so that we do not add padding or UNK
        tokens to this namespace, which would break on loading because we wouldn't find our default
        OOV token.
    max_length : `int`, optional (default = `None`)
        If not None, split the document into segments of this many tokens (including special tokens)
        before feeding into the embedder. The embedder embeds these segments independently and
        concatenate the results to get the original document representation. Should be set to
        the same value as the `max_length` option on the `PretrainedTransformerEmbedder`.
    tokenizer_kwargs : `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/tokenization_utils.py#L691)
        for `AutoTokenizer.from_pretrained`.
    """  # noqa: E501

    def __init__(
        self,
        model_name: str,
        namespace: str = "tags",
        max_length: int = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._namespace = namespace
        self._allennlp_tokenizer = PretrainedTransformerTokenizer(
            model_name, tokenizer_kwargs=tokenizer_kwargs
        )
        self._tokenizer = self._allennlp_tokenizer.tokenizer
        self._added_to_vocabulary = False

        self._num_added_start_tokens = len(self._allennlp_tokenizer.single_sequence_start_tokens)
        self._num_added_end_tokens = len(self._allennlp_tokenizer.single_sequence_end_tokens)

        self._max_length = max_length
        if self._max_length is not None:
            num_added_tokens = len(self._allennlp_tokenizer.tokenize("a")) - 1
            self._effective_max_length = (  # we need to take into account special tokens
                self._max_length - num_added_tokens
            )
            if self._effective_max_length <= 0:
                raise ValueError(
                    "max_length needs to be greater than the number of special tokens inserted."
                )

    def _add_encoding_to_vocabulary_if_needed(self, vocab: Vocabulary) -> None:
        """
        Copies tokens from ```transformers``` model's vocab to the specified namespace.
        """
        if self._added_to_vocabulary:
            return

        vocab.add_transformer_vocab(self._tokenizer, self._namespace)

        self._added_to_vocabulary = True

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, we don't need to do anything here.
        pass

    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> IndexedTokenList:
        self._add_encoding_to_vocabulary_if_needed(vocabulary)

        indices, type_ids = self._extract_token_and_type_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        output: IndexedTokenList = {
            "token_ids": indices,
            "mask": [True] * len(indices),
            "type_ids": type_ids or [0] * len(indices),
        }

        return self._postprocess_output(output)

    def indices_to_tokens(
        self, indexed_tokens: IndexedTokenList, vocabulary: Vocabulary
    ) -> List[Token]:
        self._add_encoding_to_vocabulary_if_needed(vocabulary)

        token_ids = indexed_tokens["token_ids"]
        type_ids = indexed_tokens.get("type_ids")

        return [
            Token(
                text=vocabulary.get_token_from_index(token_ids[i], self._namespace),
                text_id=token_ids[i],
                type_id=type_ids[i] if type_ids is not None else None,
            )
            for i in range(len(token_ids))
        ]

    def _extract_token_and_type_ids(self, tokens: List[Token]) -> Tuple[List[int], List[int]]:
        """
        Roughly equivalent to `zip(*[(token.text_id, token.type_id) for token in tokens])`,
        with some checks.
        """
        indices: List[int] = []
        type_ids: List[int] = []
        for token in tokens:
            indices.append(
                token.text_id
                if token.text_id is not None
                else self._tokenizer.convert_tokens_to_ids(token.text)
            )
            type_ids.append(token.type_id if token.type_id is not None else 0)
        return indices, type_ids

    def _postprocess_output(self, output: IndexedTokenList) -> IndexedTokenList:
        """
        Takes an IndexedTokenList about to be returned by `tokens_to_indices()` and adds any
        necessary postprocessing, e.g. long sequence splitting.

        The input should have a `"token_ids"` key corresponding to the token indices. They should
        have special tokens already inserted.
        """
        if self._max_length is not None:
            # We prepare long indices by converting them to (assuming max_length == 5)
            # [CLS] A B C [SEP] [CLS] D E F [SEP] ...
            # Embedder is responsible for folding this 1-d sequence to 2-d and feed to the
            # transformer model.
            # TODO(zhaofengw): we aren't respecting word boundaries when segmenting wordpieces.

            indices = output["token_ids"]
            type_ids = output.get("type_ids", [0] * len(indices))

            # Strips original special tokens
            indices = indices[
                self._num_added_start_tokens : len(indices) - self._num_added_end_tokens
            ]
            type_ids = type_ids[
                self._num_added_start_tokens : len(type_ids) - self._num_added_end_tokens
            ]

            # Folds indices
            folded_indices = [
                indices[i : i + self._effective_max_length]
                for i in range(0, len(indices), self._effective_max_length)
            ]
            folded_type_ids = [
                type_ids[i : i + self._effective_max_length]
                for i in range(0, len(type_ids), self._effective_max_length)
            ]

            # Adds special tokens to each segment
            folded_indices = [
                self._tokenizer.build_inputs_with_special_tokens(segment)
                for segment in folded_indices
            ]
            single_sequence_start_type_ids = [
                t.type_id for t in self._allennlp_tokenizer.single_sequence_start_tokens
            ]
            single_sequence_end_type_ids = [
                t.type_id for t in self._allennlp_tokenizer.single_sequence_end_tokens
            ]
            folded_type_ids = [
                single_sequence_start_type_ids + segment + single_sequence_end_type_ids
                for segment in folded_type_ids
            ]
            assert all(
                len(segment_indices) == len(segment_type_ids)
                for segment_indices, segment_type_ids in zip(folded_indices, folded_type_ids)
            )

            # Flattens
            indices = [i for segment in folded_indices for i in segment]
            type_ids = [i for segment in folded_type_ids for i in segment]

            output["token_ids"] = indices
            output["type_ids"] = type_ids
            output["segment_concat_mask"] = [True] * len(indices)

        return output

    def get_empty_token_list(self) -> IndexedTokenList:
        output: IndexedTokenList = {"token_ids": [], "mask": [], "type_ids": []}
        if self._max_length is not None:
            output["segment_concat_mask"] = []
        return output

    def as_padded_tensor_dict(
        self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        tensor_dict = {}
        for key, val in tokens.items():
            if key == "dep_att_mask":
                max_length = padding_lengths[key]
                tensor = torch.zeros((max_length, max_length))
                tensor[:len(val), :len(val)] = torch.tensor(val)
                tensor_dict[key] = tensor
                continue

            if key == "type_ids":
                padding_value = 0
                mktensor = torch.LongTensor
            elif key == "mask" or key == "wordpiece_mask":
                padding_value = False
                mktensor = torch.BoolTensor
            elif len(val) > 0 and isinstance(val[0], bool):
                padding_value = False
                mktensor = torch.BoolTensor
            else:
                padding_value = self._tokenizer.pad_token_id
                if padding_value is None:
                    padding_value = (
                        0  # Some tokenizers don't have padding tokens and rely on the mask only.
                    )
                mktensor = torch.LongTensor

            tensor = mktensor(
                pad_sequence_to_length(
                    val, padding_lengths[key], default_value=lambda: padding_value
                )
            )

            tensor_dict[key] = tensor
        return tensor_dict

    def __eq__(self, other):
        if isinstance(other, PretrainedTransformerIndexer):
            for key in self.__dict__:
                if key == "_tokenizer":
                    # This is a reference to a function in the huggingface code, which we can't
                    # really modify to make this clean.  So we special-case it.
                    continue
                if self.__dict__[key] != other.__dict__[key]:
                    return False
            return True
        return NotImplemented


@TokenIndexer.register("pretrained_transformer_mismatched_with_dep_att_mask")
class PretrainedTransformerMismatchedWithDepAttMaskIndexer(TokenIndexer):
    """
    Use this indexer when (for whatever reason) you are not using a corresponding
    `PretrainedTransformerTokenizer` on your input. We assume that you used a tokenizer that splits
    strings into words, while the transformer expects wordpieces as input. This indexer splits the
    words into wordpieces and flattens them out. You should use the corresponding
    `PretrainedTransformerMismatchedEmbedder` to embed these wordpieces and then pull out a single
    vector for each original word.

    Registered as a `TokenIndexer` with name "pretrained_transformer_mismatched".

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use.
    namespace : `str`, optional (default=`tags`)
        We will add the tokens in the pytorch_transformer vocabulary to this vocabulary namespace.
        We use a somewhat confusing default value of `tags` so that we do not add padding or UNK
        tokens to this namespace, which would break on loading because we wouldn't find our default
        OOV token.
    max_length : `int`, optional (default = `None`)
        If positive, split the document into segments of this many tokens (including special tokens)
        before feeding into the embedder. The embedder embeds these segments independently and
        concatenate the results to get the original document representation. Should be set to
        the same value as the `max_length` option on the `PretrainedTransformerMismatchedEmbedder`.
    tokenizer_kwargs : `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/tokenization_utils.py#L691)
        for `AutoTokenizer.from_pretrained`.
    """  # noqa: E501

    def __init__(
        self,
        model_name: str,
        stanza_language: str,
        stanza_use_mwt: bool,
        namespace: str = "tags",
        max_length: int = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        allow_retokenization: bool = False,
        max_distance: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # The matched version v.s. mismatched
        self._matched_indexer = PretrainedTransformerWithDepAttMaskIndexer(
            model_name,
            namespace=namespace,
            max_length=max_length,
            tokenizer_kwargs=tokenizer_kwargs,
            **kwargs,
        )
        self._allennlp_tokenizer = self._matched_indexer._allennlp_tokenizer
        self._tokenizer = self._matched_indexer._tokenizer
        self._num_added_start_tokens = self._matched_indexer._num_added_start_tokens
        self._num_added_end_tokens = self._matched_indexer._num_added_end_tokens
        self.max_distance = max_distance
        
        config = {
            "processors": "tokenize,mwt,pos,lemma,depparse" if stanza_use_mwt else "tokenize,pos,lemma,depparse",
            "lang": stanza_language,
            "use_gpu": True,
            "logging_level": "INFO",
            "tokenize_pretokenized": not allow_retokenization,
            "tokenize_no_ssplit": True,  # never allow sentence resegmentation
        }
        self.pipeline = stanza.Pipeline(**config)
        

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        return self._matched_indexer.count_vocab_items(token, counter)

    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> IndexedTokenList:
        self._matched_indexer._add_encoding_to_vocabulary_if_needed(vocabulary)

        wordpieces, offsets = self._allennlp_tokenizer.intra_word_tokenize(
            [t.ensure_text() for t in tokens]
        )

        # For tokens that don't correspond to any word pieces, we put (-1, -1) into the offsets.
        # That results in the embedding for the token to be all zeros.
        offsets = [x if x is not None else (-1, -1) for x in offsets]

        # custom code
        offsets = [(0, 0)] + offsets + [(offsets[-1][-1],) * 2]
        inputs = [stanza.Document([], text=[[t.text for t in tokens]])]
        sentence = self.pipeline(inputs)[0].to_dict()[0]
        tokens = [t for t in sentence if isinstance(t["id"], int)]
        head = [t["head"] for t in tokens]
        expanded_heads = extend_tree_with_subword_edges(offsets, head)
        dep_att_mask = generate_sla_mask_list(expanded_heads, max_distance=self.max_distance)

        output: IndexedTokenList = {
            "token_ids": [t.text_id for t in wordpieces],
            "mask": [True] * len(tokens),  # for original tokens (i.e. word-level)
            "type_ids": [t.type_id for t in wordpieces],
            "offsets": offsets,
            "wordpiece_mask": [True] * len(wordpieces),  # for wordpieces (i.e. subword-level)
            "dep_att_mask": dep_att_mask.tolist()
        }

        return self._matched_indexer._postprocess_output(output)

    def get_empty_token_list(self) -> IndexedTokenList:
        output = self._matched_indexer.get_empty_token_list()
        output["offsets"] = []
        output["wordpiece_mask"] = []
        return output

    def as_padded_tensor_dict(
        self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        tokens = tokens.copy()
        padding_lengths = padding_lengths.copy()

        offsets_tokens = tokens.pop("offsets")
        offsets_padding_lengths = padding_lengths.pop("offsets")

        tensor_dict = self._matched_indexer.as_padded_tensor_dict(tokens, padding_lengths)
        tensor_dict["offsets"] = torch.LongTensor(
            pad_sequence_to_length(
                offsets_tokens, offsets_padding_lengths, default_value=lambda: (0, 0)
            )
        )
        return tensor_dict

    def __eq__(self, other):
        if isinstance(other, PretrainedTransformerMismatchedWithDepAttMaskIndexer):
            for key in self.__dict__:
                if key == "_tokenizer":
                    # This is a reference to a function in the huggingface code, which we can't
                    # really modify to make this clean.  So we special-case it.
                    continue
                if self.__dict__[key] != other.__dict__[key]:
                    return False
            return True
        return NotImplemented


@TokenEmbedder.register("pretrained_transformer_with_dep_att_mask")
class PretrainedTransformerEmbedderWithDepAttMask(TokenEmbedder):
    """
    Uses a pretrained model from `transformers` as a `TokenEmbedder`.

    Registered as a `TokenEmbedder` with name "pretrained_transformer".

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use. Should be the same as the corresponding
        `PretrainedTransformerIndexer`.
    max_length : `int`, optional (default = `None`)
        If positive, folds input token IDs into multiple segments of this length, pass them
        through the transformer model independently, and concatenate the final representations.
        Should be set to the same value as the `max_length` option on the
        `PretrainedTransformerIndexer`.
    sub_module: `str`, optional (default = `None`)
        The name of a submodule of the transformer to be used as the embedder. Some transformers naturally act
        as embedders such as BERT. However, other models consist of encoder and decoder, in which case we just
        want to use the encoder.
    train_parameters: `bool`, optional (default = `True`)
        If this is `True`, the transformer weights get updated during training. If this is `False`, the
        transformer weights are not updated during training.
    eval_mode: `bool`, optional (default = `False`)
        If this is `True`, the model is always set to evaluation mode (e.g., the dropout is disabled and the
        batch normalization layer statistics are not updated). If this is `False`, such dropout and batch
        normalization layers are only set to evaluation mode when when the model is evaluating on development
        or test data.
    last_layer_only: `bool`, optional (default = `True`)
        When `True` (the default), only the final layer of the pretrained transformer is taken
        for the embeddings. But if set to `False`, a scalar mix of all of the layers
        is used.
    override_weights_file: `Optional[str]`, optional (default = `None`)
        If set, this specifies a file from which to load alternate weights that override the
        weights from huggingface. The file is expected to contain a PyTorch `state_dict`, created
        with `torch.save()`.
    override_weights_strip_prefix: `Optional[str]`, optional (default = `None`)
        If set, strip the given prefix from the state dict when loading it.
    reinit_modules: `Optional[Union[int, Tuple[int, ...], Tuple[str, ...]]]`, optional (default = `None`)
        If this is an integer, the last `reinit_modules` layers of the transformer will be
        re-initialized. If this is a tuple of integers, the layers indexed by `reinit_modules` will
        be re-initialized. Note, because the module structure of the transformer `model_name` can
        differ, we cannot guarantee that providing an integer or tuple of integers will work. If
        this fails, you can instead provide a tuple of strings, which will be treated as regexes and
        any module with a name matching the regex will be re-initialized. Re-initializing the last
        few layers of a pretrained transformer can reduce the instability of fine-tuning on small
        datasets and may improve performance (https://arxiv.org/abs/2006.05987v3). Has no effect
        if `load_weights` is `False` or `override_weights_file` is not `None`.
    load_weights: `bool`, optional (default = `True`)
        Whether to load the pretrained weights. If you're loading your model/predictor from an AllenNLP archive
        it usually makes sense to set this to `False` (via the `overrides` parameter)
        to avoid unnecessarily caching and loading the original pretrained weights,
        since the archive will already contain all of the weights needed.
    gradient_checkpointing: `bool`, optional (default = `None`)
        Enable or disable gradient checkpointing.
    tokenizer_kwargs: `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/tokenization_utils.py#L691)
        for `AutoTokenizer.from_pretrained`.
    transformer_kwargs: `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/modeling_utils.py#L253)
        for `AutoModel.from_pretrained`.
    """  # noqa: E501

    authorized_missing_keys = [r"position_ids$"]

    def __init__(
        self,
        model_name: str,
        *,
        max_length: int = None,
        sub_module: str = None,
        train_parameters: bool = True,
        eval_mode: bool = False,
        last_layer_only: bool = True,
        override_weights_file: Optional[str] = None,
        override_weights_strip_prefix: Optional[str] = None,
        reinit_modules: Optional[Union[int, Tuple[int, ...], Tuple[str, ...]]] = None,
        load_weights: bool = True,
        gradient_checkpointing: Optional[bool] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        from allennlp.common import cached_transformers

        self.transformer_model = cached_transformers.get(
            model_name,
            True,
            override_weights_file=override_weights_file,
            override_weights_strip_prefix=override_weights_strip_prefix,
            reinit_modules=reinit_modules,
            load_weights=load_weights,
            **(transformer_kwargs or {}),
        )

        if gradient_checkpointing is not None:
            self.transformer_model.config.update({"gradient_checkpointing": gradient_checkpointing})

        self.config = self.transformer_model.config
        if sub_module:
            assert hasattr(self.transformer_model, sub_module)
            self.transformer_model = getattr(self.transformer_model, sub_module)
        self._max_length = max_length

        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.config.hidden_size

        self._scalar_mix: Optional[ScalarMix] = None
        if not last_layer_only:
            self._scalar_mix = ScalarMix(self.config.num_hidden_layers)
            self.config.output_hidden_states = True

        tokenizer = PretrainedTransformerTokenizer(
            model_name,
            tokenizer_kwargs=tokenizer_kwargs,
        )

        try:
            if self.transformer_model.get_input_embeddings().num_embeddings != len(
                tokenizer.tokenizer
            ):
                self.transformer_model.resize_token_embeddings(len(tokenizer.tokenizer))
        except NotImplementedError:
            # Can't resize for transformers models that don't implement base_model.get_input_embeddings()
            logger.warning(
                "Could not resize the token embedding matrix of the transformer model. "
                "This model does not support resizing."
            )

        self._num_added_start_tokens = len(tokenizer.single_sequence_start_tokens)
        self._num_added_end_tokens = len(tokenizer.single_sequence_end_tokens)
        self._num_added_tokens = self._num_added_start_tokens + self._num_added_end_tokens

        self.train_parameters = train_parameters
        if not train_parameters:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

        self.eval_mode = eval_mode
        if eval_mode:
            self.transformer_model.eval()

    def train(self, mode: bool = True):
        self.training = mode
        for name, module in self.named_children():
            if self.eval_mode and name == "transformer_model":
                module.eval()
            else:
                module.train(mode)
        return self

    def get_output_dim(self):
        return self.output_dim

    def _number_of_token_type_embeddings(self):
        if isinstance(self.config, XLNetConfig):
            return 3  # XLNet has 3 type ids
        elif hasattr(self.config, "type_vocab_size"):
            return self.config.type_vocab_size
        else:
            return 0

    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
        dep_att_mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters

        token_ids: `torch.LongTensor`
            Shape: `[batch_size, num_wordpieces if max_length is None else num_segment_concat_wordpieces]`.
            num_segment_concat_wordpieces is num_wordpieces plus special tokens inserted in the
            middle, e.g. the length of: "[CLS] A B C [SEP] [CLS] D E F [SEP]" (see indexer logic).
        mask: `torch.BoolTensor`
            Shape: [batch_size, num_wordpieces].
        type_ids: `Optional[torch.LongTensor]`
            Shape: `[batch_size, num_wordpieces if max_length is None else num_segment_concat_wordpieces]`.
        segment_concat_mask: `Optional[torch.BoolTensor]`
            Shape: `[batch_size, num_segment_concat_wordpieces]`.

        # Returns

        `torch.Tensor`
            Shape: `[batch_size, num_wordpieces, embedding_size]`.

        """
        # Some of the huggingface transformers don't support type ids at all and crash when you supply
        # them. For others, you can supply a tensor of zeros, and if you don't, they act as if you did.
        # There is no practical difference to the caller, so here we pretend that one case is the same
        # as another case.
        if type_ids is not None:
            max_type_id = type_ids.max()
            if max_type_id == 0:
                type_ids = None
            else:
                if max_type_id >= self._number_of_token_type_embeddings():
                    raise ValueError("Found type ids too large for the chosen transformer model.")
                assert token_ids.shape == type_ids.shape

        fold_long_sequences = self._max_length is not None and token_ids.size(1) > self._max_length
        if fold_long_sequences:
            batch_size, num_segment_concat_wordpieces = token_ids.size()
            token_ids, segment_concat_mask, type_ids = self._fold_long_sequences(
                token_ids, segment_concat_mask, type_ids
            )

        transformer_mask = segment_concat_mask if self._max_length is not None else mask
        assert transformer_mask is not None
        # Shape: [batch_size, num_wordpieces, embedding_size],
        # or if self._max_length is not None:
        # [batch_size * num_segments, self._max_length, embedding_size]

        # We call this with kwargs because some of the huggingface models don't have the
        # token_type_ids parameter and fail even when it's given as None.
        # Also, as of transformers v2.5.1, they are taking FloatTensor masks.
        parameters = {"input_ids": token_ids, "attention_mask": transformer_mask.float()}
        if type_ids is not None:
            parameters["token_type_ids"] = type_ids
        if dep_att_mask is not None:
            parameters["dep_att_mask"] = dep_att_mask

        transformer_output = self.transformer_model(**parameters)
        if self._scalar_mix is not None:
            # The hidden states will also include the embedding layer, which we don't
            # include in the scalar mix. Hence the `[1:]` slicing.
            hidden_states = transformer_output.hidden_states[1:]
            embeddings = self._scalar_mix(hidden_states)
        else:
            embeddings = transformer_output.last_hidden_state

        if fold_long_sequences:
            embeddings = self._unfold_long_sequences(
                embeddings, segment_concat_mask, batch_size, num_segment_concat_wordpieces
            )

        return embeddings

    def _fold_long_sequences(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.LongTensor, torch.LongTensor, Optional[torch.LongTensor]]:
        """
        We fold 1D sequences (for each element in batch), returned by `PretrainedTransformerIndexer`
        that are in reality multiple segments concatenated together, to 2D tensors, e.g.

        [ [CLS] A B C [SEP] [CLS] D E [SEP] ]
        -> [ [ [CLS] A B C [SEP] ], [ [CLS] D E [SEP] [PAD] ] ]
        The [PAD] positions can be found in the returned `mask`.

        # Parameters

        token_ids: `torch.LongTensor`
            Shape: `[batch_size, num_segment_concat_wordpieces]`.
            num_segment_concat_wordpieces is num_wordpieces plus special tokens inserted in the
            middle, i.e. the length of: "[CLS] A B C [SEP] [CLS] D E F [SEP]" (see indexer logic).
        mask: `torch.BoolTensor`
            Shape: `[batch_size, num_segment_concat_wordpieces]`.
            The mask for the concatenated segments of wordpieces. The same as `segment_concat_mask`
            in `forward()`.
        type_ids: `Optional[torch.LongTensor]`
            Shape: [batch_size, num_segment_concat_wordpieces].

        # Returns:

        token_ids: `torch.LongTensor`
            Shape: [batch_size * num_segments, self._max_length].
        mask: `torch.BoolTensor`
            Shape: [batch_size * num_segments, self._max_length].
        """
        num_segment_concat_wordpieces = token_ids.size(1)
        num_segments = math.ceil(num_segment_concat_wordpieces / self._max_length)  # type: ignore
        padded_length = num_segments * self._max_length  # type: ignore
        length_to_pad = padded_length - num_segment_concat_wordpieces

        def fold(tensor):  # Shape: [batch_size, num_segment_concat_wordpieces]
            # Shape: [batch_size, num_segments * self._max_length]
            tensor = F.pad(tensor, [0, length_to_pad], value=0)
            # Shape: [batch_size * num_segments, self._max_length]
            return tensor.reshape(-1, self._max_length)

        return fold(token_ids), fold(mask), fold(type_ids) if type_ids is not None else None

    def _unfold_long_sequences(
        self,
        embeddings: torch.FloatTensor,
        mask: torch.BoolTensor,
        batch_size: int,
        num_segment_concat_wordpieces: int,
    ) -> torch.FloatTensor:
        """
        We take 2D segments of a long sequence and flatten them out to get the whole sequence
        representation while remove unnecessary special tokens.

        [ [ [CLS]_emb A_emb B_emb C_emb [SEP]_emb ], [ [CLS]_emb D_emb E_emb [SEP]_emb [PAD]_emb ] ]
        -> [ [CLS]_emb A_emb B_emb C_emb D_emb E_emb [SEP]_emb ]

        We truncate the start and end tokens for all segments, recombine the segments,
        and manually add back the start and end tokens.

        # Parameters

        embeddings: `torch.FloatTensor`
            Shape: [batch_size * num_segments, self._max_length, embedding_size].
        mask: `torch.BoolTensor`
            Shape: [batch_size * num_segments, self._max_length].
            The mask for the concatenated segments of wordpieces. The same as `segment_concat_mask`
            in `forward()`.
        batch_size: `int`
        num_segment_concat_wordpieces: `int`
            The length of the original "[ [CLS] A B C [SEP] [CLS] D E F [SEP] ]", i.e.
            the original `token_ids.size(1)`.

        # Returns:

        embeddings: `torch.FloatTensor`
            Shape: [batch_size, self._num_wordpieces, embedding_size].
        """

        def lengths_to_mask(lengths, max_len, device):
            return torch.arange(max_len, device=device).expand(
                lengths.size(0), max_len
            ) < lengths.unsqueeze(1)

        device = embeddings.device
        num_segments = int(embeddings.size(0) / batch_size)
        embedding_size = embeddings.size(2)

        # We want to remove all segment-level special tokens but maintain sequence-level ones
        num_wordpieces = num_segment_concat_wordpieces - (num_segments - 1) * self._num_added_tokens

        embeddings = embeddings.reshape(
            batch_size, num_segments * self._max_length, embedding_size  # type: ignore
        )
        mask = mask.reshape(batch_size, num_segments * self._max_length)  # type: ignore
        # We assume that all 1s in the mask precede all 0s, and add an assert for that.
        # Open an issue on GitHub if this breaks for you.
        # Shape: (batch_size,)
        seq_lengths = mask.sum(-1)
        if not (lengths_to_mask(seq_lengths, mask.size(1), device) == mask).all():
            raise ValueError(
                "Long sequence splitting only supports masks with all 1s preceding all 0s."
            )
        # Shape: (batch_size, self._num_added_end_tokens); this is a broadcast op
        end_token_indices = (
            seq_lengths.unsqueeze(-1) - torch.arange(self._num_added_end_tokens, device=device) - 1
        )

        # Shape: (batch_size, self._num_added_start_tokens, embedding_size)
        start_token_embeddings = embeddings[:, : self._num_added_start_tokens, :]
        # Shape: (batch_size, self._num_added_end_tokens, embedding_size)
        end_token_embeddings = batched_index_select(embeddings, end_token_indices)

        embeddings = embeddings.reshape(batch_size, num_segments, self._max_length, embedding_size)
        embeddings = embeddings[
            :, :, self._num_added_start_tokens : embeddings.size(2) - self._num_added_end_tokens, :
        ]  # truncate segment-level start/end tokens
        embeddings = embeddings.reshape(batch_size, -1, embedding_size)  # flatten

        # Now try to put end token embeddings back which is a little tricky.

        # The number of segment each sequence spans, excluding padding. Mimicking ceiling operation.
        # Shape: (batch_size,)
        num_effective_segments = (seq_lengths + self._max_length - 1) // self._max_length
        # The number of indices that end tokens should shift back.
        num_removed_non_end_tokens = (
            num_effective_segments * self._num_added_tokens - self._num_added_end_tokens
        )
        # Shape: (batch_size, self._num_added_end_tokens)
        end_token_indices -= num_removed_non_end_tokens.unsqueeze(-1)
        assert (end_token_indices >= self._num_added_start_tokens).all()
        # Add space for end embeddings
        embeddings = torch.cat([embeddings, torch.zeros_like(end_token_embeddings)], 1)
        # Add end token embeddings back
        embeddings.scatter_(
            1, end_token_indices.unsqueeze(-1).expand_as(end_token_embeddings), end_token_embeddings
        )

        # Now put back start tokens. We can do this before putting back end tokens, but then
        # we need to change `num_removed_non_end_tokens` a little.
        embeddings = torch.cat([start_token_embeddings, embeddings], 1)

        # Truncate to original length
        embeddings = embeddings[:, :num_wordpieces, :]
        return embeddings



@TokenEmbedder.register("pretrained_transformer_mismatched_with_dep_att_mask")
class PretrainedTransformerMismatchedEmbedderWithDepAttMask(TokenEmbedder):
    """
    Use this embedder to embed wordpieces given by `PretrainedTransformerMismatchedIndexer`
    and to get word-level representations.

    Registered as a `TokenEmbedder` with name "pretrained_transformer_mismatched".

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use. Should be the same as the corresponding
        `PretrainedTransformerMismatchedIndexer`.
    max_length : `int`, optional (default = `None`)
        If positive, folds input token IDs into multiple segments of this length, pass them
        through the transformer model independently, and concatenate the final representations.
        Should be set to the same value as the `max_length` option on the
        `PretrainedTransformerMismatchedIndexer`.
    sub_module: `str`, optional (default = `None`)
        The name of a submodule of the transformer to be used as the embedder. Some transformers naturally act
        as embedders such as BERT. However, other models consist of encoder and decoder, in which case we just
        want to use the encoder.
    train_parameters: `bool`, optional (default = `True`)
        If this is `True`, the transformer weights get updated during training.
    last_layer_only: `bool`, optional (default = `True`)
        When `True` (the default), only the final layer of the pretrained transformer is taken
        for the embeddings. But if set to `False`, a scalar mix of all of the layers
        is used.
    override_weights_file: `Optional[str]`, optional (default = `None`)
        If set, this specifies a file from which to load alternate weights that override the
        weights from huggingface. The file is expected to contain a PyTorch `state_dict`, created
        with `torch.save()`.
    override_weights_strip_prefix: `Optional[str]`, optional (default = `None`)
        If set, strip the given prefix from the state dict when loading it.
    load_weights: `bool`, optional (default = `True`)
        Whether to load the pretrained weights. If you're loading your model/predictor from an AllenNLP archive
        it usually makes sense to set this to `False` (via the `overrides` parameter)
        to avoid unnecessarily caching and loading the original pretrained weights,
        since the archive will already contain all of the weights needed.
    gradient_checkpointing: `bool`, optional (default = `None`)
        Enable or disable gradient checkpointing.
    tokenizer_kwargs: `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/tokenization_utils.py#L691)
        for `AutoTokenizer.from_pretrained`.
    transformer_kwargs: `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/modeling_utils.py#L253)
        for `AutoModel.from_pretrained`.
    sub_token_mode: `Optional[str]`, optional (default= `avg`)
        If `sub_token_mode` is set to `first`, return first sub-token representation as word-level representation
        If `sub_token_mode` is set to `avg`, return average of all the sub-tokens representation as word-level representation
        If `sub_token_mode` is not specified it defaults to `avg`
        If invalid `sub_token_mode` is provided, throw `ConfigurationError`

    """  # noqa: E501

    def __init__(
        self,
        model_name: str,
        max_length: int = None,
        sub_module: str = None,
        train_parameters: bool = True,
        last_layer_only: bool = True,
        override_weights_file: Optional[str] = None,
        override_weights_strip_prefix: Optional[str] = None,
        load_weights: bool = True,
        gradient_checkpointing: Optional[bool] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
        sub_token_mode: Optional[str] = "avg",
    ) -> None:
        super().__init__()
        # The matched version v.s. mismatched
        self._matched_embedder = PretrainedTransformerEmbedderWithDepAttMask(
            model_name,
            max_length=max_length,
            sub_module=sub_module,
            train_parameters=train_parameters,
            last_layer_only=last_layer_only,
            override_weights_file=override_weights_file,
            override_weights_strip_prefix=override_weights_strip_prefix,
            load_weights=load_weights,
            gradient_checkpointing=gradient_checkpointing,
            tokenizer_kwargs=tokenizer_kwargs,
            transformer_kwargs=transformer_kwargs,
        )
        self.sub_token_mode = sub_token_mode

    def get_output_dim(self):
        return self._matched_embedder.get_output_dim()

    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
        dep_att_mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters

        token_ids: `torch.LongTensor`
            Shape: [batch_size, num_wordpieces] (for exception see `PretrainedTransformerEmbedder`).
        mask: `torch.BoolTensor`
            Shape: [batch_size, num_orig_tokens].
        offsets: `torch.LongTensor`
            Shape: [batch_size, num_orig_tokens, 2].
            Maps indices for the original tokens, i.e. those given as input to the indexer,
            to a span in token_ids. `token_ids[i][offsets[i][j][0]:offsets[i][j][1] + 1]`
            corresponds to the original j-th token from the i-th batch.
        wordpiece_mask: `torch.BoolTensor`
            Shape: [batch_size, num_wordpieces].
        type_ids: `Optional[torch.LongTensor]`
            Shape: [batch_size, num_wordpieces].
        segment_concat_mask: `Optional[torch.BoolTensor]`
            See `PretrainedTransformerEmbedder`.

        # Returns

        `torch.Tensor`
            Shape: [batch_size, num_orig_tokens, embedding_size].
        """
        # Shape: [batch_size, num_wordpieces, embedding_size].
        embeddings = self._matched_embedder(
            token_ids, wordpiece_mask, type_ids=type_ids, segment_concat_mask=segment_concat_mask, dep_att_mask=dep_att_mask
        )

        # span_embeddings: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        # span_mask: (batch_size, num_orig_tokens, max_span_length)
        span_embeddings, span_mask = util.batched_span_select(embeddings.contiguous(), offsets)

        span_mask = span_mask.unsqueeze(-1)

        # Shape: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        span_embeddings *= span_mask  # zero out paddings

        # If "sub_token_mode" is set to "first", return the first sub-token embedding
        if self.sub_token_mode == "first":
            # Select first sub-token embeddings from span embeddings
            # Shape: (batch_size, num_orig_tokens, embedding_size)
            orig_embeddings = span_embeddings[:, :, 0, :]

        # If "sub_token_mode" is set to "avg", return the average of embeddings of all sub-tokens of a word
        elif self.sub_token_mode == "avg":
            # Sum over embeddings of all sub-tokens of a word
            # Shape: (batch_size, num_orig_tokens, embedding_size)
            span_embeddings_sum = span_embeddings.sum(2)

            # Shape (batch_size, num_orig_tokens)
            span_embeddings_len = span_mask.sum(2)

            # Find the average of sub-tokens embeddings by dividing `span_embedding_sum` by `span_embedding_len`
            # Shape: (batch_size, num_orig_tokens, embedding_size)
            orig_embeddings = span_embeddings_sum / torch.clamp_min(span_embeddings_len, 1)

            # All the places where the span length is zero, write in zeros.
            orig_embeddings[(span_embeddings_len == 0).expand(orig_embeddings.shape)] = 0

        # If invalid "sub_token_mode" is provided, throw error
        else:
            raise ConfigurationError(f"Do not recognise 'sub_token_mode' {self.sub_token_mode}")

        return orig_embeddings
