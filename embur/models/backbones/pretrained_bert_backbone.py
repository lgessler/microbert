from typing import Any, Dict, Optional

import torch
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.backbones.backbone import Backbone
from allennlp.modules.seq2seq_encoders import PytorchTransformer
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder
from allennlp.nn import util
from transformers import BertTokenizer, DataCollatorForWholeWordMask
from transformers.models.bert.modeling_bert import BertConfig, BertModel


@Backbone.register("pretrained_bert")
class PretrainedBertBackbone(Backbone):
    def __init__(self, vocab: Vocabulary, bert_model: str) -> None:
        super().__init__()
        # TODO:
        # - Need to apply corrections in pretrained_transformer_mismatched_embedder
        tokenizer = BertTokenizer.from_pretrained(bert_model)
        vocab.add_transformer_vocab(tokenizer, "tokens")
        # "tokens" is padded by default--undo that
        try:
            del vocab._token_to_index["tokens"]["@@PADDING@@"]
        except KeyError:
            pass
        try:
            del vocab._token_to_index["tokens"]["@@UNKNOWN@@"]
        except KeyError:
            pass
        assert len(vocab._token_to_index["tokens"]) == len(vocab._index_to_token["tokens"])

        self._vocab = vocab
        self._namespace = "tokens"
        self.bert = BertModel.from_pretrained(bert_model)
        self.masking_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    def _embed(self, text: TextFieldTensors) -> Dict[str, torch.Tensor]:
        """
        This implementation is borrowed from `PretrainedTransformerMismatchedEmbedder` and uses
        average pooling to yield a de-wordpieced embedding for each original token.
        Returns both wordpiece embeddings+mask as well as original token embeddings+mask
        """
        output = self.bert(
            input_ids=text["tokens"]["token_ids"],
            attention_mask=text["tokens"]["wordpiece_mask"],
            token_type_ids=text["tokens"]["type_ids"],
        )
        wordpiece_embeddings = output.last_hidden_state
        offsets = text["tokens"]["offsets"]

        # Assemble wordpiece embeddings into embeddings for each word using average pooling
        span_embeddings, span_mask = util.batched_span_select(wordpiece_embeddings.contiguous(), offsets)  # type: ignore
        span_mask = span_mask.unsqueeze(-1)
        # Shape: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        span_embeddings *= span_mask  # zero out paddings
        # return the average of embeddings of all sub-tokens of a word
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

        return {
            "wordpiece_mask": text["tokens"]["wordpiece_mask"],
            "wordpiece_embeddings": wordpiece_embeddings,
            "orig_mask": text["tokens"]["mask"],
            "orig_embeddings": orig_embeddings,
        }

    def forward(self, text: TextFieldTensors) -> Dict[str, torch.Tensor]:  # type: ignore
        bert_output = self._embed(text)

        outputs = {
            "encoded_text": bert_output["orig_embeddings"],
            "encoded_text_mask": bert_output["orig_mask"],
            "wordpiece_encoded_text": bert_output["wordpiece_embeddings"],
            "wordpiece_encoded_text_mask": bert_output["wordpiece_mask"],
            "token_ids": util.get_token_ids_from_text_field_tensors(text),
        }

        self._extend_with_masked_text(outputs, text)
        return outputs

    def _extend_with_masked_text(self, outputs: Dict[str, Any], text: TextFieldTensors) -> None:
        input_ids = text["tokens"]["token_ids"]

        # get the binary mask that'll tell us which parts to mask--this is random and dynamically done
        wwms = []
        for i in range(input_ids.shape[0]):
            tokens = [self._vocab.get_token_from_index(i.item()) for i in input_ids[i]]
            wwm = torch.tensor(self.masking_collator._whole_word_mask(tokens)).unsqueeze(0)
            wwms.append(wwm)
        wwms = torch.cat(wwms, dim=0)

        if hasattr(self.masking_collator, "torch_mask_tokens"):
            masked_ids, labels = self.masking_collator.torch_mask_tokens(input_ids.to("cpu"), wwms.to("cpu"))
        else:
            masked_ids, labels = self.masking_collator.mask_tokens(input_ids.to("cpu"), wwms.to("cpu"))
        masked_ids = masked_ids.to(input_ids.device)
        labels = labels.to(input_ids.device)
        bert_output = self.bert(
            input_ids=masked_ids,
            attention_mask=text["tokens"]["wordpiece_mask"],
            token_type_ids=text["tokens"]["type_ids"],
        )
        outputs["encoded_masked_text"] = bert_output.last_hidden_state
        outputs["masked_text_labels"] = labels

    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tokens = []
        for instance_tokens in output_dict["token_ids"]:
            tokens.append(
                [
                    self._vocab.get_token_from_index(token_id.item(), namespace=self._namespace)
                    for token_id in instance_tokens
                ]
            )
        output_dict["tokens"] = tokens
        del output_dict["token_ids"]
        del output_dict["encoded_text"]
        del output_dict["encoded_text_mask"]
        del output_dict["encoded_masked_text"]
        return output_dict
