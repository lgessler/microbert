from typing import Dict, Optional

import torch
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.backbones.backbone import Backbone
from allennlp.nn import util
from overrides import overrides


@Backbone.register("contextualized_embedding")
class ContextualizedEmbedding(Backbone):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder
    ) -> None:
        super().__init__()
        self._vocab = vocab
        self._namespace = "tokens"
        self.embedder = embedder
        self.encoder = encoder

    def forward(
        self,
        text: TextFieldTensors,
        masked_text: Optional[TextFieldTensors] = None,
        masked_positions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:  # type: ignore
        if len(text) != 1:
            raise ValueError("PretrainedTransformerBackbone is only compatible with using a single TokenIndexer")
        mask = util.get_text_field_mask(text)
        embedded_text = self.embedder(text)
        encoded_text = self.encoder(embedded_text, mask)

        outputs = {
            "encoded_text": encoded_text,
            "encoded_text_mask": mask,
            "token_ids": util.get_token_ids_from_text_field_tensors(text),
        }

        if masked_text is not None and masked_positions is not None:
            masked_text_mask = util.get_text_field_mask(masked_text)
            encoded_masked_text = self.embedder(text)
            outputs["masked_positions"] = masked_positions,
            outputs["encoded_masked_text"] = encoded_masked_text
            outputs["encoded_masked_text_mask"] = masked_text_mask
        return outputs

    @overrides
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
