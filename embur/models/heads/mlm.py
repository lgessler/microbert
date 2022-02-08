from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.heads.head import Head
from allennlp.nn import Activation
from allennlp.training.metrics import CategoricalAccuracy, Perplexity
from overrides import overrides


@Head.register("mlm")
class MlmHead(Head):
    def __init__(self, vocab: Vocabulary, embedding_dim: int):
        super().__init__(vocab)
        self.vocab_size = vocab.get_vocab_size("tokens")
        self.pad_token_index = vocab.get_token_index("[PAD]")

        self.prediction_head = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
            Activation.by_name('gelu')(),
            torch.nn.LayerNorm(embedding_dim, 1e-12),
            torch.nn.Linear(embedding_dim, self.vocab_size)
        )

        self._accuracy = CategoricalAccuracy()
        self._perplexity = Perplexity()

    @overrides
    def forward(
        self,  # type: ignore
        encoded_masked_text: torch.Tensor,
        masked_text_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        
        prediction_scores = self.prediction_head(encoded_masked_text)

        probs = F.softmax(prediction_scores, dim=-1)
        top_probs, top_indices = probs.topk(k=5, dim=-1)

        output_dict = {"prediction_probs": top_probs, "top_indices": top_indices}

        if masked_text_labels is not None:
            # Gather all masked tokens, i.e. all tokens that aren't -100 (= not masked) or padding
            not_modified_mask = (masked_text_labels == -100)
            padding_mask = (masked_text_labels == self.pad_token_index)
            loss_mask = (~(padding_mask | not_modified_mask))

            mask_predictions = prediction_scores[loss_mask]
            mask_labels = masked_text_labels[loss_mask]

            loss = F.cross_entropy(mask_predictions, mask_labels)
            self._perplexity(loss)
            output_dict["loss"] = loss
            output_dict["masked_text_labels"] = masked_text_labels

        return output_dict

    def get_metrics(self, reset: bool = False):
        return {"perplexity": self._perplexity.get_metric(reset=reset)}

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        top_words = []
        for instance_indices in output_dict["top_indices"]:
            top_words.append(
                [
                    [
                        self.vocab.get_token_from_index(index.item(), namespace=self._target_namespace)
                        for index in mask_positions
                    ]
                    for mask_positions in instance_indices
                ]
            )
        output_dict["words"] = top_words
        tokens = []
        for instance_tokens in output_dict["token_ids"]:
            tokens.append(
                [
                    self.vocab.get_token_from_index(token_id.item(), namespace=self._target_namespace)
                    for token_id in instance_tokens
                ]
            )
        output_dict["tokens"] = tokens

        return output_dict
