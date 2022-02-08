from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from allennlp.data import TextFieldTensors
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.heads.head import Head
from allennlp.modules import ConditionalRandomField, TimeDistributed
from allennlp.modules.seq2seq_encoders import GruSeq2SeqEncoder
from allennlp.nn.util import sequence_cross_entropy_with_logits
from overrides import overrides


@Head.register("xpos")
class XposHead(Head):
    def __init__(self, vocab: Vocabulary, embedding_dim: int, use_crf: bool = False, label_namespace: str = "xpos_tags"):
        super().__init__(vocab)
        self.label_namespace = label_namespace
        self.labels = vocab.get_index_to_token_vocabulary(label_namespace)
        num_labels = vocab.get_vocab_size(label_namespace)

        if use_crf:
            self.crf = ConditionalRandomField(num_labels, include_start_end_transitions=True)
            self.label_projection_layer = TimeDistributed(torch.nn.Linear(embedding_dim, num_labels))
            self.decoder = None
        else:
            self.crf = None
            self.decoder = GruSeq2SeqEncoder(
                input_size=embedding_dim,
                hidden_size=embedding_dim,
                num_layers=1,
                bidirectional=True
            )
            self.label_projection_layer = TimeDistributed(torch.nn.Linear(self.decoder.get_output_dim(), num_labels))

        from allennlp.training.metrics import CategoricalAccuracy

        self.metrics = {"accuracy": CategoricalAccuracy()}

    @overrides
    def forward(
        self,  # type: ignore
        encoded_text: torch.Tensor,
        encoded_text_mask: torch.Tensor,
        xpos_tags: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        if self.crf is not None:
            logits = self.label_projection_layer(encoded_text)
            pred_xpos = [
                best_label_seq
                for best_label_seq, viterbi_score in self.crf.viterbi_tags(logits, encoded_text_mask, top_k=None)
            ]
        else:
            logits = self.label_projection_layer(self.decoder(encoded_text, encoded_text_mask))
            pred_xpos = logits.argmax(-1)
        output = {"logits": logits, "pred_xpos": pred_xpos}

        if xpos_tags is not None:
            output["gold_xpos"] = xpos_tags

            if self.crf is not None:
                class_probabilities = logits * 0.0
                for i, instance_labels in enumerate(pred_xpos):
                    for j, id in enumerate(instance_labels):
                        class_probabilities[i, j, id] = 1

                log_likelihood = self.crf(logits, xpos_tags, encoded_text_mask)
                loss = -log_likelihood
                for metric in self.metrics.values():
                    metric(class_probabilities, xpos_tags, encoded_text_mask)
            else:
                loss = sequence_cross_entropy_with_logits(logits, xpos_tags, encoded_text_mask)
                for metric in self.metrics.values():
                    metric(logits, xpos_tags, encoded_text_mask)
            output["loss"] = loss


        return output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"tag_accuracy": self.metrics["accuracy"].get_metric(reset)}
        return metrics

    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        def decode_labels(labels):
            return [self.vocab.get_token_from_index(int(label), self.label_namespace) for label in labels]

        output_dict["pred_xpos"] = [decode_labels(t) for t in output_dict["pred_xpos"]]
        output_dict["gold_xpos"] = [decode_labels(t) for t in output_dict["gold_xpos"]]

        return output_dict
