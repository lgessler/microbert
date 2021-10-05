from typing import Dict, Optional, Any

from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, ConditionalRandomField, TimeDistributed, Seq2SeqEncoder
from allennlp.training.metrics import SpanBasedF1Measure, CategoricalAccuracy
import torch

from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn import util
from torch.nn import Dropout


@Model.register("entity_crf")
class EntityCrf(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder = None,
        dropout: float = 0.5,
        label_namespace: str = "entity_tags",
    ) -> None:
        super().__init__(vocab)
        self.vocab = vocab
        self.embedder = embedder
        self.encoder = encoder
        self.dropout = Dropout(dropout)

        self.label_namespace = label_namespace
        self.labels = vocab.get_index_to_token_vocabulary(label_namespace)
        num_labels = vocab.get_vocab_size(label_namespace)

        self.label_projection_layer = TimeDistributed(
            torch.nn.Linear(embedder.get_output_dim() if encoder is None else encoder.get_output_dim(), num_labels)
        )
        self.crf = ConditionalRandomField(num_labels, include_start_end_transitions=True)

        self.metrics = {
            "span_f1": SpanBasedF1Measure(vocab, tag_namespace=label_namespace, label_encoding="BIO"),
            "accuracy": CategoricalAccuracy(),
        }

    def forward(self, text: TextFieldTensors, entity_tags: torch.Tensor, metadata: Dict[str, Any]) -> Dict[str, torch.Tensor]:  # type: ignore
        mask = util.get_text_field_mask(text)
        encoded_text = self.embedder(text)

        if self.encoder:
            encoded_text = self.encoder(encoded_text, mask)

        encoded_text = self.dropout(encoded_text)

        logits = self.label_projection_layer(encoded_text)
        pred_entities = [
            best_label_seq for best_label_seq, viterbi_score in self.crf.viterbi_tags(logits, mask, top_k=None)
        ]

        output = {"pred_entities": pred_entities}
        if entity_tags is not None:
            output["gold_entities"] = entity_tags
            class_probabilities = logits * 0.0
            for i, instance_labels in enumerate(pred_entities):
                for j, label_id in enumerate(instance_labels):
                    class_probabilities[i, j, label_id] = 1
            log_likelihood = self.crf(logits, entity_tags, mask)
            output["loss"] = -log_likelihood
            for metric in self.metrics.values():
                metric(class_probabilities, entity_tags, mask)

        return output

    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        def decode_entities(entities):
            return [self.vocab.get_token_from_index(int(label), self.label_namespace) for label in entities]

        output_dict["pred_entities"] = [decode_entities(t) for t in output_dict["pred_entities"]]
        output_dict["gold_entities"] = [decode_entities(t) for t in output_dict["gold_entities"]]
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"tag_accuracy": self.metrics["accuracy"].get_metric(reset)}
        f1_metrics = {
            "span_" + k.replace("-overall", "").replace("-measure", ""): v
            for k, v in self.metrics["span_f1"].get_metric(reset=reset).items()
            if "overall" in k
        }
        metrics.update(f1_metrics)
        return metrics  # type: ignore
