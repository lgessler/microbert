from typing import Tuple

import torch
from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer
from allennlp.nn.util import get_token_ids_from_text_field_tensors
from transformers import BertTokenizer, DataCollatorForWholeWordMask

tokenizer = BertTokenizer.from_pretrained('./bert_out')
vocab = Vocabulary(non_padded_namespaces=["tokens"])
vocab.add_transformer_vocab(tokenizer, "tokens")

vocab.get_token_index("[PAD]", "tokens")


idx = PretrainedTransformerMismatchedIndexer("./bert_out", namespace="tokens")
def prepare_instance(s):
    tokens = [Token(t) for t in s.split(" ")]
    indexed = idx.tokens_to_indices(tokens, vocab)
    print([vocab.get_token_from_index(i) for i in indexed['token_ids']])
    return Instance({"tokens": TextField(tokens, {"tokens": idx})})

instances = [prepare_instance("ϩⲙⲡⲣⲁⲛ ⲙⲡⲛⲟⲩⲧⲉ ⲛϣⲟⲣⲡ ⲁⲛⲟⲕ"), prepare_instance("ϩⲙⲡⲣⲁⲛ ⲙⲡⲛⲟⲩⲧⲉ ⲛϣⲟⲣⲡ ⲁⲛⲟⲕ")]
for i in instances:
    i["tokens"].index(vocab)

tensors = [i.as_tensor_dict() for i in instances]

collator = DataCollatorForWholeWordMask(tokenizer=tokenizer)
ids = torch.cat([tensors[0]['tokens']['tokens']['token_ids'].unsqueeze(0),
                 tensors[1]['tokens']['tokens']['token_ids'].unsqueeze(0)], dim=0)
ids.shape
wwm = collator._whole_word_mask([[vocab.get_token_from_index(i.item()) for i in wp_ids] for wp_ids in ids])

wwms = []
for i in range(ids.shape[0]):
    tokens = [vocab.get_token_from_index(i.item()) for i in ids[i]]
    wwm = torch.tensor(collator._whole_word_mask(tokens)).unsqueeze(0)
    wwms.append(wwm)
wwms = torch.cat(wwms, dim=0)

wwm = torch.tensor(wwm).unsqueeze(0)
wwm
masked_ids, labels = collator.mask_tokens(ids, wwm)
masked_ids
labels
print([vocab.get_token_from_index(i.item()) for i in out[0][0]])

tensors[0]


import torch

labels = torch.tensor([[-100, 1, -100, -100], [-100, -100, 2, 0]])
not_modified_mask = (labels == -100)
padding_mask = (labels == 0)
padding_mask
not_modified_mask
mask = (~(padding_mask | not_modified_mask))
mask
labels[mask]


# see https://github.com/huggingface/transformers/blob/master/src/transformers/data/data_collator.py#L401
