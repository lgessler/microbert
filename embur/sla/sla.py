from collections import defaultdict
from typing import Any, Dict, List

import torch


def get_head_map(head: torch.LongTensor, head_length: torch.LongTensor):
    # split the batched head tensor into one tensor per input sequence, with padding removed
    padless_head = [head[i, :hl] for i, hl in enumerate(head_length)]
    # Map from IDs to heads. Note that this is all 1-indexed, with 0 being the dummy ROOT node.
    head_map = [{0: None, **{i + 1: h.item() for i, h in enumerate(heads)}} for heads in padless_head]
    return head_map


def get_adjacency_map(head_map) -> Dict[int, Dict[int, bool]]:
    adj_map = defaultdict(lambda: defaultdict(bool))
    for k, v in head_map.items():
        if k is not None and v is not None:
            adj_map[k][v] = True
            adj_map[v][k] = True
    return adj_map


def ids_in_range(head_map, begin: int, max_distance: int):
    head_map = {k: v for k, v in head_map.items() if k != 0 and v != 0}
    adj_map = get_adjacency_map(head_map)
    if begin not in adj_map:
        return set()

    visited = set()

    def inner(current: int, distance: int):
        if distance > max_distance:
            return
        visited.add(current)

        neighbors = {k for k, v in adj_map[current].items() if v and k not in visited}
        for v in neighbors:
            inner(v, distance + 1)

    inner(begin, 0)
    return visited


def generate_sla_mask(head: torch.LongTensor, head_length: torch.LongTensor, max_distance) -> torch.LongTensor:
    """
    Given head of [batch_size, seq_len] s.t. [CLS] and [SEP] are NOT present
    produce [batch_size, seq_len + 2, seq_len + 2] (we expand to account for [CLS] and [SEP])
    such that [b, i, j] is 1 if token i may attend to token j, and 0 otherwise. Note that
    whenever j corresponds to [CLS] or [SEP] or when i = j, [b, i, j] is always 1.
    """
    batch_size, heads = head.shape
    device = head.device
    head_maps = get_head_map(head, head_length)
    att_mask: torch.LongTensor = torch.zeros((batch_size,) + ((heads + 2,) * 2), dtype=torch.long, device=device)
    for b, head_map in enumerate(head_maps):
        for i in range(1, head.shape[1] - 1):
            in_range = ids_in_range(head_map, i, max_distance)
            if i != 1:
                in_range |= ids_in_range(head_map, i - 1, max_distance)
            if i != head.shape[1] - 2:
                in_range |= ids_in_range(head_map, i - 1, max_distance)
            for j in in_range:
                att_mask[b, i, j] = 1
    # Ensure [CLS] is not masked
    att_mask[:, :, 0] = 1
    non_packed_seq_lens = head_length + 2
    for i, l in enumerate(non_packed_seq_lens):
        # Allow [CLS] to attend to all tokens. (This is a point of departure vs. the original impl.)
        att_mask[i, 0, :l] = 1
        # Ensure [SEP] is not masked
        att_mask[i, :, l - 1] = 1
        # Mask out rows beyond sequence length
        att_mask[i, l - 1 :] = 0
    transformed_mask = transform_sla_mask(att_mask)
    return transformed_mask


LARGE_NEGATIVE_FLOAT = -1.e6


def transform_sla_mask(mask):
    """Given a mask where 1 means attend and 0 means don't attend, change 1 to 0 and 0 to -inf"""
    mask = mask.float()
    mask[mask == 0.0] = LARGE_NEGATIVE_FLOAT
    mask[mask == 1.0] = 0.0
    return mask


def generate_sla_mask_list(head: List[int], length: int, max_distance: int = 5):
    head_t = torch.tensor(head).unsqueeze(0)
    mask = generate_sla_mask(head_t, torch.LongTensor([length]), max_distance)
    return mask.squeeze(0)
