from collections import defaultdict
from typing import Any, Dict, List

import torch


def get_head_map(head: torch.LongTensor):
    # Count number of tokens in the tree: find nonzero heads to account for 0-padding, and add one
    # to account for the fact that 0:root is labeled with a head = 0.
    token_counts = (head != 0).sum(1) + 1
    # split the batched head tensor into one tensor per input sequence, with padding removed
    padless_head = [head[i, : token_counts[i]] for i, x in enumerate(token_counts)]
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


def generate_sla_mask(head: torch.LongTensor, max_distance: int = 5) -> torch.LongTensor:
    """
    Given head of [batch_size, seq_len] s.t.:
     - [CLS] and [SEP] are NOT present
     - the root of the sequence represented in head is equal to 0
    produce [batch_size, seq_len + 2, seq_len + 2] (we expand to account for [CLS] and [SEP])
    such that [b, i, j] is 1 if token i may attend to token j, and 0 otherwise. Note that
    whenever j corresponds to [CLS] or [SEP] or when i = j, [b, i, j] is always 1.
    """
    batch_size, max_seq_len = head.shape
    device = head.device
    zero_col = torch.zeros((batch_size, 1), device=device, dtype=torch.long)
    head = torch.concat((zero_col, head, zero_col), dim=1)
    head_maps = get_head_map(head[:, 1:])
    att_mask: torch.LongTensor = torch.zeros(head.shape + (head.shape[1],), dtype=torch.long, device=head.device)
    for b, head_map in enumerate(head_maps):
        for i in range(1, head.shape[1] - 1):
            in_range = ids_in_range(head_map, i, max_distance)
            for j in in_range:
                att_mask[b, i, j] = 1
    # Ensure [CLS] is not masked
    att_mask[:, :, 0] = 1
    non_packed_seq_lens = (head != 0).sum(1) + 2
    for i, l in enumerate(non_packed_seq_lens):
        # Allow [CLS] to attend to all tokens. (This is a point of departure vs. the original impl.)
        att_mask[i, 0, :l] = 1
        # Ensure [SEP] is not masked
        att_mask[i, :, l] = 1
        # Mask out rows beyond sequence length
        att_mask[i, l + 1 :] = 0
    return att_mask


def generate_sla_mask_list(head: List[int], max_distance: int = 5):
    head_t = torch.tensor(head).unsqueeze(0)
    mask = generate_sla_mask(head_t, max_distance)
    return mask.squeeze(0)
