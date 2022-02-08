import itertools
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Union

import more_itertools
from allennlp.data import Instance
from allennlp.data.data_loaders.multitask_scheduler import MultiTaskScheduler, _chunked_iterator


@MultiTaskScheduler.register("homogeneous_repeated_roundrobin")
class HomogeneousRepeatedRoundRobinScheduler(MultiTaskScheduler):
    """
    Orders instances in a round-robin fashion, but grouped into batches composed entirely of
    instances from one dataset.  We'll return one batch from one dataset, then another batch from a
    different dataset, etc.  This is currently necessary in AllenNLP if your instances have
    different fields for different datasets, as we can't currently combine instances with different
    fields.
    When one dataset runs out, we continue iterating round-robin through the rest.
    If you want more fine-grained control over which datasets can be combined, it should be
    relatively straightforward to write your own scheduler, following this logic, which allows some
    datasets to be combined and others not.
    Registered as a `MultiTaskScheduler` with name "homogeneous_roundrobin".
    # Parameters
    batch_size: `Union[int, Dict[str, int]]`
        Determines how many instances to group together in each dataset.  If this is an `int`, the
        same value is used for all datasets; otherwise, the keys must correspond to the dataset
        names used elsewhere in the multi-task code.
    """

    def __init__(self, batch_size: Union[int, Dict[str, int]], drop_last: bool = False):
        self.batch_size: Mapping[str, int]
        if isinstance(batch_size, int):
            self.batch_size = defaultdict(lambda: batch_size)  # type: ignore
        else:
            self.batch_size = batch_size
        self.drop_last = drop_last

    def batch_instances(
        self, epoch_instances: Dict[str, Iterable[Instance]]
    ) -> Iterable[List[Instance]]:
        forced_epoch_instances = {dataset: list(iterator) for dataset, iterator in epoch_instances.items()}
        max_length = max(len(iterator) for iterator in forced_epoch_instances.values())
        even_length_epoch_instances = {}
        for dataset, instances in forced_epoch_instances.items():
            even_length_epoch_instances[dataset] = []
            cycled_instances = itertools.cycle(instances)
            for i in range(max_length):
                even_length_epoch_instances[dataset].append(next(cycled_instances))

        even_length_epoch_instances = {dataset: iter(instances) for dataset, instances
                                       in even_length_epoch_instances.items()}
        chunked_iterators = [
            _chunked_iterator(iterator, self.batch_size[dataset], self.drop_last)
            for dataset, iterator in even_length_epoch_instances.items()
        ]
        return more_itertools.roundrobin(*chunked_iterators)

    def count_batches(self, dataset_counts: Dict[str, int]) -> int:
        result = 0
        for dataset, count in dataset_counts.items():
            batch_size = self.batch_size[dataset]
            result += count // batch_size
            if not self.drop_last and count % batch_size != 0:
                result += 1
        return result