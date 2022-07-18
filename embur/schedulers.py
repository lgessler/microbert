import itertools
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Union

from allennlp.data.data_loaders.multitask_scheduler import MultiTaskScheduler, _chunked_iterator
from allennlp.data.instance import Instance


@MultiTaskScheduler.register("homogeneous_weight_proportional")
class HomogeneousWeightProportionalScheduler(MultiTaskScheduler):
    """
    This is like HomogeneousRoundRobinScheduler, but batches are scheduled so that all of them will run out at
    approximately the same time. E.g., if weights are {"a": 8, "b": 1}, then the schedule will be 8 "a" batches
    followed by 1 "b" batch, in a cycle.

    # Parameters
    batch_size: `Union[int, Dict[str, int]]`
        Determines how many instances to group together in each dataset.  If this is an `int`, the
        same value is used for all datasets; otherwise, the keys must correspond to the dataset
        names used elsewhere in the multi-task code.
    weights: `Dict[str, int]`
        Should be the same weights dict that is sent to the sampler.
    """

    def __init__(self, batch_size: Union[int, Dict[str, int]], weights: Dict[str, int], drop_last: bool = False, pattern: List[str] = None):
        self.batch_size: Mapping[str, int]
        if isinstance(batch_size, int):
            self.batch_size = defaultdict(lambda: batch_size)  # type: ignore
        else:
            self.batch_size = batch_size
        self.drop_last = drop_last
        if not all(isinstance(w, int) for w in weights.values()):
            raise TypeError(f"All weights must be integral. Got: {[(w, type(w)) for w in weights.values()]}")
        self.weights = weights

        if pattern is not None:
            assert isinstance(pattern, Iterable), "Pattern must be a list"
            assert all(isinstance(s, str) for s in pattern), "Pattern must be made of strings"
            assert all(s in weights.keys() for s in pattern), "Pattern must be names of tasks"
            assert len(pattern) == sum(weights.values()), "Pattern length must match weight sum"
            self.pattern = itertools.cycle(iter(pattern))
        else:
            pattern = []
            for dataset_id, weight in weights.items():
                pattern += [dataset_id] * weight
            self.pattern = itertools.cycle(iter(pattern))


    def batch_instances(
        self, epoch_instances: Dict[str, Iterable[Instance]]
    ) -> Iterable[List[Instance]]:
        chunked_iterators = {
            dataset: _chunked_iterator(iterator, self.batch_size[dataset], self.drop_last)
            for dataset, iterator in epoch_instances.items()
        }

        def weighted_iteration(iterables, weights):
            done = {d: False for d in weights.keys()}
            counts = {d: 0 for d in weights.keys()}
            batches = {d: 0 for d in weights.keys()}

            for i, dataset_id in enumerate(self.pattern):
                if all(done.values()):
                    break
                if done[dataset_id]:
                    continue
                try:
                    print("\n",dataset_id)
                    val = next(iterables[dataset_id])
                    counts[dataset_id] += len(val)
                    batches[dataset_id] += 1
                    yield val
                except StopIteration:
                    done[dataset_id] = True
            print("counts", counts)
            print("batchs", batches)

        return weighted_iteration(chunked_iterators, self.weights)

    def count_batches(self, dataset_counts: Dict[str, int]) -> int:
        result = 0
        for dataset, count in dataset_counts.items():
            batch_size = self.batch_size[dataset]
            result += count // batch_size
            if not self.drop_last and count % batch_size != 0:
                result += 1
        return result
