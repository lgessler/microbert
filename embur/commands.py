import argparse
import json
import logging
from typing import Any, Dict

from allennlp.common import logging as common_logging
from allennlp.common.util import prepare_environment
from allennlp.data import DataLoader
from allennlp.models import load_archive
from allennlp.training.util import evaluate

logger = logging.getLogger(__name__)


# This is an older version of the function that'll tolerate dicts for input files
# function is from (from https://github.com/allenai/allennlp/commits/main/allennlp/commands/evaluate.py)
def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    common_logging.FILE_FRIENDLY_LOGGING = args.file_friendly_logging

    # Disable some of the more verbose logging statements
    logging.getLogger("allennlp.common.params").disabled = True
    logging.getLogger("allennlp.nn.initializers").disabled = True
    logging.getLogger("allennlp.modules.token_embedders.embedding").setLevel(logging.INFO)

    # Load from archive
    archive = load_archive(
        args.archive_file,
        weights_file=args.weights_file,
        cuda_device=args.cuda_device,
        overrides=args.overrides,
    )
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    # Load the evaluation data

    dataset_reader = archive.validation_dataset_reader

    evaluation_data_path = args.input_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)

    data_loader_params = config.pop("validation_data_loader", None)
    if data_loader_params is None:
        data_loader_params = config.pop("data_loader")
    if args.batch_size:
        data_loader_params["batch_size"] = args.batch_size
    data_loader = DataLoader.from_params(
        params=data_loader_params, reader=dataset_reader, data_path=evaluation_data_path
    )

    embedding_sources = (
        json.loads(args.embedding_sources_mapping) if args.embedding_sources_mapping else {}
    )

    if args.extend_vocab:
        logger.info("Vocabulary is being extended with test instances.")
        model.vocab.extend_from_instances(instances=data_loader.iter_instances())
        model.extend_embedder_vocab(embedding_sources)

    data_loader.index_with(model.vocab)

    metrics = evaluate(
        model,
        data_loader,
        args.cuda_device,
        args.batch_weight_key,
        output_file=args.output_file,
        predictions_output_file=args.predictions_output_file,
    )

    logger.info("Finished evaluating.")

    return metrics
