import json
import os
import shutil
import subprocess
import logging

from embur.language_configs import get_pretrain_config, get_eval_config


def get_git_revision_hash(self):
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


TASKS = ("mlm", "xpos", "parser")
TOKENIZATION_TYPES = ("bpe", "wordpiece")

logger = logging.getLogger(__name__)


class Config:
    def __init__(self, *args, **kwargs):
        self.language = kwargs.pop("language")
        self.bert_model_name = kwargs.pop("bert_model_name")
        self.tasks = kwargs.pop("task")
        self.tokenization_type = kwargs.pop("tokenization_type")
        self.num_layers = kwargs.pop("num_layers")
        self.num_attention_heads = kwargs.pop("num_attention_heads")
        self.embedding_dim = kwargs.pop("embedding_dim")

        self.finetune = kwargs.pop("finetune")
        self.debug = kwargs.pop("debug", False)

        self.pretrain_language_config = get_pretrain_config(self.language, self.bert_dir, self.tasks)
        self.pretrain_jsonnet = kwargs.pop("training_config")
        self.parser_eval_language_config = get_eval_config(self.language, self.bert_model_name)
        self.parser_eval_jsonnet = kwargs.pop("parser_eval_config")

        if self.debug:
            os.environ["TOY_DATA"] = "true"

    def set_tasks(self, tasks):
        self.tasks = tasks
        self.pretrain_language_config = get_pretrain_config(self.language, self.bert_dir, self.tasks)

    @property
    def bert_dir(self):
        return (
            f"berts/{self.language}/"
            + f"{'-'.join(self.tasks)}"
            + (f"_layers-{self.num_layers}" if self.num_layers is not None else "")
            + (f"_heads-{self.num_attention_heads}" if self.num_attention_heads is not None else "")
            + (f"_hidden-{self.embedding_dim}" if self.embedding_dim is not None else "")
        )

    @property
    def experiment_dir(self):
        return (
            f"models/{self.language}/"
            + f"{'-'.join(self.tasks)}"
            + (f"_layers-{self.num_layers}" if self.num_layers is not None else "")
            + (f"_heads-{self.num_attention_heads}" if self.num_attention_heads is not None else "")
            + (f"_hidden-{self.embedding_dim}" if self.embedding_dim is not None else "")
            + (f"_{self.bert_model_name}" if self.bert_model_name is not None else "")
        )

    def prepare_dirs(self, delete=False):
        if delete:
            if os.path.exists(self.bert_dir):
                logger.info(f"{self.bert_dir} exists, removing...")
                shutil.rmtree(self.bert_dir)
            if os.path.exists(self.experiment_dir):
                logger.info(f"{self.experiment_dir} exists, removing...")
                shutil.rmtree(self.experiment_dir)
        os.makedirs(self.bert_dir, exist_ok=True)

    def prepare_bert_pretrain_env_vars(self):
        os.environ["TOKENIZER_PATH"] = self.bert_dir
        os.environ["NUM_LAYERS"] = str(self.num_layers)
        os.environ["NUM_ATTENTION_HEADS"] = str(self.num_attention_heads)
        os.environ["EMBEDDING_DIM"] = str(self.embedding_dim)
        # Discard any pretraining paths we don't need
        xpos = mlm = parser = False
        for k, v in self.pretrain_language_config.items():
            os.environ[k] = json.dumps(v)
            if k == "train_data_paths":
                xpos = "xpos" in v
                mlm = "mlm" in v
                parser = "parser" in v
        os.environ["XPOS"] = json.dumps(xpos)
        os.environ["MLM"] = json.dumps(mlm)
        os.environ["PARSER"] = json.dumps(parser)
