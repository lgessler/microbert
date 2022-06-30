import os
import subprocess
import logging


def get_git_revision_hash(self):
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


logger = logging.getLogger(__name__)


class Config:
    def __init__(self, *args, **kwargs):
        self.language = kwargs.pop("language")
        self.finetune = kwargs.pop("finetune")
        self.debug = kwargs.pop("debug", False)
        self.experiment_config = None
        if self.debug:
            os.environ["TOY_DATA"] = "true"

    # Fall back on self.experiment_config if we don't get an attr hit directly
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif self.experiment_config is not None and hasattr(self, "experiment_config"):
            return getattr(self.experiment_config, attr)
        else:
            raise AttributeError("%r object has no attribute %r" % (self.__class__.__name__, attr))
