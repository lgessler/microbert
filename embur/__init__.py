# These imports are important for making the configuration files find the classes that you wrote.
# If you don't have these, you'll get errors about allennlp not being able to find
# "simple_classifier", or whatever name you registered your model with.  These imports and the
# contents of .allennlp_plugins makes it so you can just use `allennlp train`, and we will find your
# classes and use them.  If you change the name of `embur`, you'll also need to change it in
# the same way in the .allennlp_plugins file.
# from embur.rel.baseline_model import *
# from embur.rel.dataset_reader import *

# https://stackoverflow.com/questions/3365740/how-to-import-all-submodules
from embur.schedulers import HomogeneousRepeatedRoundRobinScheduler
import embur.dataset_reader
import embur.schedulers
import embur.models.entities.entity_crf
import embur.models.heads.mlm
import embur.models.heads.ud
import embur.models.heads.xpos
import embur.models.backbones.contextualized_backbone
import embur.models.backbones.static_backbone
import embur.models.backbones.bert_backbone
