import subprocess


def get_git_revision_hash(self):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


# We want:
# - All experimental parameters in here
# - An easy way to write files that is (1) human-friendly, (2) fully descriptive
# - A fully serializable representation (so it can be written into dirs)
class ExpConfig:
    def __init__(self):
        pass

_ = ["berts", "/", "$args",]
