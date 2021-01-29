from .dataset import add_dataset_args
from .classifier import add_classifier_args


def add_task_args(parser):
    add_dataset_args(parser)
    add_classifier_args(parser)
