import logging
import argparse
import importlib
import os

MODEL_REGISTRY = {}
MODEL_CLASS_NAMES = set()

def setup_model(config, *args, **kwargs):
    logging.info("Setup model: %s" % config.model_name)
    return MODEL_REGISTRY[config.model_name](config, *args, **kwargs)

def register_model(name):

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        # if not issubclass(cls, FairseqTask):
        #     raise ValueError('Task ({}: {}) must extend FairseqTask'.format(name, cls.__name__))
        if cls.__name__ in MODEL_CLASS_NAMES:
            raise ValueError('Cannot register task with duplicate class name ({})'.format(cls.__name__))

        MODEL_REGISTRY[name] = cls
        MODEL_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_model_cls

# automatically import any Python files in the models/ directory
exclude = []
models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if not file.startswith('_') and not file.startswith('.') and (file.endswith('.py') or os.path.isdir(path)):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        if model_name in exclude:
            # print(f"SKIP {model_name}")
            continue
            
        module = importlib.import_module('models.' + model_name)

        