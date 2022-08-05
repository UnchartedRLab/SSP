from abc import ABC, abstractmethod
from pathlib import Path
from typing import NoReturn, Dict
import shutil

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from src.loss import NTXentLoss
from src.models import ResNetSimCLR, ScatSimCLR, HarmSimCLR
from src.evaluation import LogisticRegressionEvaluator
from src.data import EmbeddingExtractor
from src.data.datasets import NUM_CLASSES


def get_device() -> str:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def save_config_file(model_checkpoints_folder: Path) -> NoReturn:
    SERIAL_EXEC = xmp.MpSerialExecutor()
    SERIAL_EXEC.run(model_checkpoints_folder.mkdir(parents=True, exist_ok=True))
    SERIAL_EXEC.run(shutil.copy('./config.yaml', model_checkpoints_folder / 'config.yaml'))


class BaseTrainer(ABC):
    EMBEDDINGS_MODELS = ['resnet18', 'resnet50',
                         'scatsimclr8', 'scatsimclr12', 'scatsimclr16', 'scatsimclr30', 'scatsimclr45',
                         'harmsimclr8', 'harmsimclr12', 'harmsimclr16', 'harmsimclr30', 'harmsimclr45']
    
   

    def __init__(self, config: Dict):
        self._config = config
        self._device = get_device()
        self._writer = SummaryWriter()

        self._nt_xent_criterion = NTXentLoss(self._device, config['batch_size'], **config['loss'])

    @abstractmethod
    def train(self):
        pass

    def evaluate(self) -> float:
        # load model
        model = self._get_embeddings_model(self._config['model']['base_model'])
        model = self._load_weights(model)
        #model = xmp.MpModelWrapper(model)
        model.to(self._device)
        model.eval()

        score = self._test_classification(model)
        return score

    @abstractmethod
    def _step(self, *args, **kwargs):
        pass

    @abstractmethod
    def _load_weights(self, *args, **kwargs):
        pass

    @abstractmethod
    def _validate(self, *args, **kwargs):
        pass

    def _test_classification(self, model) -> float:

        model.eval()

        input_size = eval(self._config['dataset']['input_shape'])
        dataset = self._config['dataset']['dataset_test']
        epochs = 100

        extractor = EmbeddingExtractor(model, device=self._device,
                                       dataset=dataset,
                                       input_size=input_size, batch_size=self._config['batch_size'])
        train_data, train_labels, test_data, test_labels = extractor.get_features()

        if self._config['dataset']['dump_trace']:
            np.savez_compressed(self._config['dataset']['trace_name'], train_data=train_data, train_label=train_labels, test_data=test_data, test_label=test_labels)

        evaluator = LogisticRegressionEvaluator(n_features=train_data.shape[1],
                                                n_classes=NUM_CLASSES[dataset],
                                                device=self._device, batch_size=64)
        accuracy = evaluator.run_evaluation(train_data, train_labels, test_data, test_labels, epochs)
        print(accuracy)
        return accuracy

    def _get_embeddings_model(self, model_name: str) -> nn.Module:

        if model_name not in self.EMBEDDINGS_MODELS:
            raise ValueError('Unsupported model')

        out_dim = self._config['model']['out_dim']

        if 'resnet' in model_name:
            return ResNetSimCLR(base_model=model_name, out_dim=out_dim)

        if 'harmsimclr' in model_name:
            blocks = int(model_name[10:])
            input_size = eval(self._config['dataset']['input_shape'])

            return HarmSimCLR(out_dim=out_dim)

        if 'scatsimclr' in model_name:
            blocks = int(model_name[10:])
            input_size = eval(self._config['dataset']['input_shape'])

            J = self._config['model']['J']
            L = self._config['model']['L']

            return ScatSimCLR(J=J, L=L, input_size=input_size,
                              res_blocks=blocks, out_dim=out_dim)