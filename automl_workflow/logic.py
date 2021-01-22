from .backbone_model import MyBackboneModel
from .data_augmentor import MyDataAugmentor
from .data_ingestor import MyDataIngestor
from .data_loader import MyDataLoader
from .ensembler import MyEnsembler


class LogicModel():

    def __init__(self, metadata, session=None):
        self.backbone = MyBackboneModel(metadata)

    def break_train_loop_condition(self, remaining_time_budget=None) -> bool:
        raise NotImplementedError

    def get_num_epoch(self):
        raise NotImplementedError

    def train(self, dataset, remaining_time_budget=None):
        dataset_train = MyDataIngestor(dataset)
        self.iterable_dataset_train = MyDataLoader(model='train').transform(dataset_train)
        while True:
            self.backbone.fit(self.iterable_dataset_train)
            if self.break_train_loop_condition(remaining_time_budget):
                break

    def test(self, dataset, remaining_time_budget=None):
        dataset_test = MyDataIngestor(dataset)
        self.iterable_dataset_test = MyDataLoader(model='test').transform(dataset_test)
        return self.backbone.predict(self.iterable_dataset_test)
