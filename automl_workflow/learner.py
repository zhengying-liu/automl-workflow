# Participants should implement the following classes
from automl_workflow.backbone_model import MyBackboneModel
from automl_workflow.data_augmentor import MyDataAugmentor
from automl_workflow.data_ingestor import MyDataIngestor
# from automl_workflow.ensembler import MyEnsembler
from automl_workflow.optimizer import MyOptimizer
from automl_workflow.data_loader import MyDataLoader


from automl_workflow.api import Learner, Predictor


class MyPredictor(Predictor):

    def __init__(self, pt_model):
        self.pt_model = pt_model

    def predict(self, x):
        return self.pt_model.forward(x)


class MyLearner(Learner):

    def __init__(self):
        backbone_model = MyBackboneModel()
        optimizer = MyOptimizer()
        data_loader = MyDataLoader()
        super().__init__(
            backbone_model=backbone_model,
            optimizer=optimizer,
            data_loader=data_loader,
        )

    def learn(self, train_set):
        trainloader = self.data_loader(train_set, train=True)
        self.optimizer.optimize(self.backbone_model, trainloader)
        predictor = MyPredictor(self.backbone_model)
        return predictor