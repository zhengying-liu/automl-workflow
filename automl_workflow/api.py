"""API design for a universal AutoML workflow."""

from typing import List, Tuple, Dict


class IterableDataset(object):
    """Should be compatible with tf.data.Dataset, NumPy and 
    torch.utils.data.Dataset
    """

    def __iter__(self):
        """Each element should be a **list** of NumPy array-like objects. For 
        example, [example, label] where `example` and `label` are NumPy arrays.

        Predictions over one dataset also form an IterableDataset object.
        """
        raise NotImplementedError

    def generate_train_test_split(self, train_size=0.75) \
        -> Tuple[IterableDataset, IterableDataset]:
        """Generate trai/test split from this dataset.
        
        Args:
          train_size: float or int, If float, should be between 0.0 and 1.0 and 
            represent the proportion of the dataset to include in the train 
            split. If int, represents the absolute number of train samples.

        Returns:
          A tuple `(dataset_train, dataset_test)` where both `dataset_train`
            and `dataset_test` are IterableDataset objects.
        """
        raise NotImplementedError


class FeatureUnion(IterableDataset):

    def __init__(self, datasets: List[IterableDataset]):
        """Take a list of IterableDataset objects, return one single 
        IterableDataset object, which is the union.
        """
        raise NotImplementedError


class DataIngestor(object):

    def ingest(self, dataset) -> IterableDataset:
        """Take an AutoDL dataset object and ingest the data to an 
        IterableDataset object.
        """
        raise NotImplementedError


class BaseDataTransformer(object):

    def fit(self, dataset: IterableDataset):
        """
        Args:
          dataset: IterableDataset object, e.g. training set
        """
        raise NotImplementedError

    def transform(self, dataset: IterableDataset) -> IterableDataset:
        """Transform an (iterable) dataset object into another dataset object.
        Can also be used the `predict` method in scikit-learn.

        Args:
            dataset: IterableDataset object

        Returns:
            new_dataset: IterableDataset
        """
        raise NotImplementedError


class DataLoader(BaseDataTransformer):

    def transform(self, dataset: IterableDataset) -> IterableDataset:
        raise NotImplementedError


class DataAugmentor(BaseDataTransformer):

    def transform(self, dataset: IterableDataset) -> IterableDataset:
        raise NotImplementedError


class Pipeline(BaseDataTransformer):

    def __init__(self, transformers: List[BaseDataTransformer]):
        """Take a list of BaseDataTransformer objects, form one 
        BaseDataTransformer object by chaining them.
        """
        raise NotImplementedError


class HPOptimizer(object):
    """Adjust pipeline parameters in a data-driven way."""

    def __init__(self, pipeline: Pipeline, params: Dict=None):
        """

        Args:
          pipeline: Pipeline object
          params: dict, e.g. {'transformer1': {'C':0.1, 'gamma':1.0}}. Initial
            hyper-parameters values.
        """
        raise NotImplementedError

    def fit(self, dataset: IterableDataset) -> Pipeline:
        """Adjust the parameters in the pipeline.

        Returns:
          a new Pipeline object with specified (hyper-)parameters.
        """
        raise NotImplementedError


class Ensembler(object):

    def fit(self, features: List[IterableDataset], label: IterableDataset):
        """
        Args:
          features: list of IterableDataset objects. Can be predictions from 
            different models.
          label: an IterableDataset object with list length 1.
        """
        raise NotImplementedError

    def predict(self, test_features: List[IterableDataset]) -> IterableDataset:
        """
        Args:
          test_features: list of IterableDataset objects

        Returns:
          an IterableDataset object, predicted labels
        """
        raise NotImplementedError


# Multiple transformers form a pipeline (by chaining)
# Multiple pipelines form an ensemble (in parallel)


class Model(object):

    def train(self, dataset, data_loader=None, data_augmentor=None,
              ensembler=None, hpoptimizer_cls=None):
        # Forming pipeline using multiple transformers
        pipeline = Pipeline([data_loader, data_augmentor])
        train_dataset, valid_dataset =dataset.generate_train_test_split()

        # HPO
        hpoptimizer = hpoptimizer_cls(pipeline)
        pipeline = hpoptimizer.fit(train_dataset)

        # Fit pipeline
        pipeline.fit(train_dataset)
        self.pipeline = pipeline

        # TODO: how to form ensemble
        self.ensembler = ensembler
        features = [p(train_dataset) for p in pipelines]
        label = map(dataset, lambda x,y:y)
        self.ensembler.fit(features, label)

        # Validation
        pred_valid = self.ensembler.predict(valid_dataset)
        acc_valid = accuracy(label_valid, pred_valid)

    def train(self, dataset, transformers=None, ensembler=None):
        self.ensembler = emsembler
        features = [t(dataset) for t in transformers]
        label = map(dataset, lambda x,y:y)
        self.ensembler.fit(features, label)

    def test(self, dataset):
        predictions = self.ensembler.predict(dataset)
        return predictions


############## Example code ################

class MyDataIngestor(DataIngestor):

    def ingest(self, dataset):
        return dataset

class MyDataLoader(DataLoader):

    def transform(self, dataset):
        return dataset

class MyDataAugmentor(DataAugmentor):

    def transform(self, dataset):
        return dataset
