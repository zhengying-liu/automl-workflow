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


class FeatureUnion(IterableDataset):

    def __init__(self, datasets):
        """Take a list of IterableDataset objects, return one single 
        IterableDataset object, which is the union.
        """
        raise NotImplementedError


class DataIngestor(object):

    def ingest(self, dataset):
        """Take an AutoDL dataset object and ingest the data to an 
        IterableDataset object.
        """
        raise NotImplementedError


class BaseDataTransformer(object):

    def fit(self, dataset):
        """
        Args:
          dataset: IterableDataset object, e.g. training set
        """
        raise NotImplementedError

    def transform(self, dataset):
        """Transform an (iterable) dataset object into another dataset object.

        Args:
            dataset: IterableDataset object

        Returns:
            new_dataset: IterableDataset
        """
        raise NotImplementedError


class DataLoader(BaseDataTransformer):

    def transform(self, dataset):
        raise NotImplementedError


class DataAugmentor(BaseDataTransformer):

    def transform(self, dataset):
        raise NotImplementedError


class Pipeline(BaseDataTransformer):

    def __init__(self, transformers):
        """Take a list of BaseDataTransformer objects, form one 
        BaseDataTransformer object by chaining them.
        """
        raise NotImplementedError


class HPOptimizer(object):
    """Adjust pipeline parameters in a data-driven way."""

    def __init__(self, pipeline, params=None):
        """
        Args:
          pipeline: Pipeline object
          params: dict, e.g. {'transformer1': {'C':0.1, 'gamma':1.0}}
        """
        pass

    def fit(self, dataset):
        """Adjust the parameters in the pipeline."""
        return new_pipeline


class Ensembler(object):

    def fit(self, features, label):
        """
        Args:
          features: list of IterableDataset objects. Can be predictions from 
            different models.
          label: an IterableDataset object with list length 1.
        """
        raise NotImplementedError

    def predict(self, test_features):
        """
        Args:
          test_features: list of IterableDataset objects

        Returns:
          an IterableDataset object, predicted labels
        """
        raise NotImplementedError


# Multiple transformers form a pipeline
# Multiple pipelines form an ensemble


class Model(object):

    def train(self, dataset, data_loader=None, data_augmentor=None,
              ensembler=None, hpoptimizer_cls=None):
        # Forming pipeline using multiple transformers
        pipeline = Pipeline([data_loader, data_augmentor])
        train_dataset, valid_dataset = train_valid_split(dataset)

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
