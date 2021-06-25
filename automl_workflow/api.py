"""API design for a universal AutoML workflow."""

from typing import List, Tuple, Dict

from torch.utils.data import Dataset, IterableDataset


class TFDataset(Dataset):

    def __init__(self, session, dataset, num_samples):
        super(TFDataset, self).__init__()
        self.session = session
        self.dataset = dataset
        self.num_samples = num_samples
        self.next_element = None

        self.reset()

    def reset(self):
        dataset = self.dataset
        iterator = dataset.make_one_shot_iterator()
        self.next_element = iterator.get_next()
        return self

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        session = self.session if self.session is not None else tf.Session()
        try:
            example, label = session.run(self.next_element)
        except tf.errors.OutOfRangeError:
            self.reset()
            raise StopIteration

        return example, label

    def scan(self, samples=1000000, with_tensors=False, is_batch=False, device=None, half=False):
        shapes, counts, tensors = [], [], []
        labels = []
        min_list, max_list = [], []
        is_255 = False
        for i in range(min(self.num_samples, samples)):
            try:
                example, label = self.__getitem__(i)
                if i == 0 and np.mean(example) > 1:
                    is_255 = True
            except tf.errors.OutOfRangeError:
                break
            except StopIteration:
                break

            shape = example.shape
            count = np.sum(label, axis=None if not is_batch else -1)
            labels.append(label)

            shapes.append(shape)
            counts.append(count)
            min_list.append(np.min(example))
            max_list.append(np.max(example))

            if with_tensors:
                example = torch.Tensor(example)
                label = torch.Tensor(label)

                example.data = example.data.to(device=device)
                if half and example.is_floating_point():
                    example.data = example.data.half()

                label.data = label.data.to(device=device)
                if half and label.is_floating_point():
                    label.data = label.data.half()

                tensors.append([example, label])

        shapes = np.array(shapes)
        counts = np.array(counts) if not is_batch else np.concatenate(counts)

        labels = np.array(labels) if not is_batch else np.concatenate(labels)
        num_samples = labels.shape[0]
        labels = np.sum(labels, axis=0)
        zero_count = sum(labels == 0)

        pos_weights = (num_samples - labels + 10) / (labels + 10)
        info = {
            'count': len(counts),
            'is_multiclass': counts.max() > 1.0,
            'is_video': int(np.median(shapes, axis=0)[0]) > 1,
            'example': {
                'shape': [int(v) for v in np.median(shapes, axis=0)],
                'shape_avg': [int(v) for v in np.average(shapes, axis=0)],
                'value': {'min': min(min_list), 'max': max(max_list)},
                'is_255': is_255
            },
            'label': {
                'min': counts.min(),
                'max': counts.max(),
                'average': counts.mean(),
                'median': np.median(counts),
                'zero_count': zero_count,
                'pos_weights': pos_weights
            },

        }

        if with_tensors:
            return info, tensors
        return info


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

        Examples:
          pre-processing, feature engineering, missing value imputation can be 
          done in this method.
        """
        raise NotImplementedError


class BaseDataTransformer(object):
    """Base class for implementing e.g. a model, a feature extractor"""

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

    def fit(self, dataset):
        """Apply `fit` and `transform` method of each transformer."""
        pass

    def transform(self, dataset):
        """Apple `transform` method of each transformer."""
        pass


class HPOptimizer(object):
    """Adjust pipeline parameters in a data-driven way."""

    def __init__(self, pipeline: Pipeline=None, params: Dict=None):
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


class FeatureEnsembler(object):

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


class LearnerEnsembler(object):

    def __init__(self, learners):
        pass

    def fit(self, dataset):
        pass

    def predict(self, dataset):
        pass


# Multiple transformers form a pipeline (by chaining)
# Multiple pipelines form an ensemble (in parallel)


class Model(object):

    def train(self, dataset, 
            data_loader=None, 
            data_augmentor=None,
            ensembler=None, 
            hpoptimizer_cls=None, 
            learner=None):
        """
        Args:
          dataset: IterableDataset object, training set
        """
        # Forming pipeline using multiple transformers
        pipeline = Pipeline([data_loader, data_augmentor, learner])
        train_dataset, valid_dataset =dataset.generate_train_test_split()

        # HPO
        hpoptimizer = hpoptimizer_cls(pipeline)
        pipeline = hpoptimizer.fit(train_dataset)

        # Fit pipeline
        pipeline.fit(train_dataset)
        self.pipeline = pipeline

        # # TODO: how to form ensemble
        # self.ensembler = ensembler
        # features = [p(train_dataset) for p in pipelines]
        # label = map(dataset, lambda x,y:y)
        # self.ensembler.fit(features, label)

        # Validation
        pred_valid = self.pipeline.transform(valid_dataset)
        label_valid = valid_dataset[1] # TODO
        acc_valid = accuracy(label_valid, pred_valid)

    def test(self, dataset):
        """
        Args:
          dataset: IterableDataset object, test set
        """
        predictions = self.pipeline.transform(dataset)
        return predictions



class Task(object):

    def __init__(self, train_set, metric, test_set=None):
        self.train_set = train_set
        self.test_set = test_set
        self.metric = metric


class Metric(object):

    def __call__(self, y_pred, y_true):
        """`y_pred` (and `y_true`) should be an iterable of several
        predictions.
        """
        raise NotImplementedError


class Predictor(object):

    def predict(self, test_set: Dataset) -> Dataset:
        """Return the label of examples in `test_set`."""
        raise NotImplementedError


class ClassicLearner(object):

    def learn(self, train_set: Dataset) -> Predictor:
        """Return a Predictor object."""
        raise NotImplementedError


class Learner(ClassicLearner):

    def __init__(self, 
        backbone_model=None,
        data_loader=None,
        data_ingestor=None,
        data_augmentor=None,
        ensembler=None,
        optimizer=None,
        loss_func=None,
        hp_optimizer=None,
        ):
        self.backbone_model = backbone_model
        self.data_loader = data_loader
        self.data_ingestor = data_ingestor
        self.data_augmentor = data_augmentor
        self.ensembler = ensembler
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.hp_optimizer = hp_optimizer






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
