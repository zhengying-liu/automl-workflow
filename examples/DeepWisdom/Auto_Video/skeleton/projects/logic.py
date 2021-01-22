# -*- coding: utf-8 -*-
from __future__ import absolute_import
import random

import tensorflow as tf
import torchvision as tv
import torch
import numpy as np

from .api import Model
from .others import *
import skeleton
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

LOGGER = get_logger(__name__)

class EnsembleConfig(object):
    ENABLE_PRE_ENSE = True
    ENS_TOP_VLOSS_NUM = 2
    ENS_TOP_VACC_NUM = 2
    ENS_TOP_ONE_VACC_NUM=1
    ENS_TOP_MODEL_NUM=3
    MODEL_INDEX =0
    MAX_TIMES_LIST= [3,10,12,12]
    THRESHOLD_DIFF_LIST = [0.001 for i in range(len(MAX_TIMES_LIST))]

class LogicModel(Model):
    def __init__(self, metadata, session=None):
        super(LogicModel, self).__init__(metadata)
        LOGGER.info('--------- Model.metadata ----------')
        LOGGER.info('path: %s', self.metadata.get_dataset_name())
        LOGGER.info('shape:  %s', self.metadata.get_tensor_size(0))
        LOGGER.info('size: %s', self.metadata.size())
        LOGGER.info('num_class:  %s', self.metadata.get_output_size())

        test_metadata_filename = self.metadata.get_dataset_name().replace('train', 'test') + '/metadata.textproto'
        self.num_test = [int(line.split(':')[1]) for line in open(test_metadata_filename, 'r').readlines()[:3] if 'sample_count' in line][0]

        self.release_freeze = False
        self.epoch_metrics = dict()
        self.ensembleconfig = EnsembleConfig()
        self.best_score = 0.95
        self.best_loss = 100.00
        self.history_preds = None

        self.base_info = {
            'dataset': {
                'path': self.metadata.get_dataset_name(),
                'shape': self.metadata.get_tensor_size(0),
                'size': self.metadata.size(),
                'num_class': self.metadata.get_output_size()
            },
            'terminate':False
        }

        self.base_hyper_params = {
            'dataset': {
                'max_epoch': 1000,  # initial value
            },
            'conditions': {
                'score_type': 'auc',
                'early_epoch': 1,
                'skip_valid_score_threshold': 0.90,  # if bigger then 1.0 is not use
                'skip_valid_after_test': min(10, max(3, int(self.base_info['dataset']['size'] // 1000))),
                'test_after_at_least_seconds': 1,
                'test_after_at_least_seconds_max': 90,
                'test_after_at_least_seconds_step': 2,
                'threshold_valid_score_diff': 0.01,
                'threshold_valid_best_score': 0.997,
                'decide_threshold_valid_best_score': 0.9300,
                'max_inner_loop_ratio': 0.1,
                'min_lr': 1e-6,
                'use_fast_auto_aug': True
            }
        }

        self.ensemble_test_index = 0
        self.change_next_model = False
        self.ensemble_predictions = dict()
        self.ensemble_predict_prob_list = dict()
        self.ensemble_checkpoints = dict()
        self.ensemble_g_his_eval_dict = dict()
        self.start_ensemble = False
        self.ensemble_lr = [1.0]
        self.ensemble_epoch = [-1]

        self.timers = {
            'train': skeleton.utils.Timer(),
            'test': skeleton.utils.Timer()
        }

        LOGGER.info('[init] build')
        self.build()
        LOGGER.info('[init] session')

        self.dataloaders = {
            'train': None,
            'valid': None,
            'test': None
        }
        self.is_skip_valid = True
        LOGGER.info('[init] done')

    def __repr__(self):
        return '\n---------[{0}]---------\nbase_info:{1}\ninfo:{2}\nparams:{3}\n---------- ---------'.format(
            self.__class__.__name__,
            self.base_info,self.model.info,self.model.hyper_params
        )

    def build(self):
        raise NotImplementedError

    def update_model(self):
        # call after to scan train sample
        pass

    def epoch_train(self, epoch, train):
        raise NotImplementedError

    def epoch_valid(self, epoch, valid):
        raise NotImplementedError

    def skip_valid(self, epoch):
        raise NotImplementedError

    def prediction(self, dataloader, model, checkpoints):
        raise NotImplementedError

    def adapt(self):
        raise NotImplementedError

    def is_multiclass(self):
        return self.base_info['dataset']['sample']['is_multiclass']

    def is_video(self):
        return self.base_info['dataset']['sample']['is_video']

    def release_model(self,remain_times):
        if remain_times < 1120 and self.ensembleconfig.MODEL_INDEX ==0:
            for m in self.model.layer1.modules():
                for p in m.parameters():
                    p.requires_grad_(True)
            for m in self.model.stem.modules():
                for p in m.parameters():
                    p.requires_grad_(True)
        self.release_freeze = True

    def build_or_get_train_dataloader(self, dataset):
        if not self.model.info['condition']['first']['train']:
            return self.build_or_get_dataloader('train')

        num_images = self.base_info['dataset']['size']

        # split train/valid
        num_valids = int(min(num_images * self.model.hyper_params['dataset']['cv_valid_ratio'], self.model.hyper_params['dataset']['max_valid_count']))
        num_trains = num_images - num_valids
        LOGGER.info('[cv_fold] num_trains:%d num_valids:%d', num_trains, num_valids)

        LOGGER.info('[%s] scan before', 'sample')
        num_samples = self.model.hyper_params['dataset']['train_info_sample']
        sample = dataset.take(num_samples).prefetch(buffer_size=num_samples)
        train = skeleton.data.TFDataset(self.session, sample, num_samples)
        self.base_info['dataset']['sample'] = train.scan(samples=num_samples)
        del train
        del sample
        LOGGER.info('[%s] scan after', 'sample')

        # input_shape = [min(s, self.model.hyper_params['dataset']['max_size']) for s in self.info['dataset']['shape']]
        times, height, width, channels = self.base_info['dataset']['sample']['example']['shape']
        values = self.base_info['dataset']['sample']['example']['value']
        aspect_ratio = width / height

        # fit image area to 64x64
        if aspect_ratio > 2 or 1. / aspect_ratio > 2:
            self.model.hyper_params['dataset']['max_size'] *= 2
        size = [min(s, self.model.hyper_params['dataset']['max_size']) for s in [height, width]]

        # keep aspect ratio
        if aspect_ratio > 1:
            size[0] = size[1] / aspect_ratio
        else:
            size[1] = size[0] * aspect_ratio

        # too small image use original image
        if width <= 32 and height <= 32:
            input_shape = [times, height, width, channels]
        else:
            fit_size_fn = lambda x: int(x / self.model.hyper_params['dataset']['base'] + 0.8) * self.model.hyper_params['dataset']['base']
            size = list(map(fit_size_fn, size))
            min_times = min(times, self.model.hyper_params['dataset']['max_times'])
            input_shape = [fit_size_fn(min_times) if min_times > self.model.hyper_params['dataset']['base'] else min_times] + size + [channels]

        if self.is_video():
            self.model.hyper_params['dataset']['batch_size'] = int(self.model.hyper_params['dataset']['batch_size'] // 2)
        LOGGER.info('[input_shape] origin:%s aspect_ratio:%f target:%s', [times, height, width, channels], aspect_ratio, input_shape)

        self.model.hyper_params['dataset']['input'] = input_shape

        num_class = self.base_info['dataset']['num_class']
        batch_size = self.model.hyper_params['dataset']['batch_size']
        if num_class > batch_size / 2 and not self.is_video():
            self.model.hyper_params['dataset']['batch_size'] = batch_size * 2
        batch_size = self.model.hyper_params['dataset']['batch_size']

        preprocessor1 = get_tf_resize(input_shape[1], input_shape[2], times=input_shape[0], min_value=values['min'], max_value=values['max'])
        
        dataset = dataset.map(
            lambda *x: (preprocessor1(x[0]), x[1]),
            num_parallel_calls=4
        )

        must_shuffle = self.base_info['dataset']['sample']['label']['zero_count'] / self.base_info['dataset']['num_class'] >= 0.5
        enough_count = self.model.hyper_params['dataset']['enough_count']['video'] if self.is_video() else self.model.hyper_params['dataset']['enough_count']['image']
        if must_shuffle or num_images < enough_count:
            dataset = dataset.shuffle(buffer_size=min(enough_count, num_images), reshuffle_each_iteration=False)
            LOGGER.info('[dataset] shuffle before split train/valid')

        train = dataset.skip(num_valids)
        valid = dataset.take(num_valids)
        self.datasets = {
            'train': train,
            'valid': valid,
            'num_trains': num_trains,
            'num_valids': num_valids
        }
        return self.build_or_get_dataloader('train', self.datasets['train'], num_trains)

    def build_or_get_dataloader(self, mode, dataset=None, num_items=0):
        if mode in self.dataloaders and self.dataloaders[mode] is not None:
            return self.dataloaders[mode]

        enough_count = self.model.hyper_params['dataset']['enough_count']['video'] if self.is_video() else self.model.hyper_params['dataset']['enough_count']['image']

        LOGGER.debug('[dataloader] %s build start', mode)
        values = self.base_info['dataset']['sample']['example']['value']
        if mode == 'train':
            batch_size = self.model.hyper_params['dataset']['batch_size']
            # input_shape = self.model.hyper_params['dataset']['input']
            preprocessor = get_tf_to_tensor(is_random_flip=True)
            # dataset = dataset.prefetch(buffer_size=batch_size * 3)

            if num_items < enough_count:
                dataset = dataset.cache()

            dataset = dataset.repeat()
            dataset = dataset.map(
                lambda *x: (preprocessor(x[0]), x[1]),
                num_parallel_calls=4
            )
            dataset = dataset.prefetch(buffer_size=batch_size * 8)

            dataset = skeleton.data.TFDataset(self.session, dataset, num_items)

            transform = tv.transforms.Compose([
                # skeleton.data.Cutout(int(input_shape[1] // 4), int(input_shape[2] // 4))
            ])
            dataset = skeleton.data.TransformDataset(dataset, transform, index=0)

            self.dataloaders['train'] = skeleton.data.FixedSizeDataLoader(
                dataset,
                steps=self.model.hyper_params['dataset']['steps_per_epoch'],
                batch_size=batch_size,
                shuffle=False, drop_last=True, num_workers=0, pin_memory=False
            )
        
        elif mode in ['valid', 'test']:
            batch_size = self.model.hyper_params['dataset']['batch_size_test']
            input_shape = self.model.hyper_params['dataset']['input']

            preprocessor2 = get_tf_to_tensor(is_random_flip=False)
            if mode == 'valid':
                preprocessor = preprocessor2
            else:
                preprocessor1 = get_tf_resize(input_shape[1], input_shape[2], times=input_shape[0], min_value=values['min'], max_value=values['max'])
                preprocessor = lambda *tensor: preprocessor2(preprocessor1(*tensor))

            tf_dataset = dataset.apply(
                tf.data.experimental.map_and_batch(
                    map_func=lambda *x: (preprocessor(x[0]), x[1]),
                    batch_size=batch_size,
                    drop_remainder=False,
                    num_parallel_calls=4
                )
            ).prefetch(buffer_size=8)

            dataset = skeleton.data.TFDataset(self.session, tf_dataset, num_items)

            LOGGER.info('[%s] scan before', mode)
            self.base_info['dataset'][mode], tensors = dataset.scan(
                with_tensors=True, is_batch=True,
                device=self.device, half=self.is_half
            )
            tensors = [torch.cat(t, dim=0) for t in zip(*tensors)]
            LOGGER.info('[%s] scan after', mode)

            del tf_dataset
            del dataset
            dataset = skeleton.data.prefetch_dataset(tensors)
            if 'valid' == mode:
                transform = tv.transforms.Compose([
                ])
                dataset = skeleton.data.TransformDataset(dataset, transform, index=0)

            self.dataloaders[mode] = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.model.hyper_params['dataset']['batch_size_test'],
                shuffle=False, drop_last=False, num_workers=0, pin_memory=False
            )

            self.model.info['condition']['first'][mode] = False

        LOGGER.debug('[dataloader] %s build end', mode)
        return self.dataloaders[mode]

    def update_condition(self, metrics=None):
        self.model.info['condition']['first']['train'] = False
        self.model.info['loop']['epoch'] += 1

        metrics.update({'epoch': self.model.info['loop']['epoch']})
        self.model.checkpoints.append(metrics)

        indices = np.argsort(np.array([v['valid']['score'] for v in self.model.checkpoints] if len(self.model.checkpoints) > 0 else [0]))
        indices = sorted(indices[::-1][:self.model.hyper_params['checkpoints']['keep']])
        self.model.checkpoints = [self.model.checkpoints[i] for i in indices]
    
    def break_train_loop_condition(self, remaining_time_budget=None, inner_epoch=1):
        consume = inner_epoch * self.timers['train'].step_time

        best_idx = np.argmax(np.array([c['valid']['score'] for c in self.model.checkpoints]))
        best_epoch = self.model.checkpoints[best_idx]['epoch']
        best_loss = self.model.checkpoints[best_idx]['valid']['loss']
        best_score = self.model.checkpoints[best_idx]['valid']['score']
        lr = self.model.optimizer.get_learning_rate()
        LOGGER.debug('[CONDITION] best (epoch:%04d loss:%.2f score:%.2f) lr:%.8f time delta:%.2f',
                     best_epoch, best_loss, best_score, lr, consume)

        ## model epoch <=0
        if self.model.info['loop']['epoch'] <= self.model.hyper_params['conditions']['early_epoch']:
            LOGGER.info('[BREAK] early %d epoch', self.model.hyper_params['conditions']['early_epoch'])
            return True

        ## bset_score > 0.997
        if best_score > self.model.hyper_params['conditions']['threshold_valid_best_score']:
            LOGGER.info('[BREAK] achieve best score %f', best_score)
            return True

        ## consume time > threshold
        if consume > self.model.hyper_params['conditions']['test_after_at_least_seconds'] and \
            self.model.checkpoints[best_idx]['epoch'] > self.model.info['loop']['epoch'] - inner_epoch and \
            best_score > self.model.info['loop']['best_score'] * 1.001:
            # increase hyper param
            self.model.hyper_params['conditions']['test_after_at_least_seconds'] = min(
                self.model.hyper_params['conditions']['test_after_at_least_seconds_max'],
                self.model.hyper_params['conditions']['test_after_at_least_seconds'] + self.model.hyper_params['conditions']['test_after_at_least_seconds_step']
            )

            self.model.info['loop']['best_score'] = best_score
            LOGGER.info('[BREAK] found best model (score:%f)', best_score)
            return True

        ## lr < 1e-6
        if lr < self.model.hyper_params['conditions']['min_lr']:
            LOGGER.info('[BREAK] too small lr (lr:%f < %f)', lr, self.model.hyper_params['conditions']['min_lr'])
            return True

        early_term_budget = 3 * 60
        expected_more_time = (self.timers['test'].step_time + (self.timers['train'].step_time * 2)) * 1.5
        
        if remaining_time_budget is not None and \
            remaining_time_budget - early_term_budget < expected_more_time:
            LOGGER.info('[BREAK] not enough time to train (remain:%f need:%f)', remaining_time_budget, expected_more_time)
            return True
        
        ## epoch >=20 inner_epoch > 100 
        if self.model.info['loop']['epoch'] >= 20 and \
            inner_epoch > self.model.hyper_params['dataset']['max_epoch'] * self.model.hyper_params['conditions']['max_inner_loop_ratio']:
            LOGGER.info('[BREAK] cannot found best model in too many epoch')
            return True

        return False

    def decide_change_next_model(self,remaining_time_budget=None, inner_epoch=0):
        
        if self.ensembleconfig.MODEL_INDEX ==0:
            valid_abs_losses = 10
            valid_abs_scores = False
        else:
            valid_losses = [(c['valid']['loss']-self.best_loss) for c in self.model.checkpoints][-10:]
            valid_abs_losses = sum([num > 0.0 for num in valid_losses])
            # valid_sub_scores = [(c['valid']['score']) for c in self.model.checkpoints][-5:]
            # valid_abs_scores = np.mean(np.array(valid_sub_scores))>0.997
            
        valid_scores = [c['valid']['score'] for c in self.model.checkpoints]
        diff = (max(valid_scores) - min(valid_scores)) * (1 - max(valid_scores))
        train_scores = [c['train']['score'] for c in self.model.checkpoints]
        
        threshold = self.model.hyper_params['conditions']['threshold_valid_score_diff']
        
        early_stop_scores = np.mean(np.array(train_scores[-3:])) ==1.00

        if self.ensembleconfig.MODEL_INDEX == 0 and remaining_time_budget <= 900:
            self.ensembleconfig.MODEL_INDEX == 0
            self.change_next_model = False

        elif self.ensembleconfig.MODEL_INDEX ==0 and remaining_time_budget>=1000 and early_stop_scores or \
            (diff < threshold and self.model.info['loop']['epoch'] >= 20):
            LOGGER.info('[START ENSEMBLE] too small score change (diff:%f < %f)', diff, threshold)
            self.ensemble_checkpoints[self.ensembleconfig.MODEL_INDEX] = self.model.checkpoints
            self.ensemble_predict_prob_list[self.ensembleconfig.MODEL_INDEX]= self.model.predict_prob_list
            self.ensemble_g_his_eval_dict[self.ensembleconfig.MODEL_INDEX] = self.model.g_his_eval_dict
            self.ensemble_epoch.append(self.model.info['loop']['epoch'])
            self.ensemble_lr.append(self.model.optimizer.get_learning_rate())

            self.ensembleconfig.MODEL_INDEX += 1
            if self.ensembleconfig.MODEL_INDEX == len(self.model_space):
                self.ensembleconfig.MODEL_INDEX = len(self.model_space) - 1
                self.change_next_model = False
            else:
                self.ensembleconfig.MODEL_INDEX = self.ensembleconfig.MODEL_INDEX
                self.change_next_model = True

            return self.change_next_model

        # elif valid_abs_losses == 0 or valid_abs_scores or (diff < threshold and self.model.info['loop']['epoch'] >= 20):
        elif valid_abs_losses == 0 or (diff < threshold and self.model.info['loop']['epoch'] >= 20):
            LOGGER.info('[START ENSEMBLE] too small score change (diff:%f < %f)', diff, threshold)
            self.ensemble_checkpoints[self.ensembleconfig.MODEL_INDEX] = self.model.checkpoints
            self.ensemble_predict_prob_list[self.ensembleconfig.MODEL_INDEX]= self.model.predict_prob_list
            self.ensemble_g_his_eval_dict[self.ensembleconfig.MODEL_INDEX] = self.model.g_his_eval_dict
            self.ensemble_epoch.append(self.model.info['loop']['epoch'])
            self.ensemble_lr.append(self.model.optimizer.get_learning_rate())

            self.ensembleconfig.MODEL_INDEX += 1
            if self.ensembleconfig.MODEL_INDEX == len(self.model_space):
                self.ensembleconfig.MODEL_INDEX = len(self.model_space) -1
                self.change_next_model = False
            else:
                self.ensembleconfig.MODEL_INDEX = self.ensembleconfig.MODEL_INDEX
                self.change_next_model = True

            return self.change_next_model
        else:
            self.change_next_model = False
        
            return self.change_next_model

    def terminate_train_loop_condition(self, remaining_time_budget=None, inner_epoch=0):
        early_term_budget = 3 * 60
        expected_more_time = (self.timers['test'].step_time + (self.timers['train'].step_time * 2)) * 1.5
        
        ## remaining time budget 
        if remaining_time_budget is not None and \
            remaining_time_budget - early_term_budget < expected_more_time:
            LOGGER.info('[TERMINATE] not enough time to train (remain:%f need:%f)', remaining_time_budget, expected_more_time)
            self.base_info['terminate'] = True
            self.model.info['terminate'] = True
            self.done_training = True
            return True

        ## learning lr
        if self.ensembleconfig.MODEL_INDEX >= len(self.ensembleconfig.MAX_TIMES_LIST):
            return True

        if min(self.ensemble_lr) < self.model.hyper_params['conditions']['min_lr']:
            LOGGER.info('[TERMINATE] lr=%f', min(self.ensemble_lr))
            done = True if self.base_info['terminate'] else False
            self.base_info['terminate'] = True
            self.model.info['terminate'] = True
            self.done_training = done
            return True

        if max(self.ensemble_epoch) >= 20 and \
            inner_epoch > self.base_hyper_params['dataset']['max_epoch'] * self.base_hyper_params['conditions']['max_inner_loop_ratio']:
            LOGGER.info('[TERMINATE] cannot found best model in too many epoch')
            done = True if self.base_info['terminate'] else False
            self.base_info['terminate'] = True
            self.model.info['terminate'] = True
            self.done_training = done
            return True
        
        # if self.ensembleconfig.MODEL_INDEX == len(self.ensembleconfig.MAX_TIMES_LIST)-1:
        #     valid_sub_scores = [(c['valid']['score']) for c in self.model.checkpoints][-8:]
        #     valid_abs_scores = np.mean(np.array(valid_sub_scores))>0.997
        #     self.done_training = valid_abs_scores
        #     return True

        return False

    def get_total_time(self):
        return sum([self.timers[key].total_time for key in self.timers.keys()])

    def train(self, dataset, remaining_time_budget=None):
        if self.change_next_model == True:
            self.release_freeze = False
            self.model = self.model_space[self.ensembleconfig.MODEL_INDEX].to(device=self.device, non_blocking=True)
            self.model_pred = self.model_space[self.ensembleconfig.MODEL_INDEX].to(device=self.device, non_blocking=True).eval()
            self.datasets.clear()
            self.dataloaders.clear()
            self.change_next_model = False
        
        LOGGER.debug(self)
        LOGGER.debug('[train] [%02d] budget:%f', self.model.info['loop']['epoch'], remaining_time_budget)
        self.timers['train']('outer_start', exclude_total=True, reset_step=True)
        self.model.hyper_params['dataset']['max_times'] = self.ensembleconfig.MAX_TIMES_LIST[self.ensembleconfig.MODEL_INDEX]
        self.model.hyper_params['conditions']['threshold_valid_score_diff'] = self.ensembleconfig.THRESHOLD_DIFF_LIST[self.ensembleconfig.MODEL_INDEX]

        train_dataloader = self.build_or_get_train_dataloader(dataset)
        if self.model.info['condition']['first']['train']:
            self.update_model()
            LOGGER.info(self)
        self.timers['train']('build_dataset')

        inner_epoch = 0
        while True:
            inner_epoch += 1
            remaining_time_budget -= self.timers['train'].step_time

            self.timers['train']('start', reset_step=True)
            train_metrics = self.epoch_train(self.model.info['loop']['epoch'], train_dataloader)
            self.timers['train']('train')

            train_score = np.min([c['train']['score'] for c in self.model.checkpoints[-20:] + [{'train': train_metrics}]])
            if train_score > self.model.hyper_params['conditions']['skip_valid_score_threshold'] or \
                self.model.info['loop']['test'] >= self.base_hyper_params['conditions']['skip_valid_after_test']:
                self.is_skip_valid = False
                is_first = self.model.info['condition']['first']['valid']
                valid_dataloader = self.build_or_get_dataloader('valid', self.datasets['valid'], self.datasets['num_valids'])
                self.timers['train']('valid_dataset', exclude_step=is_first)
                valid_metrics = self.epoch_valid(self.model.info['loop']['epoch'], valid_dataloader)
                
            else:
                self.is_skip_valid = True
                valid_metrics = self.skip_valid(self.model)
            self.timers['train']('valid')

            if self.best_loss >= valid_metrics['loss']:
                self.best_loss = valid_metrics['loss']

            metrics = {
                'epoch': self.model.info['loop']['epoch'],
                'model': self.model.state_dict().copy(),
                'train': train_metrics,
                'valid': valid_metrics,
            }

            self.epoch_metrics = metrics
            self.update_condition(metrics)
            self.timers['train']('adapt', exclude_step=True)

            LOGGER.info(
                '[train] [%02d] time(budge:%.2f, total:%.2f, step:%.2f) loss:(train:%.3f, valid:%.3f) score:(train:%.3f valid:%.3f) lr:%f num_steps:%d',
                self.model.info['loop']['epoch'], remaining_time_budget, self.get_total_time(), self.timers['train'].step_time,
                metrics['train']['loss'], metrics['valid']['loss'], metrics['train']['score'], metrics['valid']['score'],
                self.model.optimizer.get_learning_rate(), metrics['train']['run_steps']
            )
            LOGGER.debug('[train] [%02d] Timer:%s', self.model.info['loop']['epoch'], self.timers['train'])

            self.model.hyper_params['dataset']['max_epoch'] = self.model.info['loop']['epoch'] + remaining_time_budget // self.timers['train'].step_time
            LOGGER.info('[ESTIMATE] max_epoch: %d', self.model.hyper_params['dataset']['max_epoch'])

            if self.break_train_loop_condition(remaining_time_budget, inner_epoch):
                break

            self.timers['train']('end')

        remaining_time_budget -= self.timers['train'].step_time
        
        if self.release_freeze== False:
            self.release_model(remaining_time_budget)

        self.decide_change_next_model(remaining_time_budget, inner_epoch)
        self.terminate_train_loop_condition(remaining_time_budget, inner_epoch)

        if not self.done_training:
            self.adapt(remaining_time_budget)

        self.timers['train']('outer_end')
        LOGGER.info(
            '[train] [%02d] time(budge:%.2f, total:%.2f, step:%.2f) loss:(train:%.3f, valid:%.3f) score:(train:%.3f valid:%.3f) lr:%f',
            self.model.info['loop']['epoch'], remaining_time_budget, self.get_total_time(), self.timers['train'].step_time,
            metrics['train']['loss'], metrics['valid']['loss'], metrics['train']['score'], metrics['valid']['score'],
            self.model.optimizer.get_learning_rate()
        )

    def test(self, dataset, remaining_time_budget=None):
        if (self.ensembleconfig.MODEL_INDEX == 0) and (self.model == self.model_space[self.ensembleconfig.MODEL_INDEX]):
            rv=self.base_test(dataset,self.model,self.model.checkpoints,remaining_time_budget=None)
        else:
            self.base_test(dataset,self.model,self.model.checkpoints,remaining_time_budget=None)
            rv=self.ensemble_prediction()
        LOGGER.info('Ensemble Score Length [%02d] ******** Starting Ensemble Model Num [%02d]',len(self.model.ensemble_scores),self.ensembleconfig.MODEL_INDEX)
        self.history_preds = rv
        return rv

    def base_test(self, dataset, model, checkpoints, remaining_time_budget=None):
        self.model.ensemble_test_index += 1

        self.timers['test']('start', exclude_total=True, reset_step=True)
        is_first = self.model.info['condition']['first']['test']
        self.model.info['loop']['test'] += 1

        dataloader = self.build_or_get_dataloader('test', dataset, self.num_test)
        self.timers['test']('build_dataset', reset_step=is_first)
        rv = self.prediction(dataloader, model=model,checkpoints=model.checkpoints)
        self.timers['test']('end')

        scores = [c['valid']['score'] for c in self.model.checkpoints]
        diff = (max(scores) - min(scores)) * (1 - max(scores))
        if self.best_score <= max(scores):
            self.best_score = max(scores)
        
        if (self.ensembleconfig.MODEL_INDEX == 0):
            self.model.ensemble_scores[self.model.ensemble_test_index] = {
                'loss':self.epoch_metrics['valid']['loss'],
                'score':self.epoch_metrics['valid']['score']
            }
            self.model.ensemble_predictions[self.model.ensemble_test_index]={
                'predictions':rv
            }
        elif self.epoch_metrics['valid']['score'] >= 0.99 * self.best_score and self.epoch_metrics['valid']['loss'] <= self.best_loss:
            self.model.ensemble_scores[self.model.ensemble_test_index] = {
                'loss':self.epoch_metrics['valid']['loss'],
                'score':self.epoch_metrics['valid']['score']
            }
            self.model.ensemble_predictions[self.model.ensemble_test_index]={
                'predictions':rv
            }
        return rv

    def get_top_players(self, data, sort_keys, reverse=True, n=2, order=True):
        top = sorted(data.items(), key=lambda x: x[1][sort_keys], reverse=reverse)[:n]
        if order:
            return OrderedDict(top)
        return dict(top)

    def ensemble_prediction(self):
        """
        get best val loss round id list, best val acc round id list, and mean to prevent overfitting.
        :return:
        """
        preds_dict = list()
        for _,item_model in enumerate(self.model_space[:self.ensembleconfig.MODEL_INDEX+1]):
            if len(item_model.ensemble_predictions)>=1:
                preds_dict.append(self.ensemble_single_prediction(item_model.ensemble_predictions,item_model.ensemble_scores))

        if len(preds_dict) == 0:
            return self.history_preds
        elif len(preds_dict) == 1:
            return preds_dict[0]
        else:
            for key,logits in enumerate(preds_dict):
                logits = (logits - np.min(logits))/(np.max(logits)-np.min(logits))
                if key ==0:
                    logits = logits
                else:
                    logits += logits
            return logits/(key+1)

    def ensemble_single_prediction(self,predict_prob_list,g_his_eval_dict):
        """
        get best val loss round id list, best val acc round id list, and mean to prevent overfitting.
        :return:
        """
        key_t_loss = "loss"
        key_t_acc = "score"
        key_loss = key_t_loss
        key_acc = key_t_acc

        topn_vloss = self.ensembleconfig.ENS_TOP_VLOSS_NUM
        topn_vacc = self.ensembleconfig.ENS_TOP_VACC_NUM

        pre_en_eval_rounds = list(predict_prob_list.keys())
        
        cur_eval_dict = {k: g_his_eval_dict.get(k) for k in pre_en_eval_rounds}
        top_n_val_acc_evals = self.get_top_players(data=cur_eval_dict, sort_keys=key_acc, n=topn_vacc, reverse=True)
        top_n_val_acc_evals = list(top_n_val_acc_evals.items())
        topn_valacc_roundidx = [a[0] for a in top_n_val_acc_evals]

        top_n_val_loss_evals = self.get_top_players(data=cur_eval_dict, sort_keys=key_loss, n=topn_vloss,reverse=False)
        top_n_val_loss_evals = list(top_n_val_loss_evals.items())
        topn_valloss_roundidx = [a[0] for a in top_n_val_loss_evals]
        
        
        merge_roundids = list()
        merge_roundids.extend(topn_valloss_roundidx)
        merge_roundids.extend(topn_valacc_roundidx)

        merge_preds_res = [predict_prob_list[roundid] for roundid in merge_roundids]
        if len(merge_roundids) == 1:
            return merge_preds_res[0]['predictions']
        else:
            for key,items in enumerate(merge_preds_res):
                logits = items['predictions']
                if key ==0:
                    logits = logits
                else:
                    logits += logits
            return logits/(key+1)