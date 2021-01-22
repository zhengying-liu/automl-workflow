# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os

os.system(
    "pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali && pip3 install torch torchvision")

import threading
import random

import tensorflow as tf
import torch
import torchvision as tv
import numpy as np

import skeleton
from architectures.mc3 import ResNet as MC3
from architectures.mc3_2 import ResNet as MC3_2
from architectures.r2plus1d import ResNet as R2PLUS1D
from architectures.r3d import ResNet as R3D
from architectures.resnet import ResNet18
from architectures.resnet_34 import ResNet34
# from architectures.resnet_3d import ResNet3d
from architectures.resnet_50 import ResNet50
from architectures.resnet_101 import ResNet101
from skeleton.projects import LogicModel, get_logger
from skeleton.projects.others import NBAC, AUC
import time
from collections import OrderedDict
from skeleton.utils.log_utils import timeit

import pdb

torch.backends.cudnn.benchmark = True
threads = [
    threading.Thread(target=lambda: torch.cuda.synchronize()),
    threading.Thread(target=lambda: tf.Session())
]
[t.start() for t in threads]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

LOGGER = get_logger(__name__)

def set_random_seed_all(seed, deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_random_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_top_players(data, sort_keys, reverse=True, n=2, order=True):
    top = sorted(data.items(), key=lambda x: x[1][sort_keys], reverse=reverse)[:n]
    if order:
        return OrderedDict(top)
    return dict(top)

class Model(LogicModel):
    def __init__(self, metadata):
        set_random_seed_all(0xC0FFEE,True)
        super(Model, self).__init__(metadata)

    def build(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        in_channels = self.base_info['dataset']['shape'][-1]
        num_class = self.base_info['dataset']['num_class']
        # torch.cuda.synchronize()

        LOGGER.info('[init] session')
        [t.join() for t in threads]

        self.device = torch.device('cuda', 0)
        self.session = tf.Session()

        LOGGER.info('[init] Model')

        ## ResNet 3-times
        ## ResNet18 8-times
        ## ResNet18 8-times
        ## ResNet34 8-times
        ## ResNet50 8-times
        # model_space = [ResNet,ResNet18,ResNet,ResNet34,ResNet50]
        model_space = [MC3,MC3,MC3,MC3]
        self.model_space =[]
        self.ensembleconfig.MODEL_LENGTH = len(self.model_space)
        for key,model in enumerate(model_space):
            if model in [MC3,MC3_2,R3D,R2PLUS1D,ResNet18]:
                Network = model(in_channels, num_class)
                model_path = os.path.join(base_dir, 'models')
                LOGGER.info('model path: %s', model_path)
                Network.init(model_dir=model_path, gain=1.0)
            else:
                Network = model(in_channels, num_class)
                Network.init(model_dir=model_path, gain=1.0)
            self.model_space.append(Network)

        self.model = self.model_space[self.ensembleconfig.MODEL_INDEX]
        self.model_pred = self.model_space[self.ensembleconfig.MODEL_INDEX].eval()

        LOGGER.info('[init] copy to device')
        self.model = self.model.to(device=self.device, non_blocking=True)  # .half()
        self.model_pred = self.model_pred.to(device=self.device, non_blocking=True)  # .half()self。
        LOGGER.info('[init] done.')

    def good_to_predict(self):
        flag = ((self.model.round_idx < 6)
            or (self.model.round_idx < 21 and self.model.round_idx % 2 == 1)
            or (self.model.round_idx - self.model.last_y_pred_round > 3)
        )

        # best_idx = np.argmax(np.array([c['valid']['score'] for c in self.model.checkpoints]))
        # best_score = self.model.checkpoints[best_idx]['valid']['score']
        # last_two_score = self.model.checkpoints[-2]['valid']['score']
        # last_one_score = self.model.checkpoints[-1]['valid']['scare']
        # diff = last_one_score-last_two_score 
        # if diff <= 0.0:
        #     flag = True
        return flag

    def update_model(self):
        self.is_half = self.model._half
        num_class = self.base_info['dataset']['num_class']

        epsilon = min(0.1, max(0.001, 0.001 * pow(num_class / 10, 2)))
        if self.is_multiclass():
            self.model.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
            self.tau = 8.0
            LOGGER.info('[update_model] %s (tau:%f, epsilon:%f)', self.model.loss_fn.__class__.__name__, self.tau,
                        epsilon)
        else:
            self.model.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            self.tau = 8.0
            LOGGER.info('[update_model] %s (tau:%f, epsilon:%f)', self.model.loss_fn.__class__.__name__, self.tau,
                        epsilon)
        self.model_pred.loss_fn = self.model.loss_fn

        # if self.is_video():
        # not use fast auto aug
        self.model.hyper_params['conditions']['use_fast_auto_aug'] = False
        times = self.model.hyper_params['dataset']['input'][0]
        self.model.set_video(times=times)
        self.model_pred.set_video(times=times)

        self.init_opt()
        LOGGER.info('[update] done.')

    def init_opt(self):
        steps_per_epoch = self.model.hyper_params['dataset']['steps_per_epoch']
        batch_size = self.model.hyper_params['dataset']['batch_size']
        init_lr = self.model.hyper_params['optimizer']['lr']
        warmup_multiplier = self.model.hyper_params['optimizer']['warmup_multiplier']
        warmup_epoch = self.model.hyper_params['optimizer']['warmup_epoch']
        self.model.init_opt(steps_per_epoch,batch_size,init_lr,warmup_multiplier,warmup_epoch)
        LOGGER.info('[optimizer] %s (batch_size:%d)', self.model.optimizer._optimizer.__class__.__name__, batch_size)

    @timeit
    def adapt(self, remaining_time_budget=None):
        epoch = self.model.info['loop']['epoch']
        input_shape = self.model.hyper_params['dataset']['input']
        height, width = input_shape[:2]
        batch_size = self.model.hyper_params['dataset']['batch_size']

        train_score = np.average([c['train']['score'] for c in self.model.checkpoints[-5:]])
        valid_score = np.average([c['valid']['score'] for c in self.model.checkpoints[-5:]])
        LOGGER.info('[adapt] [%04d/%04d] train:%.3f valid:%.3f',
                    epoch, self.model.hyper_params['dataset']['max_epoch'],
                    train_score, valid_score)

        self.model.use_test_time_augmentation = self.model.info['loop']['test'] > 1

        # print('use_test_time_augmentation :', self.model.use_test_time_augmentation)

        if self.model.hyper_params['conditions']['use_fast_auto_aug']:
            self.model.hyper_params['conditions']['use_fast_auto_aug'] = valid_score < 0.995

        # Adapt Apply Fast auto aug
        if self.model.hyper_params['conditions']['use_fast_auto_aug'] and \
                (train_score > 0.995 or self.model.info['terminate']) and \
                remaining_time_budget > 120 and \
                valid_score > 0.01 and \
                self.dataloaders['valid'] is not None and \
                not self.model.update_transforms:
            LOGGER.info('[adapt] search fast auto aug policy')
            self.update_transforms = True

    def activation(self, logits):
        if self.is_multiclass():
            logits = torch.sigmoid(logits)
            prediction = (logits > 0.5).to(logits.dtype)
        else:
            logits = torch.softmax(logits, dim=-1)
            _, k = logits.max(-1)
            prediction = torch.zeros(logits.shape, dtype=logits.dtype, device=logits.device).scatter_(-1, k.view(-1, 1),
                                                                                                      1.0)
        return logits, prediction
    
    def decision_if_single_ensemble(self,model):
        pre_ens_round_start = 3
        if model.round_idx > pre_ens_round_start:
            model.single_ensemble = True
            return model.single_ensemble
        else:
            model.single_ensemble = False
            return model.single_ensemble

    @timeit
    def epoch_train(self, epoch, train, model=None):
        model = model if model is not None else self.model
        model.round_idx += 1
        if epoch < 0:
            optimizer = model.optimizer_fc
        else:
            optimizer = model.optimizer

        model.train()
        model.zero_grad()
        if model.info['condition']['first']['train']:
            num_steps = 10000
        else:
            num_steps = model.hyper_params['dataset']['steps_per_epoch']
        # train
        metrics = []
        scores = []
        score = 0
        step = 0

        for step, (examples, labels) in zip(range(num_steps), train):
            if examples.shape[0] == 1:
                examples = examples[0]
                labels = labels[0]
            original_labels = labels
            if not self.is_multiclass():
                labels = labels.argmax(dim=-1)
            # LOGGER.info('*'*30+str(self.is_multiclass()) + '*'*30)
            skeleton.nn.MoveToHook.to((examples, labels), self.device, self.is_half)
            logits, loss = model(examples, labels, tau=self.tau, reduction='avg')
            loss = loss.sum()
            loss.backward()

            max_epoch =model.hyper_params['dataset']['max_epoch']
            optimizer.update(maximum_epoch=max_epoch)
            optimizer.step()
            model.zero_grad()
            if model.info['condition']['first']['train']:
                logits, prediction = self.activation(logits.float())
                score = AUC(logits, original_labels.float())
                scores.append(score)
                if step > 10 and sum(scores[-10:]) > 2.:
                    break
        
            if step == num_steps - 1:
                logits, prediction = self.activation(logits.float())
                score = AUC(logits, original_labels.float())

            metrics.append({
                'loss': loss.detach().float().cpu(),
                'score': 0,
            })

        train_loss = np.average([m['loss'] for m in metrics])
        train_score = score
        optimizer.update(train_loss=train_loss)

        self.train_loss = train_loss
        self.train_score = train_score

        return {
            'loss': train_loss,
            'score': train_score,
            'run_steps': step
        }

    @timeit
    def epoch_valid(self, epoch, valid, model=None,reduction='avg'):
        test_time_augmentation = False
        model = model if model is not None else self.model
        model.eval()
        num_steps = len(valid)
        metrics = []
        tau = self.tau

        with torch.no_grad():
            for step, (examples, labels) in zip(range(num_steps), valid):
                original_labels = labels
                if not self.is_multiclass():
                    labels = labels.argmax(dim=-1)
                batch_size = examples.size(0)

                # Test-Time Augment flip
                if model.use_test_time_augmentation and test_time_augmentation:
                    examples = torch.cat([examples, torch.flip(examples, dims=[-1])], dim=0)
                    labels = torch.cat([labels, labels], dim=0)

                skeleton.nn.MoveToHook.to((examples, labels), self.device, self.is_half)
                logits, loss = model(examples, labels, tau=tau, reduction=reduction)

                # avergae
                if model.use_test_time_augmentation and test_time_augmentation:
                    logits1, logits2 = torch.split(logits, batch_size, dim=0)
                    logits = (logits1 + logits2) / 2.0

                logits, prediction = self.activation(logits.float())
                tpr, tnr, nbac = NBAC(prediction, original_labels.float())
                if reduction == 'avg':
                    auc = AUC(logits, original_labels.float())
                else:
                    auc = max([AUC(logits[i:i + 16], original_labels[i:i + 16].float()) for i in
                               range(int(len(logits)) // 16)])

                score = auc if model.hyper_params['conditions']['score_type'] == 'auc' else float(nbac.detach().float())
                metrics.append({
                    'loss': loss.detach().float().cpu(),
                    'score': score,
                })

                LOGGER.debug(
                    '[valid] [%02d] [%03d/%03d] loss:%.6f AUC:%.3f NBAC:%.3f tpr:%.3f tnr:%.3f, lr:%.8f',
                    epoch, step, num_steps, loss, auc, nbac, tpr, tnr,
                    model.optimizer.get_learning_rate()
                )
            
            if reduction == 'avg':
                valid_loss = np.average([m['loss'] for m in metrics])
                valid_score = np.average([m['score'] for m in metrics])
            elif reduction in ['min', 'max']:
                valid_loss = np.min([m['loss'] for m in metrics])
                valid_score = np.max([m['score'] for m in metrics])
            else:
                raise Exception('not support reduction method: %s' % reduction)
            
        model.optimizer.update(valid_loss=np.average(valid_loss))

        self.valid_loss = valid_loss
        self.valid_score = valid_score
        
        model.g_his_eval_dict[model.round_idx] = {
            "t_loss": self.train_loss,
            "t_acc": self.train_score,
            "v_loss": self.valid_loss,
            "v_acc": self.valid_score
        }

        return {
            'loss': valid_loss,
            'score': valid_score,
        }

    def skip_valid(self, model):

        model.g_his_eval_dict[model.round_idx] = {
            "t_loss": self.train_loss,
            "t_acc": self.train_score,
            "v_loss": 99.9,
            "v_acc": model.info['loop']['epoch'] * 1e-4
        }

        return {
            'loss': 99.9,
            'score': model.info['loop']['epoch'] * 1e-4,
        }

    @timeit
    def prediction(self, dataloader, model, checkpoints,test_time_augmentation=True, detach=True, num_step=None):
        tau = self.tau
        best_idx = np.argmax(np.array([c['valid']['score'] for c in checkpoints]))
        best_loss = checkpoints[best_idx]['valid']['loss']
        best_score = checkpoints[best_idx]['valid']['score']

        states = checkpoints[best_idx]['model']
        model.load_state_dict(states)
        LOGGER.info('best checkpoints at %d/%d (valid loss:%f score:%f) tau:%f',
                    best_idx + 1, len(checkpoints), best_loss, best_score, tau)

        num_step = len(dataloader) if num_step is None else num_step
        model.eval()
        
        with torch.no_grad():
            predictions = []
            for step, (examples, labels) in enumerate(dataloader):

                batch_size = examples.size(0)
                if model.use_test_time_augmentation and test_time_augmentation:
                    examples = torch.cat([examples, torch.flip(examples, dims=[-1])], dim=0)

                # skeleton.nn.MoveToHook.to((examples, labels), self.device, self.is_half)
                logits = model(examples, tau=tau)

                # avergae
                if model.use_test_time_augmentation and test_time_augmentation:
                    logits1, logits2 = torch.split(logits, batch_size, dim=0)
                    logits = (logits1 + logits2) / 2.0

                logits, prediction = self.activation(logits)

                if detach:
                    predictions.append(logits.detach().float().cpu().numpy())
                else:
                    predictions.append(logits)

            if detach:
                predictions = np.concatenate(predictions, axis=0).astype(np.float)
            else:
                predictions = torch.cat(predictions, dim=0)


        # if self.good_to_predict():
        #     if self.ensembleconfig.ENABLE_PRE_ENSE and self.decision_if_single_ensemble(model):
        #         model.predict_prob_list[model.round_idx - 1] = predictions
        #         y_pred = self.test_pred_ensemble(model.predict_prob_list,model.g_his_eval_dict)
        #     else:
        #         y_pred = predictions
            
        #     model.last_y_pred_round = model.round_idx
        #     predictions =  y_pred
        # else:
        #     predictions = predictions

        return predictions

    @timeit
    def test_pred_ensemble(self,predict_prob_list,g_his_eval_dict):
        """
        get best val loss round id list, best val acc round id list, and mean to prevent overfitting.
        :return:
        """
        key_t_loss = "v_loss"
        key_t_acc = "v_acc"
        key_loss = key_t_loss
        key_acc = key_t_acc

        topn_vloss = self.ensembleconfig.ENS_TOP_VLOSS_NUM
        topn_vacc = self.ensembleconfig.ENS_TOP_VACC_NUM
        pre_en_eval_rounds = list(predict_prob_list.keys())

        cur_eval_dict = {k: g_his_eval_dict.get(k) for k in pre_en_eval_rounds}

        top_n_val_loss_evals = get_top_players(data=cur_eval_dict, sort_keys=key_loss, n=topn_vloss,
                                               reverse=False)
        top_n_val_acc_evals = get_top_players(data=cur_eval_dict, sort_keys=key_acc, n=topn_vacc, reverse=True)
        top_n_val_loss_evals = list(top_n_val_loss_evals.items())
        top_n_val_acc_evals = list(top_n_val_acc_evals.items())
        topn_valloss_roundidx = [a[0] for a in top_n_val_loss_evals]
        topn_valacc_roundidx = [a[0] for a in top_n_val_acc_evals]

        merge_roundids = list()
        merge_roundids.extend(topn_valloss_roundidx)
        merge_roundids.extend(topn_valacc_roundidx)

        merge_preds_res = [predict_prob_list[roundid] for roundid in merge_roundids]
        
        if len(merge_roundids) == 1:
            return merge_preds_res[0]
        else:
            for key,logits in enumerate(merge_preds_res):
                if key ==0:
                    logits = merge_preds_res[key]
                else:
                    logits += merge_preds_res[key]
                    # pdb.set_trace()
            return logits/len(merge_preds_res)
        ## 根据var的大小进行 ensemble 
        ## 加权求 ensemble