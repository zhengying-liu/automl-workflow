# coding:utf-8
import numpy as np
from at_toolkit import info
from at_speech.data_space.raw_data_space import RawDataNpDb
from at_speech.data_space.feats_engine import (
    AbsFeatsMaker,
    KapreMelSpectroGramFeatsMaker,
    LibrosaMelSpectroGramFeatsMaker,
    LbrsTr34FeatsMaker,
)
from at_speech.at_speech_cons import *


class FeatsDataDb:
    def __init__(self, raw_train_num, raw_test_num):
        self.raw_train_num = raw_train_num
        self.raw_test_num = raw_test_num
        self.raw_train_feats_data_tables = dict()
        self.raw_test_feats_data_tables = dict()
        self.split_val_feats_data_tables = dict()
        self.raw_feat_makers_table = dict()
        self.split_val_num = None
        self.raw_data_db = RawDataNpDb(self.raw_train_num, self.raw_test_num)

        # init all feats_makers, and add to register table.
        self.kapre_melspecgram_featmaker = KapreMelSpectroGramFeatsMaker("KAPRE", FEAT_KAPRE_MELSPECGRAM)
        self.lbs_melspecgram_featmaker = LibrosaMelSpectroGramFeatsMaker("LIBROSA", FEAT_LBS_MELSPECGRAM)
        self.lbs_tr34_featmaker = LbrsTr34FeatsMaker("LIBROSA", FEAT_LBS_TR34)

        self.add_feats_data_table(FEAT_KAPRE_MELSPECGRAM, self.kapre_melspecgram_featmaker)
        self.add_feats_data_table(FEAT_LBS_MELSPECGRAM, self.lbs_melspecgram_featmaker)
        self.add_feats_data_table(FEAT_LBS_TR34, self.lbs_tr34_featmaker)

        # status, False means raw_test_table is None, need to be made.
        self.raw_test_feats_status_table = {
            FEAT_KAPRE_MELSPECGRAM: False,
            FEAT_LBS_MELSPECGRAM: False,
            FEAT_LBS_TR34: False,
        }
        # status, False means raw_test_table is None, need to be made.
        self.split_val_feats_status_table = {
            FEAT_KAPRE_MELSPECGRAM: False,
            FEAT_LBS_MELSPECGRAM: False,
            FEAT_LBS_TR34: False,
        }

    def add_feats_data_table(self, feat_name, feats_maker: AbsFeatsMaker):
        if feat_name not in self.raw_train_feats_data_tables.keys():
            self.raw_train_feats_data_tables[feat_name] = np.array([None] * self.raw_train_num)
            self.raw_test_feats_data_tables[feat_name] = np.array([None] * self.raw_test_num)
            self.split_val_feats_data_tables[feat_name] = np.array([None] * self.raw_train_num) # fixme: need to modify to split_val_num.
            self.raw_feat_makers_table[feat_name] = feats_maker

    def put_raw_test_feats(self, feat_name, raw_test_feats_np):
        assert feat_name in self.raw_test_feats_data_tables.keys(), "feat_name {} not exists in db".format(feat_name)
        assert (
            len(raw_test_feats_np) == self.raw_test_num
        ), "Error, raw_test_num={}, but len raw_test_feats_np={}".format(self.raw_test_num, len(raw_test_feats_np))
        self.raw_test_feats_data_tables[feat_name] = raw_test_feats_np

    def get_raw_test_feats(self, feat_name, feats_maker_params: dict = None, forced=False):
        """
        if None, use feats_maker to gen and write.
        then read.
        :param feat_name:
        :param forced: if force to regenerate test features, classifiers can force to use augmented features for test data.
        :return:
        """
        info("Test: feat_name={}. feats_make_params={}, forced={}".format(feat_name, feats_maker_params, forced))
        if self.raw_test_feats_status_table.get(feat_name) is False or forced is True:
            raw_test_feats_np = self.raw_feat_makers_table.get(feat_name).make_features(
                self.raw_data_db.raw_test_x_np_table, feats_maker_params
            )
            self.put_raw_test_feats(feat_name, raw_test_feats_np)
            self.raw_test_feats_status_table[feat_name] = True

        return self.raw_test_feats_data_tables.get(feat_name)

    def get_raw_train_feats(self, feat_name, raw_train_idxs, feats_maker_params: dict = None, forced=False):
        """
        1. check need_make_feats_idxs
        2. get raw_data by make_feats_idx
        3. make_feats(feats_maker, raw_data_np)
        4. write_back(update_featss
        5. read updated feats.
        :param feat_name:
        :param raw_train_idxs:
        :param feats_maker_params:
        :param forced: if True, re-generate and write back to feat_np_table.
        :return:
        """
        # check if is None first, if is None, using feats_maker
        need_make_feats_idxs = list()
        if forced:
            # clear feat_name table.
            self.raw_train_feats_data_tables[feat_name] = np.array([None] * self.raw_train_num)
            need_make_feats_idxs = raw_train_idxs
        else:
            for raw_train_idx in raw_train_idxs:
                if self.raw_train_feats_data_tables.get(feat_name)[raw_train_idx] is None:
                    need_make_feats_idxs.append(raw_train_idx)

        info(
            "if_forced={}, feat_name={}, need_make_feats_idx len={}, content={}".format(
                forced, feat_name, len(need_make_feats_idxs), need_make_feats_idxs[:5]
            )
        )
        # fixme: check length.
        if len(need_make_feats_idxs) > 0:
            # 2. get raw data
            need_make_feats_rawdata = self.raw_data_db.raw_train_x_np_table[need_make_feats_idxs]

            # 3. make_feats
            make_feats_done = self.raw_feat_makers_table.get(feat_name).make_features(
                need_make_feats_rawdata, feats_maker_params
            )
            make_feats_done = np.array(make_feats_done)
            info("make_feats_done, type={}, shape={}".format(type(make_feats_done), make_feats_done.shape))

            # 4. write back to feat_table
            for i in range(len(need_make_feats_idxs)):
                self.raw_train_feats_data_tables.get(feat_name)[need_make_feats_idxs[i]] = make_feats_done[i]

        # 5. read from updated feat_table.
        cur_train_feats = [self.raw_train_feats_data_tables.get(feat_name)[i].shape for i in raw_train_idxs]
        info("cur_train_feats, shape_list={}".format(cur_train_feats[:3]))
        return np.stack(self.raw_train_feats_data_tables.get(feat_name)[raw_train_idxs])

    def get_raw_train_y(self, raw_train_idxs):
        return np.stack(self.raw_data_db.raw_train_y_np_table[raw_train_idxs])

    def get_split_val_feats(self, feat_name:str, split_val_idxs:list, feats_maker_params: dict = None, forced=False):
        """
        if None, use feats_maker to gen and write.
        then read.
        :param feat_name:
        :param forced: if force to regenerate test features, classifiers can force to use augmented features for test data.
        :return:
        """
        if self.split_val_num is None:
            self.split_val_num = self.raw_data_db.split_val_sample_num

        if self.split_val_feats_status_table.get(feat_name) is False or forced is True:
            # 1. clear old.
            self.split_val_feats_data_tables[feat_name] = np.array([None] * self.split_val_num)
            # 2. get raw data
            need_make_feats_rawdata = self.raw_data_db.raw_train_x_np_table[split_val_idxs]
            # 3. use feats params to make TEST feats
            make_feats_done = self.raw_feat_makers_table.get(feat_name).make_features(
                need_make_feats_rawdata, feats_maker_params
            )
            make_feats_done = np.array(make_feats_done)
            # 3. put it into val_feats_table.
            assert (
                    len(make_feats_done) == self.split_val_num
            ), "Error, split_val_num={}, but len split_val_feats={}".format(self.split_val_num, len(make_feats_done))
            self.split_val_feats_data_tables[feat_name] = make_feats_done

            self.split_val_feats_status_table[feat_name] = True
            info("Note: split_val_feats new updated, feat_num={}".format(feat_name))

        return self.split_val_feats_data_tables.get(feat_name)


