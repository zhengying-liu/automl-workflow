import os, sys
import numpy as np

here = os.path.abspath(os.path.dirname(__file__))
CODE_SUB_DIR = os.path.abspath(os.path.join(here, "..", ".."))
print(CODE_SUB_DIR)
sys.path.append(CODE_SUB_DIR)


from at_speech import SLLRLiblinear, SLLRSag, CNNClassifier, ThinResnet34Classifier
from at_speech.data_space.raw_data_space import RawDataNpDb
from at_toolkit.at_sampler import AutoSamplerPasa, AutoSpSamplerNew, AutoValidSplitor, minisamples_edaer, sample_y_edaer
from at_speech.data_space.feats_data_space import FeatsDataDb
from at_speech.policy_space.decision_making import DecisionMaker
from at_toolkit.at_tfds_convertor import TfdsConvertor
from at_toolkit import AdlClassifier, AdlSpeechDMetadata
from at_toolkit.at_evalator import  ATEvaluator
from at_toolkit.at_utils import info, as_timer
from at_speech.at_speech_cons import *
from at_speech.at_speech_config import TFDS2NP_TAKESIZE_RATION_LIST, TR34_TRAINPIP_WARMUP, IF_VAL_ON, Tr34SamplerHpParams


"""
    1. init all classifiers.
"""


CLS_REG_TABLE = {
    CLS_LR_LIBLINEAER: SLLRLiblinear,
    CLS_LR_SAG: SLLRSag,
    CLS_CNN: CNNClassifier,
    CLS_TR34: ThinResnet34Classifier,
}

CLS_2_FEATNAME_REG_TABLE = {
    CLS_LR_LIBLINEAER: FEAT_KAPRE_MELSPECGRAM,
    CLS_LR_SAG: FEAT_KAPRE_MELSPECGRAM,
    CLS_TR34: FEAT_LBS_TR34,
}


class MetaClsHPParams:
    lr_sag_cls_init_params = {"max_iter": 50}  # 50/100


class ModelExecutor:
    def __init__(self, ds_metadata):
        """
        :param ds_metadata:
            {
                'train_num': train_num,
                'test_num': test_num,
                'class_num': class_num,
                "domain": "speech"
            }
        """
        self.class_num = ds_metadata.get("class_num")
        self.train_num = ds_metadata.get("train_num")
        self.test_num = ds_metadata.get("test_num")
        self.aspeech_metadata = AdlSpeechDMetadata(ds_metadata)

        self.cls_tpye_libs = [CLS_LR_LIBLINEAER, CLS_LR_SAG, CLS_TR34]
        # 1. cls
        self.lr_libl_cls = None
        self.lr_sag_cls = None
        self.tr34_cls = None
        self.tr34_cls_train_pip_run = 0
        # 2. raw_data_np_db, config.
        # self.tfds_convertor = TfdsConvertor(if_train_shuffle=True, train_shuffle_size=self.train_num)
        self.tfds_convertor = TfdsConvertor()
        # 3. dataspace
        self.feats_data_db = FeatsDataDb(self.train_num, self.test_num)
        # self.raw_data_np_db = RawDataNpDb(self.train_num, self.test_num)

        self.init_pipeline()

        self.train_pip_id = 0
        self.test_pip_id = 0

        self.token_train_size = 0

        self.cur_cls_ins_table = {
            CLS_LR_LIBLINEAER: self.lr_libl_cls,
            CLS_LR_SAG: self.lr_sag_cls,
            CLS_TR34: self.tr34_cls,
        }

        self.decision_maker = DecisionMaker(self.aspeech_metadata)
        self.cur_cls = None
        self.cur_sampler = None
        # for val
        self.val_sample_idxs = list()
        self.cur_val_examples_y = None
        # self.cur_val_auc = None
        self.cur_val_nauc = None
        self.cur_train_his_report = dict()

        # for minis_eda.
        self.minis_eda_report = None
        self.is_multilabel = False

        # init sampler.
        self.lr_sampler = AutoSamplerPasa(self.class_num)
        self.tr34_sampler = AutoSpSamplerNew(None)
        self.val_splitor = AutoValidSplitor(self.class_num)

        self.cur_sampler_table = {
            CLS_LR_LIBLINEAER: self.lr_sampler,
            CLS_LR_SAG: self.lr_sampler,
            CLS_TR34: self.tr34_sampler,
        }

        # for meta hyper params.
        self.tfds2np_take_size_array = TFDS2NP_TAKESIZE_RATION_LIST
        self.tfds2np_takesize_flag = False
        self.decision_maker.infer_model_select_def()
        self.tr34_trainpip_warmup = self.decision_maker.infer_tr34_trainpip_warmup()
        self.tr34_hps_epochs_dict = self.decision_maker.infer_tr34_hps_epoch()
        self.tr34_hps_sample_dict = self.decision_maker.infer_tr34_hps_samplenum()

        as_timer("model_executor_init")

    def init_pipeline(self):
        # 1. classifiers init
        self.lr_libl_cls = SLLRLiblinear()
        self.lr_libl_cls.init(self.class_num)

        self.lr_sag_cls = SLLRSag()
        self.lr_sag_cls.init(self.class_num, MetaClsHPParams.lr_sag_cls_init_params)

        self.tr34_cls = ThinResnet34Classifier()
        self.tr34_cls.init(self.class_num)

        # 2. init

    def train_pipeline(self, train_tfds, update_train_data=True):
        if self.train_pip_id < len(self.tfds2np_take_size_array):
            if self.train_pip_id == 1:
                take_train_size = max(200, int(self.tfds2np_take_size_array[self.train_pip_id] * self.train_num))
            else:
                take_train_size = int(self.tfds2np_take_size_array[self.train_pip_id] * self.train_num)
        else:
            take_train_size = 200
        self.token_train_size += take_train_size
        self.cur_train_his_report = dict()
        as_timer("train_start")

        # 1. raw data: tfds2np.
        self.tfds_convertor.init_train_tfds(train_tfds, self.train_num)
        if update_train_data is True and self.feats_data_db.raw_data_db.raw_train_np_filled_num < self.train_num:
            accm_raw_train_np_dict = self.tfds_convertor.get_train_np_accm(take_train_size)
            # if self.minis_eda_report is None:
            self.minis_eda_report = minisamples_edaer(accm_raw_train_np_dict["x"], accm_raw_train_np_dict["y"])
            # decide if re-shuffle train tfds.
            if self.minis_eda_report.get("y_cover_rate") <= 0.5:
                info("Warning, old_y_cover_rate={} is too low, need re_shuffle train_tfds.".format(self.minis_eda_report.get("y_cover_rate")))
                self.tfds_convertor.init_train_tfds(train_tfds, self.train_num, force_shuffle=True)
                # renew data, shuffle, get data and get report.
                accm_raw_train_np_dict = self.tfds_convertor.get_train_np_accm(take_train_size)
                self.minis_eda_report = minisamples_edaer(accm_raw_train_np_dict["x"], accm_raw_train_np_dict["y"])
                info("Note, new_y_cover_rate={} ".format(self.minis_eda_report.get("y_cover_rate")))

            # update: meta-is_multilabel
            self.is_multilabel = self.minis_eda_report.get("is_multilabel")
            self.tr34_cls.renew_if_multilabel(self.is_multilabel)

            # if self.minis_eda_report is None:

            if self.tfds2np_takesize_flag is False:
                self.decision_maker.learn_train_minisamples_report(self.minis_eda_report)
                self.tfds2np_take_size_array = self.decision_maker.decide_tfds2np_array()
                self.tfds2np_takesize_flag = True

            info("Note, mini_eda_report = {}, tfds2np_takesize_array={}".format(self.minis_eda_report, self.tfds2np_take_size_array))
            self.feats_data_db.raw_data_db.put_raw_train_np(accm_raw_train_np_dict["x"], accm_raw_train_np_dict["y"])

        # 1-1: option: split val.
        if_split_val = self.decision_maker.decide_if_split_val(self.token_train_size)
        info("Val: if_val_on={}, if_split_val={}, len={}".format(IF_VAL_ON, if_split_val, len(self.val_sample_idxs)))
        if IF_VAL_ON and if_split_val and len(self.val_sample_idxs) == 0:
            val_mode = "bal"
            # val_mode = "random"
            val_num = self.decision_maker.decide_g_valid_num()
            self.val_sample_idxs = self.val_splitor.get_valid_sample_idxs(
                np.stack(self.feats_data_db.raw_data_db.raw_train_y_np_table_filled), val_num=val_num, mode=val_mode
            )
            self.feats_data_db.raw_data_db.put_split_valid_np(self.val_sample_idxs)
            self.cur_val_examples_y = self.feats_data_db.get_raw_train_y(self.val_sample_idxs)
            info(
                "Note, do val_split, mode={}, val_num={}, real_num={}, val_sampld_idxs={}".format(
                    val_mode, val_num, len(self.val_sample_idxs), self.val_sample_idxs
                )
            )

        # 2. model select.
        self.cur_cls_name = self.decision_maker.decide_model_select(self.train_pip_id)
        info("---------Model_Select_CLS={}---------".format(self.cur_cls_name))
        self.cur_cls = self.cur_cls_ins_table.get(self.cur_cls_name)
        self.cur_sampler = self.cur_sampler_table.get(self.cur_cls_name)

        if self.cur_cls_name in [CLS_LR_LIBLINEAER, CLS_LR_SAG]:
            # 3. sample train_idxs.
            # sample-even_sample.
            if self.is_multilabel is False:
                self.lr_sampler.init_train_y(self.feats_data_db.raw_data_db.raw_train_y_np_table_filled)
                class_inverted_index_array = self.lr_sampler.init_each_class_index_by_y(self.lr_sampler.train_y)
                # info("class_inverted_array={}".format(class_inverted_index_array))
                info(
                    "class_inverted_array len={}, some={}".format(
                        len(class_inverted_index_array), class_inverted_index_array[:3]
                    )
                )
                cur_train_sample_idxs = self.lr_sampler.init_even_class_index_by_each(class_inverted_index_array)
                info("cur_train_sample_idxs len={}, some={}".format(len(cur_train_sample_idxs), cur_train_sample_idxs[:3]))
                cur_train_sample_idxs = [item for sublist in cur_train_sample_idxs for item in sublist]
                info("cur_train_sample_idxs len={}, some={}".format(len(cur_train_sample_idxs), cur_train_sample_idxs[:3]))
                # filter val idxs out.
                cur_train_sample_idxs = [i for i in cur_train_sample_idxs if i not in self.val_sample_idxs]
                info("cur_train_sample_idxs len={}, some={}".format(len(cur_train_sample_idxs), cur_train_sample_idxs[:3]))
                as_timer("t_s3_trainidx_{}".format(len(cur_train_sample_idxs)))

            # config: all put into: nosample
            else:
                cur_train_sample_idxs = range(len(self.feats_data_db.raw_data_db.raw_train_y_np_table_filled))

            # 4. get asso-feats train.
            self.cur_feat_name = CLS_2_FEATNAME_REG_TABLE.get(self.cur_cls_name)
            self.use_feat_params = {"len_sample": 5, "sr": 16000}
            cur_train_examples_x = self.feats_data_db.get_raw_train_feats(
                self.cur_feat_name, cur_train_sample_idxs, self.use_feat_params
            )
            cur_train_examples_y = self.feats_data_db.get_raw_train_y(cur_train_sample_idxs)
            # info("cur_train_examples_x, shape={}, data={}".format(cur_train_examples_x.shape, cur_train_examples_x))
            # info("cur_train_examples_y, shape={}, data={}".format(cur_train_examples_y.shape, cur_train_examples_y))
            info("cur_train_examples_x, shape={},".format(cur_train_examples_x.shape))
            info("cur_train_examples_y, shape={},".format(cur_train_examples_y.shape))

            train_eda_report = sample_y_edaer(cur_train_examples_y)
            info("Note, train_eda_y_report = {}".format(train_eda_report))

            as_timer("t_s4_texamples_{}".format(len(cur_train_examples_x)))

            # 5. cur_cls train and fit
            if self.cur_cls_name == CLS_LR_LIBLINEAER:
                assert isinstance(self.cur_cls, SLLRLiblinear), "Error cur_cls is not {}".format(SLLRLiblinear.__name__)
                self.cur_cls.offline_fit(cur_train_examples_x, cur_train_examples_y, fit_params={"if_multilabel": self.is_multilabel})
            elif self.cur_cls_name == CLS_LR_SAG:
                assert isinstance(self.cur_cls, SLLRSag), "Error cur_cls is not {}".format(SLLRSag.__name__)
                self.cur_cls.offline_fit(cur_train_examples_x, cur_train_examples_y, fit_params={"if_multilabel": self.is_multilabel})

            as_timer("t_s5_fit_{}".format(len(cur_train_examples_x)))

        elif self.cur_cls_name in [CLS_TR34]:
            assert isinstance(self.cur_cls, ThinResnet34Classifier), "Error, cls select is {}".format(
                type(self.cur_cls)
            )
            # 3. sample train_idxs.
            train_use_y_labels = np.stack(self.feats_data_db.raw_data_db.raw_train_y_np_table_filled)
            info(
                "train_use_y_labels, type={}, shape={}".format(
                    # type(train_use_y_labels), train_use_y_labels.shape, train_use_y_labels
                    type(train_use_y_labels), train_use_y_labels.shape
                )
            )

            self.tr34_sampler = AutoSpSamplerNew(y_train_labels=train_use_y_labels)

            if self.is_multilabel is False:
                # old: use sampling.
                self.tr34_sampler.set_up()
                cur_train_sample_idxs = self.tr34_sampler.get_downsample_index_list_by_class(
                    per_class_num=Tr34SamplerHpParams.SAMPL_PA_F_PERC_NUM,
                    max_sample_num=self.tr34_hps_sample_dict.get("SAMP_MAX_NUM"),
                    min_sample_num=self.tr34_hps_sample_dict.get("SAMP_MIN_NUM"),
                )
            else:
                # try: use all or random.
                cur_train_sample_idxs = self.tr34_sampler.get_downsample_index_list_by_random(
                    max_sample_num=self.tr34_hps_sample_dict.get("SAMP_MAX_NUM"),
                    min_sample_num=self.tr34_hps_sample_dict.get("SAMP_MIN_NUM"))

            info("cur_train_sample_idxs len={}, some={}".format(len(cur_train_sample_idxs), cur_train_sample_idxs[:3]))
            # filter val idxs out.
            cur_train_sample_idxs = [i for i in cur_train_sample_idxs if i not in self.val_sample_idxs]
            info("cur_train_sample_idxs len={}, some={}".format(len(cur_train_sample_idxs), cur_train_sample_idxs[:3]))

            # 4. get asso-feats train.
            self.cur_feat_name = CLS_2_FEATNAME_REG_TABLE.get(self.cur_cls_name)
            if_train_feats_force = self.cur_cls.decide_if_renew_trainfeats()
            self.use_feat_params = self.cur_cls.imp_feat_args
            info("train_feats_params={}".format(self.use_feat_params))
            cur_train_examples_x = self.feats_data_db.get_raw_train_feats(
                self.cur_feat_name, cur_train_sample_idxs, self.use_feat_params, if_train_feats_force
            )
            cur_train_examples_y = self.feats_data_db.get_raw_train_y(cur_train_sample_idxs)
            # info("cur_train_examples_x, shape={}, data={}".format(cur_train_examples_x.shape, cur_train_examples_x[:1]))
            info("cur_train_examples_x, shape={}".format(cur_train_examples_x.shape))
            # info("cur_train_examples_y, shape={}, data={}".format(cur_train_examples_y.shape, cur_train_examples_y[:1]))
            info("cur_train_examples_y, shape={}".format(cur_train_examples_y.shape))
            train_eda_report = sample_y_edaer(cur_train_examples_y)
            info("Note, train_eda_y_report = {}".format(train_eda_report))

            as_timer("t_s4_texamples_{}".format(len(cur_train_examples_x)))

            # 5. cur_cls train and fit
            self.tr34_cls_train_pip_run += 1
            self.cur_train_his_report = self.cur_cls.online_fit(cur_train_examples_x, cur_train_examples_y, fit_params=self.tr34_hps_epochs_dict)
            info("tr34_cls_train_pip_run={}".format(self.tr34_cls_train_pip_run))
            as_timer("t_s5_fit_{}".format(len(cur_train_examples_x)))

        # 6. option: val
        if len(self.val_sample_idxs) > 0:
            # cur_val_examples_x = self.feats_data_db.get_raw_train_feats(self.cur_feat_name, self.val_sample_idxs,
            #                                                             self.use_feat_params)
            info("Note, cur_cls_name={}, get_val_feats done".format(self.cur_cls_name))
            if self.cur_cls_name == CLS_TR34:
                assert isinstance(self.cur_cls, ThinResnet34Classifier)
                if_force_val_feats = self.cur_cls.decide_if_renew_valfeats()
                use_feat_params = self.cur_cls.imp_feat_args
                cur_val_examples_x = self.feats_data_db.get_split_val_feats(
                    self.cur_feat_name, self.val_sample_idxs, use_feat_params, if_force_val_feats
                )
                cur_val_examples_x = np.array(cur_val_examples_x)
                cur_val_examples_x = cur_val_examples_x[:, :, :, np.newaxis]
                cur_val_preds = self.cur_cls.predict_val_proba(cur_val_examples_x)
            else:
                cur_val_examples_x = self.feats_data_db.get_split_val_feats(self.cur_feat_name, self.val_sample_idxs, self.use_feat_params)
                cur_val_preds = self.cur_cls.predict_proba(cur_val_examples_x, predict_prob_params={"if_multilabel": self.is_multilabel})

            info("Note, cur_cls_name={}, get_val_preds done".format(self.cur_cls_name))
            self.cur_val_nauc = ATEvaluator.autodl_auc(solution=self.cur_val_examples_y, prediction=cur_val_preds)
            # self.cur_val_nauc = round(2*self.cur_val_auc - 1, 6)
            info("Note, cur_cls_name={}, \033[1;31;m cur_val_nauc={}\033[0m".format(self.cur_cls_name, self.cur_val_nauc))
            as_timer("t_s6_val_{}".format(len(self.val_sample_idxs)))
        else:
            # self.cur_val_auc, self.cur_val_nauc = -1, -1
            self.cur_val_nauc = -1

        self.train_pip_id += 1
        as_timer("train_end")
        info(as_timer)
        self.cur_train_his_report["val_nauc"] = self.cur_val_nauc
        self.cur_train_his_report["cls_name"] = self.cur_cls_name
        # return self.cur_cls_name, self.cur_train_his_report
        return self.cur_train_his_report.copy()

        # 6. option: cur_cls eval.

    def test_pipeline(self, test_tfds):
        as_timer("test_start")
        # 1. raw data: tfds2np.
        self.tfds_convertor.init_test_tfds(test_tfds)
        if not self.feats_data_db.raw_data_db.if_raw_test_2_np_done:
            raw_test_np = self.tfds_convertor.get_test_np()
            assert isinstance(raw_test_np, list), "raw_test_np is not list"
            info("raw_test_np, len={}, ele={}".format(len(raw_test_np), raw_test_np[0]))
            self.feats_data_db.raw_data_db.put_raw_test_np(raw_test_np)
            as_timer("te_s0_tfds2np_{}".format(len(raw_test_np)))

        if self.cur_cls_name in [CLS_LR_LIBLINEAER, CLS_LR_SAG]:
            # 2. get asso-feats test.
            use_feat_params = {"len_sample": 5, "sr": 16000}
            cur_test_examples_x = self.feats_data_db.get_raw_test_feats(self.cur_feat_name, use_feat_params)

            as_timer("te_s1_examples_{}".format(len(cur_test_examples_x)))

            # 3. cur_cls test and predict.
            assert isinstance(self.cur_cls, AdlClassifier)
            cur_test_preds = self.cur_cls.predict_proba(cur_test_examples_x, predict_prob_params={"if_multilabel": self.is_multilabel})
            self.test_pip_id += 1
            as_timer("test_end")
            info(as_timer)
            return np.array(cur_test_preds)

        if self.cur_cls_name in [CLS_TR34]:
            # tr34 train pipeline warmup.
            while self.tr34_cls_train_pip_run < self.tr34_trainpip_warmup:
            # while self.tr34_cls_train_pip_run < TR34_TRAINPIP_WARMUP:
                self.train_pipeline(train_tfds=None, update_train_data=False)

            assert isinstance(self.cur_cls, ThinResnet34Classifier), "Error, cur_cls type error."
            # 2. get asso-feats test.
            if_force_test_feats = self.cur_cls.decide_if_renew_testfeats()
            use_feat_params = self.cur_cls.imp_feat_args
            cur_test_examples_x = self.feats_data_db.get_raw_test_feats(
                self.cur_feat_name, use_feat_params, if_force_test_feats
            )
            # need reformat for test
            info("tr34_test, type={}".format(type(cur_test_examples_x)))
            # cur_test_examples_x = np.array(cur_test_examples_x)
            cur_test_examples_x = np.asarray(cur_test_examples_x)

            cur_test_examples_x = cur_test_examples_x[:, :, :, np.newaxis]
            info("tr34_test, type={}, shape={}".format(type(cur_test_examples_x), cur_test_examples_x.shape))

            as_timer("te_s1_examples_{}".format(len(cur_test_examples_x)))

            # 3. cur_cls test and predict.
            assert isinstance(self.cur_cls, AdlClassifier)

            cur_test_preds = self.cur_cls.predict_proba(cur_test_examples_x)

            # del
            del cur_test_examples_x

            self.test_pip_id += 1
            as_timer("test_end")
            info(as_timer)
            return cur_test_preds

        as_timer("test_end")
        info(as_timer)


def main():
    ds_metadata = {"train_num": 3000, "test_num": 720, "class_num": 100}
    model_executor = ModelExecutor(ds_metadata)
    model_executor.init_pipeline()


if __name__ == "__main__":
    main()
