# coding:utf-8
from at_speech.at_speech_cons import CLS_LR_LIBLINEAER, CLS_LR_SAG, CLS_CNN, CLS_TR34
from at_speech.at_speech_config import MODEL_SELECT_DEF


class SpeechMetaLearner(object):
    def __init__(self):
        self.d_metadata = None
        self.d_eda_report = None
        self.class_num = None
        self.train_num = None
        self.mini_samples = None

        self.speech_domin_classifier = SpeechDomainClassifier()
        self.model_select_learner = ModelSelectLearner()
        self.tr34_hypparams_learner = Tr34HParamsLearner()

    def init_metadata(self, d_metadata):
        self.d_metadata = d_metadata
        self.class_num = self.d_metadata.get("class_num")
        self.train_num = self.d_metadata.get("train_num")

    def init_eda_report(self, d_eda_report):
        self.d_eda_report = d_eda_report

    def init_minisamples(self, minisamples):
        self.mini_samples = minisamples

    def predict_tr34_hypparams(self):
        tr34_hypparams = dict()
        tr34_hypparams["freeze_layer_num"] = self.tr34_hypparams_learner.predict_freeze_layernum(self.d_metadata)
        tr34_hypparams["spec_len_config"] = self.tr34_hypparams_learner.predict_freeze_layernum(self.d_metadata)
        return tr34_hypparams

    def predict_model_select(self, train_loop_num):
        return self.model_select_learner.predict_train_cls_select(train_loop_num)


class ModelSelectLearner:

    OFFLINE_CLS_ZOO = [CLS_LR_LIBLINEAER, CLS_LR_SAG]
    ONLINE_CLS_ZOO = [CLS_CNN, CLS_TR34]

    def __init__(self):
        self.cur_train_cls_name = None
        self.model_select_def = MODEL_SELECT_DEF

    def predict_train_cls_select(self, train_loop_num):
        if train_loop_num in self.model_select_def:
            self.cur_train_cls_name = self.model_select_def.get(train_loop_num)

        return self.cur_train_cls_name


class SpeechDomainClassifier(object):
    DOMAIN_SPEAKER = "SPEAKER"
    DOMAIN_EMOTION = "EMOTION"
    DOMAIN_ACCENT = "ACCENT"
    DOMAIN_MUSIC = "MUSIC"
    DOMAIN_LANGUAGE = "LANGU"

    DOMAIN_LIST = [DOMAIN_SPEAKER, DOMAIN_EMOTION, DOMAIN_ACCENT, DOMAIN_MUSIC, DOMAIN_LANGUAGE]

    def predict_speech_domain(self) -> str:
        """
        TODO: to speech domain.
        :return:
        """
        i = 0
        return self.DOMAIN_LIST[i]


class Tr34HParamsLearner:
    INIT_FRZ_L_NUM = 124
    INIT_FRZ_L_NUM_WILD = 100
    CLASS_NUM_THS = 37

    # for HP: FE_RS_SPEC_LEN
    FE_RS_SPEC_LEN_CONFIG_AGGR = {
        1: (100, 100, 0.002),
        10: (500, 500, 0.003),
        21: (1500, 1500, 0.004),
        50: (2250, 2250, 0.004),
    }

    FE_RS_SPEC_LEN_CONFIG_MILD = {
        1: (250, 250, 0.002),
        10: (500, 500, 0.004),
        20: (1000, 1000, 0.002),
        50: (1500, 1500, 0.004),
    }

    def predict_freeze_layernum(self, d_metadata: dict) -> int:
        if d_metadata.get("class_num") >= self.CLASS_NUM_THS:
            frz_layer_num = self.INIT_FRZ_L_NUM
        else:
            frz_layer_num = self.INIT_FRZ_L_NUM_WILD
        return frz_layer_num

    def predict_round_spec_len(self, d_metadata: dict) -> dict:
        if d_metadata.get("class_num") >= self.CLASS_NUM_THS:
            round_spec_len = self.FE_RS_SPEC_LEN_CONFIG_AGGR
        else:
            round_spec_len = self.FE_RS_SPEC_LEN_CONFIG_MILD
        return round_spec_len


def main():
    speech_d_metadata = {"train_num": 3000, "class_num": 100}

    # tr34_hp_learner = Tr34HParamsLearner()
    # tr34_hp_learner.init_metadata(speech_d_metadata)
    # freeze_layer_num = tr34_hp_learner.predict_freeze_layernum()
    # print("freeze_layer_num={}".format(freeze_layer_num))

    speech_metalearner = SpeechMetaLearner()
    speech_metalearner.init_metadata(speech_d_metadata)

    # test: tr34_params
    tr34_hpparams = speech_metalearner.predict_tr34_hypparams()
    print(tr34_hpparams)

    # test: model_select
    train_loop_num = 2
    cur_cls_name = speech_metalearner.predict_model_select(train_loop_num)
    print(cur_cls_name)


if __name__ == "__main__":
    main()
