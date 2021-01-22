import os
import json
from collections import namedtuple
import copy

# Config for Covertor
IF_RESET_TFGRAPH_SESS_RUN = False
TF_DATASET_TO_NUMPY_MODE = "graph"  # eager/graph
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 全局配置数据
autodl_global_config = {
    "meta_solution": {
        "cv_solution": "DeepWisdom",
        # "cv_solution": "kakaobrain",
        # "nlp_solution": "DeepBlueAI",
        "nlp_solution": "upwind_flys",
        "speech_solution": "PASA_NJU",
        # "speech_solution": "rank_2_fuzhi",
    },
    "data_space": {
        "domain_dataset": {
            "text": {"if_sample": True, "sample_ratio": 0.5},
            "speech": {"if_sample": True, "sample_ratio": 0.5},
        }
    },
    "speech_global_conf": None,
}


speech_global_conf_data = {
    "data_space": {
        "tf_dataset": {
            "if_shuffle": False,
            "shuffle_ratio": 0.5,
            "if_sample": True,
            # "sample_ratio": 0.2,
            # "sample_ratio": [0.1, 0.2, 0.3, 0.2, 0.2, 0.2],
            "sample_ratio": [0.1, 0.2, 0.4, 0.1, 0.2, 0.2],
        }
    },
    "model_space": {
        "model_loop_planniing": {
            "lightwei_train_end_loop": 3,
            "midwei_train_start_loop": 3,
            "midwei_predict_block_loop": 11,
        }
    },
}

autodl_global_config["speech_global_conf"] = speech_global_conf_data

# Config for Covertor
IF_RESET_TFGRAPH_SESS_RUN = False
TF_DATASET_TO_NUMPY_MODE = "graph"  # eager/graph

# Config for Solution
VIDEO_SOLUTION_FLAG = "2d" # 2d/3d



class MetaSoluConf(object):
    def __init__(self):
        self.cv_solution = None
        self.nlp_solution = None
        self.speech_solution = None


class DsDomainDatasetConf(object):
    def __init__(self):
        self.if_sample = None
        self.sample_ratio = None


class DsDomainDatasetSets(object):
    def __init__(self):
        self.text = DsDomainDatasetConf()
        self.speech = DsDomainDatasetConf()


class DsConf(object):
    def __init__(self):
        self.domain_dataset = DsDomainDatasetSets()


class AutoDlConf(object):
    def __init__(self):
        self.meta_solution = MetaSoluConf()
        self.data_space = DsConf()


class ConfigParserA(object):
    def _json_object_hook(self, d):
        return namedtuple("X", d.keys())(*d.values())

    def json2obj(self, data):
        return json.loads(data, object_hook=self._json_object_hook)

    def from_type_autodlconf(self, conf_data) -> AutoDlConf:
        # obj: typeclass = copy.deepcopy(self.json2obj(json.dumps(conf_data)))
        return copy.deepcopy(self.json2obj(json.dumps(conf_data)))


autodl_g_conf_repr = json.dumps(autodl_global_config, indent=4)

config_parser_a = ConfigParserA()

AUTODL_G_CONF = config_parser_a.from_type_autodlconf(autodl_global_config)
META_SOLUS = AUTODL_G_CONF.meta_solution
DM_DS_PARAS = AUTODL_G_CONF.data_space.domain_dataset

speech_global_conf = AUTODL_G_CONF.speech_global_conf
speech_ds_tds_conf = speech_global_conf.data_space.tf_dataset
speech_ms_conf = speech_global_conf.model_space
speech_ms_mlp_conf = speech_ms_conf.model_loop_planniing


def main():
    config_parser_a = ConfigParserA()
    autodl_g_conf = config_parser_a.from_type_autodlconf(autodl_global_config)
    print(autodl_g_conf.meta_solution.speech_solution)
    print(autodl_g_conf.data_space.domain_dataset.text.if_sample)
    print(autodl_g_conf.data_space.domain_dataset.speech.sample_ratio)


if __name__ == "__main__":
    main()
