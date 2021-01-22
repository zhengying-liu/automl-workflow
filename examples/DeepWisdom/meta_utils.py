import os
import pandas as pd
import json

# feature_dict = {}

json_file = os.path.join(os.path.dirname(__file__), "feature.json")

feature_dict = {'avg_upper_cnt': 0.13243329407894736, 'check_len': 224.19571428571427, 'imbalance_level': 0,
                'avg_punct_cnt': 0.009241672368421052, 'is_shuffle': False, 'train_num': 40000, 'language': 'EN',
                'kurtosis': -2.0, 'avg_digit_cnt': 0.0056662657894736845, 'seq_len_std': 170,
                'first_detect_normal_std': 0.00025, 'test_num': 10000, 'max_length': 1830, 'avg_length': 230,
                'class_num': 2, 'min_length': 1}

time_record = {"get_nlp_train_dataset_new": 8.679076194763184, "init_test_tfds": 0.0028562545776367188,
               "init_new_train_tfds": 0.0025186538696289062, "shuffle dataset": 0.0013530254364013672,
               "_tokenize_chinese_words_train": 0.735037088394165, "svm_predict_proba": 0.010643959045410156,
               "check_label_distribution": 0.0019190311431884766, "check_input_length": 0.0022983551025390625,
               "init_train_tfds": 0.0048444271087646484, "_tokenize_chinese_words_test": 1.1358966827392578,
               "clean_zh_text_valid": 0.017694950103759766, "vectorize_data": 0.17250323295593262,
               "_tokenize_chinese_words_valid": 0.11153650283813477, "clean_zh_text_test": 0.20546817779541016,
               "del trainsformer and init": 0.002473592758178711, "get_nlp_test_dataset_numpy_test": 1.998854637145996,
               "clean_zh_text_train": 0.042249202728271484, "snoop_data": 0.0045893192291259766,
               "build model": 6.818771362304688e-05, "svm fit": 0.18763494491577148,
               "svm_token_transform_test": 0.22721481323242188, "to_corpus_test": 1.2939767837524414,
               "sample_data_from_input": -0.007325410842895508,
               "get_nlp_train_dataset_to_numpy_train": 0.8225908279418945}

dir = os.path.dirname(__file__)
feature_json_dir = os.path.join(dir, "AutoNLP//upwind_flys_update//0215")
# feature_files = os.listdir(feature_json_dir)
time_json_dir = os.path.join(dir, "AutoNLP//upwind_flys_update//0216")
# time_files = os.listdir(time_json_dir)


def to_feature_csv():
    cnt = 0
    df = pd.DataFrame()
    for file in feature_files:
        if not file.endswith(".json"):
            continue
        with open(os.path.join(feature_json_dir, file), "r") as f:
            features = json.load(f)
            new = pd.DataFrame({key: val for key, val in features.items()}, index=[1])
            new["dataset_name"] = file.split(".")[0]
            df = pd.concat([df, new], axis=0)
            cnt += 1
            print(new["dataset_name"])
    df.to_csv(os.path.join(feature_json_dir, "feature.csv"), index="dataset_name")


def to_time_csv():
    cnt = 0
    df = pd.DataFrame()
    for file in time_files:
        if not file.endswith(".json"):
            continue
        with open(os.path.join(time_json_dir, file), "r") as f:
            features = json.load(f)
            new = pd.DataFrame({key: val for key, val in features.items()}, index=[1])
            new["dataset_name"] = file.split(".")[0]
            df = pd.concat([df, new], axis=0)
            cnt += 1
            print(new["dataset_name"])
    df.to_csv(os.path.join(time_json_dir, "time_record.csv"))

# to_time_csv()