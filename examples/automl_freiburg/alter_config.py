# Author: Zhengying Liu
# Date: 17 Feb 2021

import os
import yaml

from pprint import pprint

config_deepwisdom_path = 'configs/config_deepwisdom_dataloading.yaml'
with open(config_deepwisdom_path, 'r') as f:
    config_deepwisdom = yaml.safe_load(f)

config_dir = 'configs/effnet_optimized_per_dataset_new_cs_new_data_03_14_DeepWisdom'
for filename in os.listdir(config_dir):
    if filename.endswith(".yaml"):
        print("Begin altering for {}...".format(filename))
        filepath = os.path.join(config_dir, filename)
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        config['autocv']['dataset'] = config_deepwisdom['autocv']['dataset']
        with open(filepath, 'w') as f:
            yaml.dump(config, f)
        print("Success: {}".format(filepath))

print("Successfully altered config!")