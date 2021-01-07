The directory includes `model.py` and `logic.py` which modifies the original files with same names in the submission code for DeepWisdom. 

To test different configurations:

1. Replace `model.py` and `logic.py` in the original files by those in this directory;
2. Edit the `config_path` variable in `logic.py` to select a configuration to run.



Available Configs:
`config1.yaml`: The original solution from DeepWisdom

`config_chucky_with_ensemble`: DeepWisdom + ensembling from DeepBlueAI

`config_chucky_with_freiburg_params`: DeepWisdom + hyperparameters form Freibrug

`config_chucky_with_freiburg_params_and_ensemble`: DeepWisdom + hyperparameters form Freibrug + ensembling from DeepBlueAI

