To test the new submission, please first add these 3 (big) files under directory `Auto_Image/`(same for `Auto_Video`). The files are available in the original submission of DeepWisdom.

```markdown
Auto_Image/
|- ... 
|- models/
      |- r9-70e4b5c2.pth.tar 
      |- resnet18-5c106cde.pth

Auto_Video/
|-
|- models/
			|- mc3_18-a90a0ba3.pth
```

Possible modifications needed to run the experiment on a different environment:

(Same for `Auto_Image` and `Auto_Video` )

1. Edit the path of installing the package `ASlibScenario-master` in `Auto_Image/skeleton/projects/logic.py`
2. Edit the path of `model_fn` in `Auto_Image/skeleton/projects/logic.py`
3. Edit the system path (`sys.path.insert(...)`) in `AutoFolio/autofolio/autofolio.py`
4. Edit configurations `USE_DEEPBLUE_ENSEMBLE` and `USE_FREIBURG_PARAMS` in `logic.py`
5. Change path of config files if needed.

