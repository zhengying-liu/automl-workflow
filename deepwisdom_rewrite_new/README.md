To test the new submission, please first add these 2 (big) files under directory `Auto_Image/`:

```markdown
AutoImage/
|- ... 
|- models/
      |- r9-70e4b5c2.pth.tar 
      |- resnet18-5c106cde.pth
```

Possible modifications needed to run the experiment on a different environment:

1. Edit the path of installing the package `ASlibScenario-master` in `Auto_Image/skeleton/projects/logic.py`
2. Edit the path of `model_fn` in `Auto_Image/skeleton/projects/logic.py`
3. Edit the system path (`sys.path.insert(...)`) in `AutoFolio/autofolio/autofolio.py`

