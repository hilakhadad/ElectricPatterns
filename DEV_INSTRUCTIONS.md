Development copy instructions

This file explains how to work with the development copy `role_based_segregation_dev` and how to restore the original if needed.

Quick start

1. Open a terminal and go to the dev copy:

```powershell
cd C:\Users\hilak\PycharmProjects\role_based_segregation_dev
conda activate role_seg_env
```

2. Work on the `dev` branch (created automatically):

```powershell
git status
# If needed, create a feature branch
git checkout -b feat/my-change
```

3. Run the local single-house test (edit paths inside the script first):

```powershell
cd experiment_pipeline
# Edit test_single_house.py to set LOCAL_INPUT_PATH and LOCAL_OUTPUT_PATH
python test_single_house.py
```

How to restore the original copy

If you want to discard the dev copy and start again from the original:

```powershell
Remove-Item -Path "C:\Users\hilak\PycharmProjects\role_based_segregation_dev" -Recurse -Force
Copy-Item -Path "C:\Users\hilak\PycharmProjects\role_based_segregation" -Destination "C:\Users\hilak\PycharmProjects\role_based_segregation_dev" -Recurse -Force
```

Notes and recommendations

- Commit small changes frequently and use branches for features/experiments.
- Keep the original `role_based_segregation` folder unchanged; treat `role_based_segregation_dev` as the working copy.
- If you'd like, I can create a script to reset the dev copy automatically.

Contact

If you want me to add CI, tests, or a `setup.py`/`pyproject.toml`, tell me which you'd prefer and I'll scaffold it.