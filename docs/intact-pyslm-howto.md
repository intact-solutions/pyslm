# Intact-Pyslm Quick Start (Windows)

Follow these steps in Windows PowerShell to install into a virtual environment and run the examples.

## 1) Prerequisites

- Install Python 3.10+ (64-bit) from https://www.python.org/downloads/
- Install Git for Windows: https://git-scm.com/download/win

## 2) Clone and create a virtual environment

```powershell
# Choose a folder (e.g., Documents)
cd $HOME\Documents

# Get the code
git clone https://github.com/intact-solutions/pyslm.git
cd pyslm

# Create and activate a virtual environment
py -3 -m venv .venv
.\.venv\Scripts\Activate

# Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel
```

## 3) Install requirements and Intact PySLM

```powershell
# Install dependencies from the repo root
pip install -r requirements.txt

# Use editable install so local changes are picked up
pip install -e .
```

If you see errors building C/C++ packages, install "Visual Studio Build Tools" (C++ workload):
https://visualstudio.microsoft.com/visual-cpp-build-tools/

## 4) Verify your install

```powershell
python -c "import sys, pyslm; print('Python:', sys.version); print('pyslm:', pyslm.__file__)"
```
The output path should point inside this cloned `pyslm` folder.

## 5) Run the examples

Run from the repository root.

- Visualize multi-scale island hatching
```powershell
python examples_intact\example_island_multiscale_viz.py
```

- Export SCODE files
```powershell
python examples_intact\export_scode_examples.py
```
This writes two files into `examples_intact/`:
- `neighborhood_paths_L0_x{X}_y{Y}_r{R}.scode`
- `layer_islands_L0.scode`

- Verify SCODE visually (replace the filename with yours if different)
```powershell
python examples_intact\plot_scode_verify.py --paths examples_intact\neighborhood_paths_L0_x-29.154_y-86.947_r4.00.scode --islands examples_intact\layer_islands_L0.scode
```

## 6) Genera tips

- Activate env in a new PowerShell window:
```powershell
.\.venv\Scripts\Activate
```
- Deactivate when finished:
```powershell
deactivate
```
- Update to latest code:
```powershell
git pull
pip install -e . --upgrade
```
- Confirm which PySLM is importing:
```powershell
python -c "import pyslm; print(pyslm.__file__)"
```

## 7) Troubleshooting

- Build errors for `triangle`, `manifold3d`, or `shapely`:
  - Install Visual C++ Build Tools (C++ workload) and retry `pip install -r requirements.txt`.
- The examples import the wrong PySLM (e.g., global install):
  - Ensure the virtual environment is activated and re-run `pip install -e .`.

