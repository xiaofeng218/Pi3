# Asset Registry

This directory documents and bootstraps external assets for this repository.
Actual downloaded files are stored under `PI3_DATA_ROOT`, which defaults to
repo-root `data/` and is ignored by git.

Standard layout:

```text
data/
  model/
    pi3x/Pi3X/
    hamer/_DATA/
  dataset/
    dexycb/
```

To store assets outside the repository, set one variable before sourcing the
environment:

```bash
export PI3_DATA_ROOT=/data/pi3-assets
source asset_registry/env.sh
```

Load the standard environment before running training or debug scripts:

```bash
source asset_registry/env.sh
```

Download model assets:

```bash
bash asset_registry/model/download_models.sh
```

Download dataset assets:

```bash
bash asset_registry/dataset/download_datasets.sh
```

For DexYCB object canonical views, run after the dataset download:

```bash
bash dex-ycb-toolkit/run_obj_render.sh "$DEXYCB_ROOT" canonical_views_224 45 224
```

The object rendering step requires EGL/OpenGL system libraries and a working
PyOpenGL environment.
