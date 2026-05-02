# Dataset Assets

Dataset assets are stored under `$PI3_DATA_ROOT/dataset/`; by default this is
`data/dataset/`.

| Asset | Default path | Source |
| --- | --- | --- |
| DexYCB | `data/dataset/dexycb` | Google Drive IDs used by `dex-ycb-toolkit/download_dexycb.sh` |

Run:

```bash
bash asset_registry/dataset/download_datasets.sh
source asset_registry/env.sh
```

After downloading DexYCB, generate object canonical views if training uses
`object_multiview_subdir: canonical_views_224`:

```bash
bash dex-ycb-toolkit/run_obj_render.sh "$DEXYCB_ROOT" canonical_views_224 45 224
```
