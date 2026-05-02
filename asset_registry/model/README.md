# Model Assets

Model assets are stored under `$PI3_DATA_ROOT/model/`; by default this is
`data/model/`.

| Asset | Default path | Source |
| --- | --- | --- |
| Pi3X | `data/model/pi3x/Pi3X` | Hugging Face repo `yyfz233/Pi3X` |
| HaMeR | `data/model/hamer/_DATA` | HaMeR demo data archive |

Expected HaMeR files after download:

```text
data/model/hamer/_DATA/
  hamer_ckpts/checkpoints/hamer.ckpt
  hamer_ckpts/model_config.yaml
  vitpose_ckpts/vitpose+_huge/wholebody.pth
  data/mano/MANO_RIGHT.pkl
  data/mano/MANO_LEFT.pkl
  data/mano_mean_params.npz
```

Run:

```bash
bash asset_registry/model/download_models.sh
source asset_registry/env.sh
```
