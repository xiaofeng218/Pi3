from __future__ import annotations

from .forehoi_rerun_export import export_forehoi_rerun_sample
from .pi3x_rerun_export import (
    build_scene_gt_metric,
    convert_scene_gt_to_pred_scale,
    export_pi3x_rerun_sample as export_dexycb_rerun_sample,
)


def _infer_dataset_name(batch) -> str:
    if not batch:
        return ""
    dataset_value = batch[0].get("dataset", "")
    if isinstance(dataset_value, (list, tuple)):
        return str(dataset_value[0]) if dataset_value else ""
    return str(dataset_value)


def export_pi3x_rerun_sample(output_path, batch, pred, gt, sample_index, data_root, mano_layer=None, item_name="pi3x_train_sample"):
    dataset_name = _infer_dataset_name(batch).lower()
    if dataset_name == "forehoi":
        return export_forehoi_rerun_sample(
            output_path=output_path,
            batch=batch,
            pred=pred,
            gt=gt,
            sample_index=sample_index,
            data_root=data_root,
            mano_layer=mano_layer,
            item_name=item_name,
        )
    return export_dexycb_rerun_sample(
        output_path=output_path,
        batch=batch,
        pred=pred,
        gt=gt,
        sample_index=sample_index,
        data_root=data_root,
        mano_layer=mano_layer,
        item_name=item_name,
    )


__all__ = [
    "build_scene_gt_metric",
    "convert_scene_gt_to_pred_scale",
    "export_dexycb_rerun_sample",
    "export_forehoi_rerun_sample",
    "export_pi3x_rerun_sample",
]
