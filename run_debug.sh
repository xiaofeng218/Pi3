# # # 测试batch输出的depth、mask、rgb的效果。
# # python debug/export-dexycb-batch-rrd.py export-dexycb-batch-rrd \
# #     --data-root "$DEXYCB_ROOT" \
# #     --output ./outputs/export.rrd \
# #     --batch-size 2 \
# #     --frame-num 8 \
# #     --resolution 224 224 \
# #     --batch-index 0 \
# #     --sample-index 0

# # 测试输入的的depth和pi3预测的depth之间的差距
# PI3_DATA_ROOT="${PI3_DATA_ROOT:-data}"
# DEXYCB_ROOT="${DEXYCB_ROOT:-$PI3_DATA_ROOT/dataset/dexycb}"

# python debug/inspect_pi3x_pred_vs_gt.py\
#     --data-root "$DEXYCB_ROOT" \
#     --subject 20200709-subject-01 \
#     --batch-size 2 \
#     --frame-num 8 \
#     --batch-index 0 \
#     --sample-index 0 \
#     --output-rrd outputs/pi3x_pred_vs_gt_bundle/sequence.rrd

# 仅仅测试推理过程，损失函数为什么会出现none
source asset_registry/env.sh
python debug/pi3x_forward_finite_check.py \
    --config-name pi3x_hand_object \
    --max-iters 100 \
    --override train.auto_resume=false \
    --override train.resume=null \
    --override vis.enabled=false \
    --override train.num_workers=0 \
    --override test.num_workers=0 \
    --override log.output_dir=outputs/pi3x_forward_check \
    --override log.ckpt_dir=outputs/pi3x_forward_check/ckpts
