export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/train_pi3x.py --config-name pi3x_hand_object