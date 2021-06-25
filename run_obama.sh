GPU_ID=$1
export CUDA_VISIBLE_DEVICES=$GPU_ID
NUM_IMAGES=4

python run.py \
 -face_comparer_config configs/arcface_best.yml \
 -output_dir experiment_outputs/obama/fair \
 -overwrite \
 -duplicates=$NUM_IMAGES \
 -loss_str=100*L2+0.05*GEOCROSS+5.0*IDENTITY_SCORE \
 -input_dir=experiments/obama_small \
 -targets_dir=experiments/obama_large \
 -gpu_id=0 \
 -eps=0.006 && \
python run.py \
 -face_comparer_config configs/arcface_best.yml \
 -output_dir experiment_outputs/obama/black \
 -overwrite \
 -duplicates=$NUM_IMAGES \
 -loss_str=100*L2+0.05*GEOCROSS+2.0*ATTR_0_IS_1 \
 -input_dir=experiments/obama_small \
 -targets_dir=experiments/obama_large \
 -gpu_id=0 \
 -eps=0.006 && \
python run.py \
 -face_comparer_config configs/arcface_best.yml \
 -output_dir experiment_outputs/obama/white \
 -overwrite \
 -duplicates=$NUM_IMAGES \
 -loss_str=100*L2+0.05*GEOCROSS+2.0*ATTR_6_IS_1 \
 -input_dir=experiments/obama_small \
 -targets_dir=experiments/obama_large \
 -gpu_id=0 \
 -eps=0.006

