arch=resnet50
wrr=0.018
seed=42

python main.py \
--experiment=PX \
--experiment_name=ImageNet/${arch}/${wrr}/PX/run${s} \
--experiment_args="{'weight_remaining_ratio': ${wrr}, 'rounds': 100, 'batch_limit': 23, 'aux_model': '${arch}_no_act'}" \
--dataset=ImageNet \
--dataset_args="{'root': 'data/ImageNet'}" \
--arch=${arch} \
--batch_size=448 \
--epochs=90 \
--num_workers=4 \
--seed=${s} \
--pruner=PX \
--grad_accum_steps=1
