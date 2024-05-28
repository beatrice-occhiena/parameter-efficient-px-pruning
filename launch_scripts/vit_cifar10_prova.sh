arch=vitB32
pretrain=openai
wrr=0.018 #[!! 0.64, 0.4096, 0.262, 0.1678, !! 0.107, 0.0688, 0.044, 0.0283, !! 0.018]
seed=42
pruner=SynFlow #SNIP, SynFlow
enhancement=LoRAinspired
rank=10

python main.py \
--experiment=${pruner} \
--experiment_name=CIFAR10/${arch}/${pretrain}/${wrr}/${pruner}/run${seed} \
--experiment_args="{'weight_remaining_ratio': ${wrr}, 'rounds': 100, 'batch_limit': 2, 'aux_model': '${arch}_no_act', 'pretrain': '${pretrain}'}" \
--enhancement_args="{'enhancement': '${enhancement}', 'rank': ${rank}}" \
--dataset=CIFAR10 \
--dataset_args="{'root': 'data/CIFAR10'}" \
--arch=${arch} \
--batch_size=128 \
--epochs=160 \
--num_workers=4 \
--seed=${seed} \
--pruner=${pruner} \
--grad_accum_steps=1