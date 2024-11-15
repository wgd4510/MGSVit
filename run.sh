# 自监督预训练S1模型
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 ours_train_pretrain.py --lmdb_name B2 --output_dir ./output/pretrain/mcmae_v1_bs32_epoch100_shortnum0_B2 --epochs 100 --shortnum 0 

# 自监督预训练S2模型
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 ours_train_pretrain.py --lmdb_name B12 --output_dir ./output/pretrain/mcmae_v1_bs32_epoch100_shortcut0_B12 --epochs 100 --shortnum 0 

# 基于SSL的预训练模型进行S1和S2融合微调训练
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 ours_train_linear.py --output_dir ./output/linear/SSL_mcmae_v1_bs32_epoch100_gmu_shortcut0_pre99_99_B12_B2 --lmdb_name B14 --s1_weights output/pretrain/mcmae_v1_bs32_epoch100_shortnum0_B2/checkpoint-99.pth --s2_weights output/pretrain/mcmae_v1_bs32_epoch100_shortcut0_B12/checkpoint-99.pth --shortnum 0

# 基于SSL的预训练模型进行S1微调训练
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 ours_train_linear.py --output_dir ./output/linear/SSL_mcmae_v1_bs32_epoch100_gmu_shortcut0_pre99_B2 --lmdb_name B2 --s1_weights output/pretrain/mcmae_v1_bs32_epoch100_shortnum0_B2/checkpoint-99.pth --shortnum 0

# 基于SSL的预训练模型进行S2微调训练
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 ours_train_linear.py --output_dir ./output/linear/SSL_mcmae_v1_bs32_epoch100_gmu_shortcut0_pre99_B12 --lmdb_name B12 --s2_weights output/pretrain/mcmae_v1_bs32_epoch100_shortcut0_B12/checkpoint-99.pth --shortnum 0

# 不加载预训练模型，直接训练S1模型
python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 ours_train_linear.py --output_dir ./output/linear/Sup_mcmae_v1_bs32_epoch100_gmu_shortcut2_B2 --lmdb_name B2 --shortnum 2

# 不加载预训练模型，直接训练S2模型
python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 ours_train_linear.py --output_dir ./output/linear/Sup_mcmae_v1_bs32_epoch100_gmu_shortcut2_B12 --lmdb_name B12 --shortnum 2

# 不加载预训练模型，直接训练S1+S2模型
python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 ours_train_linear.py --output_dir ./output/linear/Sup_mcmae_v1_bs32_epoch100_gmu_shortcut2_B12_B2 --lmdb_name B14 --shortnum 2

# 不加载预训练模型，直接训练S1模型
python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 ours_train_linear.py --output_dir ./output/linear/Sup_mcmae_v1_bs32_epoch100_gmu_shortcut6_B2 --lmdb_name B2 --shortnum 6

# 不加载预训练模型，直接训练S2模型
python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 ours_train_linear.py --output_dir ./output/linear/Sup_mcmae_v1_bs32_epoch100_gmu_shortcut6_B12 --lmdb_name B12 --shortnum 6

# 不加载预训练模型，直接训练S1+S2模型
python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 ours_train_linear.py --output_dir ./output/linear/Sup_mcmae_v1_bs32_epoch100_gmu_shortcut6_B12_B2 --lmdb_name B14 --shortnum 6