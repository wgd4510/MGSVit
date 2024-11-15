# 监督训练 B12+B2 的模型，原始模型结构
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 main_linprobe.py --batch_size 64 --lmdb_name B14 --opt adamw --blr 0.0001 --output_dir ./output/supervised_v1_B14


# python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 main_linprobe_multimodel.py --shortnum 0 --output_dir ./output/supervised_v1_B14_multimodel_gmu_shortnum0

# python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 main_linprobe_multimodel.py --shortnum 3 --output_dir ./output/supervised_v1_B14_multimodel_gmu_shortnum3

# python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 main_linprobe_multimodel.py --shortnum 2 --output_dir ./output/supervised_v1_B14_multimodel_gmu_shortnum2

# python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 main_linprobe_multimodel.py --shortnum 4 --output_dir ./output/supervised_v1_B14_multimodel_gmu_shortnum4 

# python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 main_linprobe_multimodel.py --shortnum 6 --output_dir ./output/supervised_v1_B14_multimodel_gmu_shortnum6


python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 main_linprobe_multiheadloss.py --shortnum 4 --output_dir ./output/supervised_v1_B14_multiheadloss_shortnum4

python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 main_linprobe_multiheadloss.py --shortnum 3 --output_dir ./output/supervised_v1_B14_multiheadloss_shortnum3

python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 main_linprobe_multimodel.py --shortnum 6 --output_dir ./output/supervised_v1_B14_multimodel_gmu_shortnum6