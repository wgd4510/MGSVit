python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 mocov3_train_pretrain.py --lmdb_name B2 --arch vit_base --checkpoints ./output/pretrain/moco_bs32_epoch100_B2 --epochs 100

python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 mocov3_train_pretrain.py --lmdb_name B12 --arch vit_base --checkpoints ./output/pretrain/moco_bs32_epoch100_B12 --epochs 100 