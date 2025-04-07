python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=$1 \
    --master_addr="173.17.0.2" \
    --master_port=1234 run.py