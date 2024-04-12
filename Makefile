train:
	rlaunch -P1 --positive-tags=2080ti --preemptible=no --cpu=16 --gpu=8 --memory=204800 -- torchrun --nproc_per_node 8 train.py --resume

eval:
	rlaunch --positive-tags=2080ti --cpu=6 --gpu=1 --memory=20480 -- python3 infer_to_json.py epoch_200
