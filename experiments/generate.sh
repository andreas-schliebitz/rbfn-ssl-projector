#!/usr/bin/env bash

datasets=("imagenet-100" "open-images-v7-50" "open-images-v7-30" "open-images-v7-10")

for dataset in "${arr[@]}"; do
	# Generate runs using architecture's default MLP heads
	python3 generate.py \
	    --config "configs/default_${dataset}.yaml" \
	    --slurm \
	    --output "runs/default_${dataset}" \
	    --slurm-cpus-per-task 32 \
	    --slurm-gres gpu:ampere80:8 \
	    --slurm-ntasks 8

	# Generate runs using RBFN heads
	python3 generate.py \
	    --config "configs/rbfn_${dataset}.yaml" \
	    --slurm \
	    --output "runs/rbfn_${dataset}" \
	    --slurm-cpus-per-task 32 \
	    --slurm-gres gpu:ampere80:8 \
	    --slurm-ntasks 8
done

