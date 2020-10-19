# Produce the results on our paper with our docker image
# We test our code on a Nvidia RTX 2080Ti GPU
## Prerequisites
You need install singularity (https://singularity.lbl.gov/docs-installation)

## Testing
1. Modify the paths: code_path_in_host, and code_path_in_container as the root directory of our code, in job-test.sh and run_with_docker.sh
2. Run job-test.sh to get our results
	sh ./job-test.sh

## Training
1. Modify the paths: code_path_in_host, and code_path_in_container as the root directory of our code, in job-train.sh and train_last.sh
2. Run job-train.sh to get our results
	sh ./job-train.sh
