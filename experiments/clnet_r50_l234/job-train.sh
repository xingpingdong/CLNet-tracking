#!/bin/bash

sudo singularity exec -B code_path_in_host:code_path_in_container --nv docker://d243100603/pysot:v1 bash train_last.sh
