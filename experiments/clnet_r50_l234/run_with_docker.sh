#!bin/bash
export PYTHONPATH=code_path_in_container:$PYTHONPATH
source /software/conda/etc/profile.d/conda.sh

conda activate pysot

sh ./run_test_vot.sh
sh ./run_test_dtb.sh
sh ./run_test_nfs.sh
sh ./run_test_uav.sh
sh ./run_test_lasot.sh

