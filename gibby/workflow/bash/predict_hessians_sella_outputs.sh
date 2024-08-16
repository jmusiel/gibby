#!/bin/bash

python /home/jovyan/working/repos/gibby/gibby/workflow/python/predict_hessians_sella_outputs.py \
    --pickle_file "/home/jovyan/shared-scratch/Brook/gibbs_proj/sella_sps/sella_sp_ml_ts_pretrained_df_fixed_constraints_for_hessian.pkl" \
    --output_dir "/home/jovyan/shared-scratch/Brook/sella_hessians/" \
    --checkpoint_path "/home/jovyan/shared-scratch/joe/personal_checkpoints/ocp_public_3_3_2023/EqV2/eq2_153M_ec4_allmd.pt" \
