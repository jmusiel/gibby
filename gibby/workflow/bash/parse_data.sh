#!/bin/bash

python /home/jovyan/gibby/gibby/workflow/python/parse_data.py \
    --traj_path "/home/jovyan/shared-scratch/adeesh/splits_02_07/mappings/adslab_tags_full.pkl" \
    --adsorbate_path "/home/jovyan/shared-scratch/Brook/case_study2/data_and_input_files/adsorbate_db_neb_paper.pkl" \
    --data_path "/home/jovyan/shared-scratch/Brook/gibbs_proj/calculations_to_fair/" \
    --tag_mapping_path "/home/jovyan/shared-scratch/adeesh/splits_02_07/mappings/adslab_tags_full.pkl" \
    --metadata_path "" \
    --output_path "processed_data.pkl" \