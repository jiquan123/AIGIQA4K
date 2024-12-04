#!/bin/bash

# FR-AIGCIQA on the I2IQA subset
for true_score in  MOS_q MOS_a MOS_c 
do  
    echo "python -u train.py --lr=1e-5 --backbone=vit --using_prompt=1 --log_info=$true_score  --true_score=$true_score --benchmark=I2I "
    python -u train.py --lr=1e-5 --backbone=vit --using_prompt=1 --log_info=$true_score  --true_score=$true_score  --benchmark=I2I
done

# PR-AIGCIQA on the PKU-AIGIQA-4K dataset
for true_score in MOS_q MOS_a MOS_c 
do  
    echo "python -u train.py --lr=1e-5 --backbone=vit --using_prompt=1 --log_info=$true_score  --true_score=$true_score --benchmark=AIGIQA4K "
    python -u train.py --lr=1e-5 --backbone=vit --using_prompt=1 --log_info=$true_score  --true_score=$true_score  --benchmark=AIGIQA4K
done

# NR-AIGCIQA on the PKU-AIGIQA-4K dataset
for true_score in  MOS_q MOS_a MOS_c 
do  
    echo "python -u train.py --lr=1e-5 --backbone=vit --using_prompt=0 --log_info=$true_score  --true_score=$true_score --benchmark=AIGIQA4K "
    python -u train.py --lr=1e-5 --backbone=vit --using_prompt=0 --log_info=$true_score  --true_score=$true_score  --benchmark=AIGIQA4K
done
