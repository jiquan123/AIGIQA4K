#!/bin/bash
for true_score in  MOS_q MOS_a MOS_c 
do  
    echo "python -u train.py --backbone=vit --lr=1e-5 --using_image_prompt=1 --log_info=$true_score  --true_score=$true_score --benchmark=I2I"
    python -u train.py --backbone=vit --lr=1e-5 --using_image_prompt=1 --log_info=$true_score  --true_score=$true_score  --benchmark=I2I
done

for true_score in MOS_a MOS_c 
do  
    echo "python -u train.py --backbone=vit --lr=1e-5 --using_image_prompt=0 --log_info=$true_score  --true_score=$true_score --benchmark=AIGIQA4K"
    python -u train.py --backbone=vit --lr=1e-5 --using_image_prompt=0 --log_info=$true_score  --true_score=$true_score  --benchmark=AIGIQA4K
done

for true_score in  MOS_q MOS_a MOS_c 
do  
    echo "python -u train.py --backbone=vit --lr=1e-5 --using_image_prompt=1 --log_info=$true_score  --true_score=$true_score --benchmark=AIGIQA4K"
    python -u train.py --backbone=vit --lr=1e-5 --using_image_prompt=1 --log_info=$true_score  --true_score=$true_score  --benchmark=AIGIQA4K
done

for true_score in  MOS_q MOS_a MOS_c 
do  
    echo "python -u train.py --backbone=vit --lr=1e-5 --using_image_prompt=0 --log_info=$true_score  --true_score=$true_score --benchmark=T2I"
    python -u train.py --backbone=vit --lr=1e-5 --using_image_prompt=0 --log_info=$true_score  --true_score=$true_score  --benchmark=T2I
done

for true_score in  MOS_q MOS_a MOS_c 
do  
    echo "python -u train.py --backbone=vit --lr=1e-5 --using_image_prompt=0 --log_info=$true_score  --true_score=$true_score --benchmark=I2I"
    python -u train.py --backbone=vit --lr=1e-5 --using_image_prompt=0 --log_info=$true_score  --true_score=$true_score  --benchmark=I2I
done




