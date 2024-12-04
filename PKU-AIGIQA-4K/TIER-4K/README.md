# TIER-4K
This is the pytorch code implementation of TIER-NR/TIER-FR/TIER-PR introduced in the paper [PKU-AIGIQA-4K: A Perceptual Quality Assessment Database for Both Text-to-Image and Image-to-Image AI-Generated Images](https://arxiv.org/abs/2404.18409).

## Pre-trained visual backbone
For feature extraction from input images, we selected several backbone
network models pre-trained on the ImageNet dataset, including:
-  ResNet18 [weights](https://download.pytorch.org/models/resnet18-f37072fd.pth)
-  ResNet50 [weights](https://download.pytorch.org/models/resnet50-0676ba61.pth)
-  InceptionV4 [weights](http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth)
If you want to train a model with the above pretrained models, ensure that all downloaded models are placed in the designated directories as follows.
```
├── pretrained
│   ├── inceptionv4-8e4777a0.pth
│   ├── resnet18.pth
│   ├── resnet50.pth
```

### Bert-base-uncased
In our paper, we choose Bert-base-uncased as the text encoder for extracting features from text prompts. 

It can be downloaded using this link: [[百度网盘](https://pan.baidu.com/s/19TDTIm6_0QJtm8N1YlfMjg) (提取码：text)].

or it can be accessed on [huggingface](https://huggingface.co/docs/transformers/installation).

Ensure that the downloaded model is placed in the designated directories as follows.
```
├── bert-base-uncased
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   ├── vocab.txt
```

## Database
The constructed PKU-AIGIQA-4K database can be accessed using the links below.
Download PKU-AIGIQA-4K database:
1.[[百度网盘](https://pan.baidu.com/s/1Co7Sca7Yl_RWNz_UP7zHuA) 
(提取码：AIGI)].
2.[[Google Drive](https://drive.google.com/file/d/1EuXe_6UNONJSH91uI3edrMMe7utOmpFz/view?usp=sharing)]

The data structure used for this repo should be:
```
├── Dataset
│   ├── PKU-AIGIQA-4K
│   │   ├── All
│   │   │   ├── DALLE_1000_00.jpg
│   │   │   ├── ...
│   │   │   ├── SD_1199_11.jpg
│   │   ├── I2I
│   │   │   ├── Generated_image
│   │   │   │   ├── All
│   │   │   │   │   ├── ....jpg
│   │   │   │   ├── MJ
│   │   │   │   │   ├── ....jpg
│   │   │   │   ├── SD
│   │   │   │   │   ├── ....jpg
│   │   │   ├── Image_prompt
│   │   │   │   ├── 0.jpg
│   │   │   │   ├── ...
│   │   │   │   ├── 199.jpg
│   │   ├── T2I
│   │   │   ├── All
│   │   │   │   ├── ....jpg
│   │   │   ├── DALLE
│   │   │   │   ├── ....jpg
│   │   │   ├── SD
│   │   │   │   ├── ....jpg
│   │   │   ├── MJ
│   │   │   │   ├── ....jpg
│   │   │   ├── SD
│   │   │   │   ├── ....jpg
│   │   ├── annotation.xlsx
```

## Training
First, you should install the required packages and
```
cd TIER-4K
```

To train the TIER-NR/TIER-FR model on the I2IQA subset.

```bash
# true_score: MOS_q MOS_a MOS_c
# TIER-NR 
python -u train.py --lr=1e-5 --backbone=vit --using_image_prompt=0 --log_info=$true_score  --true_score=$true_score  --benchmark=I2I
# TIER-FR
python -u train.py --lr=1e-5 --backbone=vit --using_image_prompt=1 --log_info=$true_score  --true_score=$true_score  --benchmark=I2I
```

To train the TIER-NR model on the T2IQA subset.

```bash
# true_score: MOS_q MOS_a MOS_c
# TIER-NR 
python -u train.py --lr=1e-5 --backbone=vit --using_image_prompt=0 --log_info=$true_score  --true_score=$true_score  --benchmark=T2I
```

To train the TIER-NR/TIER-PR model on the whole PKU-AIGIQA-4K dataset.

```bash
# true_score: MOS_q MOS_a MOS_c
# TIER-NR
python -u train.py --lr=1e-5 --backbone=vit --using_image_prompt=0 --log_info=$true_score  --true_score=$true_score  --benchmark=AIGIQA4K
# TIER-PR 
python -u train.py --lr=1e-5 --backbone=vit --using_image_prompt=1 --log_info=$true_score  --true_score=$true_score  --benchmark=AIGIQA4K
```

You can also use the following command to train the model.
```bash
bash run.sh
```

## Evaluation of CLIPScore/BLIPScore/ImageReward/PickScore on the PKU-AIGIQA-4K dataset
```bash
# pip install image-reward
# CLIPScore or BLIPScore 
# Default: BLIPScore
# You can change the model to CLIPScore
python BLIPScore_evalaute.py 
# ImageReward
python ImageReward_evalaute.py
# PickScore
python Pick_evalaute.py
```

These codes are based on [ImageReward](https://github.com/THUDM/ImageReward) and [PickScore](https://github.com/yuvalkirstain/PickScore), thanks for their great work.

## Contact
If you have any question, please contact yuanjiquan@stu.pku.edu.cn


