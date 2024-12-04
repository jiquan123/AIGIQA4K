# AIGIQA4K
This is the pytorch code implementation of NR-AIGCIQA/FR-AIGCIQA/PR-AIGCIQA introduced in the paper [PKU-AIGIQA-4K: A Perceptual Quality Assessment Database for Both Text-to-Image and Image-to-Image AI-Generated Images](https://arxiv.org/abs/2404.18409).

## Pre-trained visual backbone
For feature extraction from input images, we selected several backbone
network models pre-trained on the ImageNet dataset, including:
-  VGG16 [weights](https://download.pytorch.org/models/vgg16-397923af.pth)
-  VGG19 [weights](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)
-  ResNet18 [weights](https://download.pytorch.org/models/resnet18-f37072fd.pth)
-  ResNet50 [weights](https://download.pytorch.org/models/resnet50-0676ba61.pth)
-  InceptionV4 [weights](http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth)

If you want to train a model with the above pretrained models, ensure that all downloaded models are placed in the designated directories as follows.
```
├── pretrained
│   ├── inceptionv4-8e4777a0.pth
│   ├── resnet18.pth
│   ├── resnet50.pth
│   ├── vgg16-397923af.pth
│   ├── vgg19-dcbb9e9d.pth
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
cd AIGIQA4K
```

To train the NR-AIGCIQA/FR-AIGCIQA model on the I2IQA subset.

```bash
# true_score: MOS_q MOS_a MOS_c
# NR-AIGCIQA 
python -u train.py --lr=1e-5 --backbone=vit --using_prompt=0 --log_info=$true_score  --true_score=$true_score  --benchmark=I2I
# FR-AIGCIQA 
python -u train.py --lr=1e-5 --backbone=vit --using_prompt=1 --log_info=$true_score  --true_score=$true_score  --benchmark=I2I
```

To train the NR-AIGCIQA model on the T2IQA subset.

```bash
# true_score: MOS_q MOS_a MOS_c
# NR-AIGCIQA 
python -u train.py --lr=1e-5 --backbone=vit --using_prompt=0 --log_info=$true_score  --true_score=$true_score  --benchmark=T2I
```

To train the NR-AIGCIQA/PR-AIGCIQA model on the whole PKU-AIGIQA-4K dataset.

```bash
# true_score: MOS_q MOS_a MOS_c
# NR-AIGCIQA 
python -u train.py --lr=1e-5 --backbone=vit --using_prompt=0 --log_info=$true_score  --true_score=$true_score  --benchmark=AIGIQA4K
# PR-AIGCIQA 
python -u train.py --lr=1e-5 --backbone=vit --using_prompt=1 --log_info=$true_score  --true_score=$true_score  --benchmark=AIGIQA4K
```

You can also use the following command to train the model.
```bash
bash run.sh
```

## Contact
If you have any question, please contact yuanjiquan@stu.pku.edu.cn


