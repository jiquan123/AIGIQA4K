import torch
import torchvision.transforms as tr
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
from PIL import Image
import pandas as pd
import numpy as np


def random_shuffle(generated_image, prompt_image, label):
    randnum = 1
    np.random.seed(randnum)
    np.random.shuffle(generated_image)
    np.random.seed(randnum)
    np.random.shuffle(prompt_image)
    np.random.seed(randnum)
    np.random.shuffle(label)
    return generated_image, prompt_image, label


def load_image_label(args, generated_image_path, prompt_image_path, label_path):
    data = pd.read_excel(label_path)
    generated_image_files_name = data['Generated_image']
    prompt_image_files_name = data['Image_prompt']
    text_prompt = data['Text_prompt']
    labels = data[args.true_score]

    generated_image_list = []
    for name in generated_image_files_name:
        file = os.path.join(generated_image_path, str('{}'.format(name)))
        image = Image.open(file).convert('RGB')
        generated_image_list.append(image)

    prompt_image_list = []
    name_mask = prompt_image_files_name.isna()
    for i in range(len(prompt_image_files_name)):
        #print(name)
        if  name_mask[i] == True:
            prompt_image_list.append([])
        else:
            file = os.path.join(prompt_image_path, str('{}'.format(prompt_image_files_name[i])))
            image = Image.open(file).convert('RGB')
            prompt_image_list.append(image)

    text_prompt_list = []
    for prompt in text_prompt:
        text_prompt_list.append(prompt)

    label_list = []
    for label in labels:
        label_list.append(label)
    return generated_image_list, prompt_image_list, text_prompt_list, label_list


class AIGIQA4KDataset(Dataset):
    def __init__(self, generated_image, prompt_image, text_prompt, label, transforms, text_encoder_path):
        self.generated_image = generated_image
        self.prompt_image = prompt_image
        self.text_prompt = text_prompt
        self.label = label
        self.transforms = transforms
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)

    def __len__(self):
        return len(self.generated_image)

    def __getitem__(self, idx):
        data = {}
        data['generated_image'] = self.transforms(self.generated_image[idx])
        input_id0 = torch.ones(1)
        if self.prompt_image[idx] != []:
            data['prompt_image'] = self.transforms(self.prompt_image[idx])
            input_id1 = torch.ones(1)
        else:
            data['prompt_image'] = torch.zeros(data['generated_image'].shape)
            input_id1 = torch.zeros(1)
        data['input_ids'] = torch.cat([input_id0, input_id1], dim=-1)
        text_prompt = self.text_prompt[idx]
        encoded = self.tokenizer.encode_plus(
            text_prompt,
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=512
        )
        data['prompt'], data['attention_mask'] = encoded['input_ids'], encoded['attention_mask']
        data['true_score'] = self.label[idx]
        return data


def get_AIGIQA4K_dataloaders(args):
    label_path = "./Dataset/PKU-AIGIQA-4K/annotation.xlsx"
    generated_image_path = "./Dataset/PKU-AIGIQA-4K/All"
    prompt_image_path = "./Dataset/PKU-AIGIQA-4K/I2I/Image_prompt"
    text_encoder_path = "./bert-base-uncased"
    train_generated_image = []
    test_generated_image = []
    train_text_prompt = []
    test_text_prompt = []
    train_prompt_image = []
    test_prompt_image = []
    train_label = []
    test_label = []
    generated_image, prompt_image, text_prompt, label = load_image_label(args, generated_image_path, prompt_image_path, label_path)
    for i in range(len(generated_image)):
        if i % 4 == 3:
            test_generated_image.append(generated_image[i])
            test_prompt_image.append(prompt_image[i])
            test_text_prompt.append(text_prompt[i])
            test_label.append(label[i])
        else:
            train_generated_image.append(generated_image[i])
            train_prompt_image.append(prompt_image[i])
            train_text_prompt.append(text_prompt[i])
            train_label.append(label[i])

    if args.backbone == 'inceptionv4':
        resize_img_size = 320
        crop_img_size = 299
    else:
        resize_img_size = 256
        crop_img_size = 224

    train_transforms = tr.Compose([
        tr.Resize(resize_img_size),
        tr.RandomCrop(crop_img_size),
        tr.RandomHorizontalFlip(),
        tr.ToTensor(),
        #tr.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transforms = tr.Compose([
        tr.Resize(resize_img_size),
        tr.CenterCrop(crop_img_size),
        tr.ToTensor(),
        #tr.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataloaders = {}

    dataloaders['train'] = torch.utils.data.DataLoader(AIGIQA4KDataset(train_generated_image, train_prompt_image, train_text_prompt, train_label, train_transforms, text_encoder_path),
                                                       batch_size=args.train_batch_size,
                                                       shuffle=True,
                                                       pin_memory=True)

    dataloaders['test'] = torch.utils.data.DataLoader(AIGIQA4KDataset(test_generated_image, test_prompt_image, test_text_prompt, test_label, test_transforms, text_encoder_path),
                                                      batch_size=args.test_batch_size,
                                                      shuffle=False,
                                                      pin_memory=True)
    return dataloaders

def get_AIGIQA4KT2I_dataloaders(args):
    label_path = "./Dataset/PKU-AIGIQA-4K/annotation.xlsx"
    generated_image_path = "./Dataset/PKU-AIGIQA-4K/All"
    prompt_image_path = "./Dataset/PKU-AIGIQA-4K/I2I/Image_prompt"
    text_encoder_path = "./bert-base-uncased"
    train_generated_image = []
    test_generated_image = []
    train_text_prompt = []
    test_text_prompt = []
    train_prompt_image = []
    test_prompt_image = []
    train_label = []
    test_label = []
    generated_image, prompt_image, text_prompt, label = load_image_label(args, generated_image_path, prompt_image_path, label_path)
    generated_image, prompt_image, text_prompt, label = generated_image[1600:], prompt_image[1600:], text_prompt[1600:], label[1600:]
    for i in range(len(generated_image)):
        if i % 4 == 3:
            test_generated_image.append(generated_image[i])
            test_prompt_image.append(prompt_image[i])
            test_text_prompt.append(text_prompt[i])
            test_label.append(label[i])
        else:
            train_generated_image.append(generated_image[i])
            train_prompt_image.append(prompt_image[i])
            train_text_prompt.append(text_prompt[i])
            train_label.append(label[i])

    if args.backbone == 'inceptionv4':
        resize_img_size = 320
        crop_img_size = 299
    else:
        resize_img_size = 256
        crop_img_size = 224

    train_transforms = tr.Compose([
        tr.Resize(resize_img_size),
        tr.RandomCrop(crop_img_size),
        tr.RandomHorizontalFlip(),
        tr.ToTensor(),
        #tr.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transforms = tr.Compose([
        tr.Resize(resize_img_size),
        tr.CenterCrop(crop_img_size),
        tr.ToTensor(),
        #tr.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataloaders = {}

    dataloaders['train'] = torch.utils.data.DataLoader(AIGIQA4KDataset(train_generated_image, train_prompt_image, train_text_prompt, train_label, train_transforms, text_encoder_path),
                                                       batch_size=args.train_batch_size,
                                                       shuffle=True,
                                                       pin_memory=True)

    dataloaders['test'] = torch.utils.data.DataLoader(AIGIQA4KDataset(test_generated_image, test_prompt_image, test_text_prompt, test_label, test_transforms, text_encoder_path),
                                                      batch_size=args.test_batch_size,
                                                      shuffle=False,
                                                      pin_memory=True)
    return dataloaders

def get_AIGIQA4KI2I_dataloaders(args):
    label_path = "./Dataset/PKU-AIGIQA-4K/annotation.xlsx"
    generated_image_path = "./Dataset/PKU-AIGIQA-4K/All"
    prompt_image_path = "./Dataset/PKU-AIGIQA-4K/I2I/Image_prompt"
    text_encoder_path = "./bert-base-uncased"
    train_generated_image = []
    test_generated_image = []
    train_text_prompt = []
    test_text_prompt = []
    train_prompt_image = []
    test_prompt_image = []
    train_label = []
    test_label = []
    generated_image, prompt_image, text_prompt, label = load_image_label(args, generated_image_path, prompt_image_path, label_path)
    generated_image, prompt_image, text_prompt, label = generated_image[:1600], prompt_image[:1600], text_prompt[:1600], label[:1600]
    for i in range(len(generated_image)):
        if i % 4 == 3:
            test_generated_image.append(generated_image[i])
            test_prompt_image.append(prompt_image[i])
            test_text_prompt.append(text_prompt[i])
            test_label.append(label[i])
        else:
            train_generated_image.append(generated_image[i])
            train_prompt_image.append(prompt_image[i])
            train_text_prompt.append(text_prompt[i])
            train_label.append(label[i])

    if args.backbone == 'inceptionv4':
        resize_img_size = 320
        crop_img_size = 299
    else:
        resize_img_size = 256
        crop_img_size = 224

    train_transforms = tr.Compose([
        tr.Resize(resize_img_size),
        tr.RandomCrop(crop_img_size),
        tr.RandomHorizontalFlip(),
        tr.ToTensor(),
        #tr.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transforms = tr.Compose([
        tr.Resize(resize_img_size),
        tr.CenterCrop(crop_img_size),
        tr.ToTensor(),
        #tr.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataloaders = {}

    dataloaders['train'] = torch.utils.data.DataLoader(AIGIQA4KDataset(train_generated_image, train_prompt_image, train_text_prompt, train_label, train_transforms, text_encoder_path),
                                                       batch_size=args.train_batch_size,
                                                       shuffle=True,
                                                       pin_memory=True)

    dataloaders['test'] = torch.utils.data.DataLoader(AIGIQA4KDataset(test_generated_image, test_prompt_image, test_text_prompt, test_label, test_transforms, text_encoder_path),
                                                      batch_size=args.test_batch_size,
                                                      shuffle=False,
                                                      pin_memory=True)
    return dataloaders





