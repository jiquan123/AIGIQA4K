import torch
import os
import sys
import pandas as pd
from scipy import stats
from ImageReward import BLIPScore, CLIPScore, AestheticScore
from config import get_parser
from util import get_logger, log_and_print
import random
import torch.backends.cudnn as cudnn

sys.path.append('../')
torch.backends.cudnn.enabled = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
if __name__ == '__main__':

    args = get_parser().parse_known_args()[0]

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    base_logger = get_logger(f'exp/TIER-4K.log', args.log_info)
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    label_path = "./Dataset/AGIQA-4K/annotation.xlsx"
    generated_image_path = "./Dataset/AGIQA-4K/All"
    prompt_image_path = "./Dataset/AGIQA-4K/I2I/Image_prompt"

    data = pd.read_excel(label_path)
    generated_image_files_name = data['Generated_image']
    text_prompt = data['Text_prompt']
    label = data[args.true_score]
    generated_image_files_name, text_prompt, label = generated_image_files_name[:1600], text_prompt[:1600], label[:1600]
    print(len(text_prompt))
    
    model = BLIPScore('/root/.cache/ImageReward/med_config.json')
    #model = CLIPScore('/root/.cache/ImageReward')
    pred_scores = []
    with torch.no_grad():
        for index in range(len(text_prompt)):
            preds = model.score(text_prompt[index], os.path.join(generated_image_path, str('{}'.format(generated_image_files_name[index]))))
            #print(preds)
            pred_scores.append(preds)

    rho_s, _ = stats.spearmanr(pred_scores, label)
    rho_p, _ = stats.pearsonr(pred_scores, label)

    log_and_print(base_logger, f'spearmanr_correlation: {rho_s}, pearsonr_correlation: {rho_p}')

    # run: HF_ENDPOINT=https://hf-mirror.com python BLIPScore_evalaute.py
    #pip install image-reward


    



