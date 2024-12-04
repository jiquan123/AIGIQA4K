import torch
import torch.nn as nn
import os
import sys
from scipy import stats
from tqdm import tqdm
from backbone.resnet import resnet18_backbone, resnet50_backbone
from backbone.inceptionv4 import inceptionv4
from backbone.vit import ViTExtractor
from dataloader.AIGIQA4K import get_AIGIQA4K_dataloaders, get_AIGIQA4KT2I_dataloaders, get_AIGIQA4KI2I_dataloaders
from model import Encoder, MLP
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

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.backbone == 'resnet50':
        backbone = resnet50_backbone().to(device)
        encoder = Encoder(backbone).to(device)
        regressor = MLP(args, 2048).to(device)
    elif args.backbone == 'resnet18':
        backbone = resnet18_backbone().to(device)
        encoder = Encoder(backbone).to(device)
        regressor = MLP(args, 512).to(device)
    elif args.backbone == 'vit':
        backbone = ViTExtractor().to(device)
        encoder = Encoder(backbone).to(device)
        regressor = MLP(args, 1024).to(device)
    else:
        backbone = inceptionv4(num_classes=1000, pretrained='imagenet').to(device)
        encoder = Encoder(backbone).to(device)
        regressor = MLP(args, 1536).to(device)

    if args.benchmark =='AIGIQA4K':
        dataloaders = get_AIGIQA4K_dataloaders(args)
    elif args.benchmark =='I2I':
        dataloaders = get_AIGIQA4KI2I_dataloaders(args)
    else:
        dataloaders = get_AIGIQA4KT2I_dataloaders(args)

    criterion = nn.MSELoss(reduction='mean').cuda()
    # criterion = nn.SmoothL1Loss(reduction='mean')
    optimizer = torch.optim.Adam([*encoder.parameters()] + [*regressor.parameters()],
                                 lr=args.lr, weight_decay=args.weight_decay)

    epoch_best = 0
    rho_s_best = 0.0
    rho_p_best = 0.0
    for epoch in range(args.num_epochs):
        log_and_print(base_logger, f'Epoch: {epoch}')

        for split in ['train', 'test']:
            true_scores = []
            pred_scores = []

            if split == 'train':
                encoder.train()
                regressor.train()
                torch.set_grad_enabled(True)
            else:
                encoder.eval()
                regressor.eval()
                torch.set_grad_enabled(False)

            for data in tqdm(dataloaders[split]):
                true_scores.extend(data['true_score'].numpy())

                image = data['generated_image'].to(device)  # B, C, H, W
                image_prompt = data['prompt_image'].to(device)
                input_ids = data['input_ids'].to(device).unsqueeze(-1)
                text_prompt = data['prompt'].to(device)
                mask = data['attention_mask'].to(device)
                if args.using_image_prompt == 1:
                    total_image = torch.cat([image, image_prompt], dim=0)
                    feature = encoder(args, total_image, input_ids, text_prompt, mask)
                else:
                    feature = encoder(args, image, input_ids, text_prompt, mask)
                preds = regressor(feature).view(-1)

                pred_scores.extend([i.item() for i in preds])

                if split == 'train':
                    loss = criterion(preds, data['true_score'].float().to(device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            rho_s, _ = stats.spearmanr(pred_scores, true_scores)
            rho_p, _ = stats.pearsonr(pred_scores, true_scores)

            log_and_print(base_logger, f'{split} spearmanr_correlation: {rho_s}, pearsonr_correlation: {rho_p}')

        if rho_s > rho_s_best:
            rho_s_best = rho_s
            epoch_best = epoch
            log_and_print(base_logger, '##### New best correlation #####')
            # path = 'ckpts/' + str(rho) + '.pt'
            path = 'ckpts/' + 'best_model.pt'
            torch.save({'epoch': epoch,
                        'encoder': encoder.state_dict(),
                        'regressor': regressor.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'rho_best': rho_s_best}, path)
        if rho_p > rho_p_best:
            rho_p_best = rho_p
        log_and_print(base_logger, ' EPOCH_best: %d, SRCC_best: %.6f, PLCC_best: %.6f' % (epoch_best, rho_s_best, rho_p_best))

