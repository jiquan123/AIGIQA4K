import torch
import torch.nn as nn
import os
import sys
from scipy import stats
from tqdm import tqdm
from backbone.resnet import resnet18_backbone, resnet50_backbone
from backbone.inceptionv4 import inceptionv4
from backbone.vgg import vgg16, vgg19
from backbone.vit import ViTExtractor
from regressor import MLP
from dataloader import get_AIGIQA4K_dataloaders, get_AIGIQA4KT2I_dataloaders, get_AIGIQA4KI2I_dataloaders
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

    if not os.path.exists('./exp'):
        os.mkdir('./exp')
    if not os.path.exists('./ckpts'):
        os.mkdir('./ckpts')

    base_logger = get_logger(f'exp/AIGIQA4K.log', args.log_info)

    # print configuration
    print('=' * 40)
    for k, v in vars(args).items():
        #print(f'{k}: {v}')
        log_and_print(base_logger, f'{k}: {v}')
    print('=' * 40)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.backbone == 'resnet50':
        backbone = resnet50_backbone().to(device)
        regressor = MLP(args, 2048).to(device)
    elif args.backbone == 'resnet18':
        backbone = resnet18_backbone().to(device)
        regressor = MLP(args, 512).to(device)
    elif args.backbone == 'vgg16':
        backbone = vgg16().to(device)
        regressor = MLP(args, 512).to(device)
    elif args.backbone == 'vgg19':
        backbone = vgg19().to(device)
        regressor = MLP(args, 512).to(device)
    elif args.backbone == 'vit':
        backbone = ViTExtractor().to(device)
        regressor = MLP(args, 1024).to(device)
    else:
        backbone = inceptionv4(num_classes=1000, pretrained='imagenet').to(device)
        regressor = MLP(args, 1536).to(device)


    if args.benchmark =='AIGIQA4K':
        dataloaders = get_AIGIQA4K_dataloaders(args)
    elif args.benchmark =='I2I':
        dataloaders = get_AIGIQA4KI2I_dataloaders(args)
    else:
        dataloaders = get_AIGIQA4KT2I_dataloaders(args)

    criterion = nn.MSELoss(reduction='mean').cuda()
    # criterion = nn.SmoothL1Loss(reduction='mean')
    optimizer = torch.optim.Adam([*backbone.parameters()] + [*regressor.parameters()],
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
                backbone.train()
                regressor.train()
                torch.set_grad_enabled(True)
            else:
                backbone.eval()
                regressor.eval()
                torch.set_grad_enabled(False)

            for data in tqdm(dataloaders[split]):
                true_scores.extend(data['true_score'].numpy())

                generated_image = data['generated_image'].to(device)  # B, C, H, W
                prompt_image = data['prompt_image'].to(device)
                input_ids = data['input_ids'].to(device).unsqueeze(-1) #B, 2, 1

                if args.using_prompt == 1:
                    total_image = torch.cat([generated_image, prompt_image], dim=0)
                    if args.backbone in ['vgg16', 'vgg19']:
                        total_feature = backbone.features(total_image)
                        total_feature = avgpool(total_feature).squeeze(2).squeeze(2)  # 2*B, C
                        feature_1 = total_feature[:total_feature.shape[0] // 2]
                        feature_2 = total_feature[total_feature.shape[0] // 2:]
                        if args.benchmark == 'I2I':
                            feature = torch.cat([feature_1, feature_2], dim=1)  #FR
                        else:
                            features = torch.cat([feature_1.unsqueeze(1), feature_2.unsqueeze(1)], dim=1)
                            feature = torch.sum(features * input_ids, dim=1) / torch.sum(input_ids, dim=1)
                        preds = regressor(feature).view(-1)
                    else:
                        total_feature = backbone(total_image)  # 2*B, C
                        feature_1 = total_feature[:total_feature.shape[0] // 2]
                        feature_2 = total_feature[total_feature.shape[0] // 2:]
                        if args.benchmark == 'I2I':
                            feature = torch.cat([feature_1, feature_2], dim=1)  #FR
                        else:
                            features = torch.cat([feature_1.unsqueeze(1), feature_2.unsqueeze(1)], dim=1)
                            feature = torch.sum(features * input_ids, dim=1) / torch.sum(input_ids, dim=1)
                        preds = regressor(feature).view(-1)
                else:
                    if args.backbone in ['vgg16', 'vgg19']:
                        feature = backbone.features(generated_image)
                        feature = avgpool(feature).squeeze(2).squeeze(2)  # 2*B, C
                        preds = regressor(feature).view(-1)
                    else:
                        feature = backbone(generated_image)  # 2*B, C
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
                        'backbone': backbone.state_dict(),
                        'regressor': regressor.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'rho_best': rho_s_best}, path)
        if rho_p > rho_p_best:
            rho_p_best = rho_p
        log_and_print(base_logger, ' EPOCH_best: %d, SRCC_best: %.6f, PLCC_best: %.6f' % (epoch_best, rho_s_best, rho_p_best))