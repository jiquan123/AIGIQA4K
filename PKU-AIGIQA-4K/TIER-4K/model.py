import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.squeeze(1).unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings



class Encoder(nn.Module):
    def __init__(self, image_encoder):
        super(Encoder, self).__init__()

        self.text_encoder = AutoModel.from_pretrained("./bert-base-uncased")
        self.image_encoder = image_encoder
        self.pooler = MeanPooling()

    def forward(self, args, x, input_ids, ids, mask):
        out = self.text_encoder(input_ids=ids.squeeze(1), attention_mask=mask,
                         output_hidden_states=False)
        text_features = self.pooler(out.last_hidden_state, mask) #B, D
        if args.using_image_prompt == 1:
            total_feature = self.image_encoder(x)
            feature_1 = total_feature[:total_feature.shape[0] // 2]#.unsqueeze(1)
            feature_2 = total_feature[total_feature.shape[0] // 2:]#.unsqueeze(1)
            if args.benchmark == 'I2I':
                image_features = torch.cat([feature_1, feature_2], dim=1)
            else:
                features = torch.cat([feature_1.unsqueeze(1), feature_2.unsqueeze(1)], dim=1)
                image_features = torch.sum(features * input_ids, dim=1) / torch.sum(input_ids, dim=1)
        else:
            image_features = self.image_encoder(x)
        out_features = torch.cat([text_features, image_features], dim=-1)
        return out_features



class MLP(nn.Module):
    def __init__(self, args, image_hidden_size):
        super(MLP, self).__init__()
        self.config = AutoConfig.from_pretrained("./bert-base-uncased")
        if args.benchmark == 'I2I' and args.using_image_prompt == 1:
            self.image_hidden_size = image_hidden_size * 2
        else:
            self.image_hidden_size = image_hidden_size
        self.fc1 = nn.Linear((self.image_hidden_size + self.config.hidden_size), (self.image_hidden_size  + self.config.hidden_size) // 2)
        self.fc2 = nn.Linear((self.image_hidden_size + self.config.hidden_size) // 2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
