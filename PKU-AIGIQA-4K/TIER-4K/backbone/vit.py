import torch
import timm
from timm.models.vision_transformer import Block

class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

class ViTExtractor(torch.nn.Module):
    def __init__(self):
        super(ViTExtractor, self).__init__()
        self.vit = timm.create_model('vit_large_patch16_224', pretrained=True)
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

    def extract_feature(self, save_output):
        x = save_output.outputs[-1][:, 0]
        return x

    def forward(self, x):
        _x = self.vit(x)
        features = self.extract_feature(self.save_output)
        self.save_output.outputs.clear()
        return features

