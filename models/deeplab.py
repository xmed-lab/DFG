import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.models.segmentation.segmentation import my_load_model
from torchvision.models.segmentation import deeplabv3_resnet50
import sys
'''
def my_deeplabv3_resnet50(pretrained=False, progress=True,
                       num_classes=21, aux_loss=None, only_feature=True, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return my_load_model('deeplabv3', 'resnet50', pretrained, progress, num_classes, aux_loss, only_feature, **kwargs)
'''
class my_deeplabv3_resnet50(nn.Module):
    def __init__(self, num_classes):
        super(my_deeplabv3_resnet50, self).__init__()
        model = deeplabv3_resnet50(num_classes = num_classes)
        self.backbone = model.backbone
        mod = list(model.classifier.children())
        self.outc = mod.pop()
        self.classifier = nn.Sequential(*mod)
        
    
    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        #result = OrderedDict()
        x = features["out"]
        fea = self.classifier(x)
        x = self.outc(fea)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        
        return fea, x
    
if __name__ == '__main__':
    #model = my_deeplabv3_resnet50(num_classes=2,only_feature=True)
    model = my_deeplabv3_resnet50(num_classes = 2)
    #mod = list(model.classifier.children())
    #last = mod.pop()
    #print(model.outc)

    device = torch.device('cuda:1')
    model = model.to(device)
    img = torch.rand((4,3,256,256)).to(device)
    out = model(img)
    print(out[0].shape,out[1].shape)