from models.unet import UNet
from models.DeepLabV3Plus.network import deeplabv3plus_resnet50
from .deeplab import my_deeplabv3_resnet50

def get_model(cfg):
    
    if cfg['arch'] == 'UNet':
        model = UNet(n_channels=cfg['input_dim'],n_classes=cfg['num_classes'],only_feature=False)
    elif cfg['arch'] == 'DeepLab':
        #model = deeplabv3plus_resnet50(num_classes=cfg['num_classes'],only_feature=False)
        model = my_deeplabv3_resnet50(num_classes=cfg['num_classes'])

    return model