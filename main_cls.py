import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from utils.layer_transform import switch_layers, set_quant_minmax_data, merge_batchnorm, quantize_targ_layer
from PyTransformer.transformers.torchTransformer import TorchTransformer

from utils.quantize import QuantNConv2d, QuantNLinear, QuantAdaptiveAvgPool2d, QuantMaxPool2d, QuantMeasure, set_layer_bits, set_update_stat
def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", action='store_true')
    parser.add_argument("--imagenet_path", default='/home/jakc4103/windows/Toshiba/workspace/dataset/ILSVRC/Data/CLS-LOC')
    parser.add_argument("--model", type=str, default='mobilenetv2', help='One of mobilenetv2, resnet50, inceptionv3')
    parser.add_argument("--qtype", type=str, default='uniform', help='One of uniform, pwg, pwl, pws')
    parser.add_argument("--log", action='store_true')

    parser.add_argument("--bits_weight", type=int, default=8)
    parser.add_argument("--bits_activation", type=int, default=8)
    return parser.parse_args()


def inference_all(model, path):
    print("Start inference")
    imagenet_dataset = datasets.ImageFolder(path, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ]))

    dataloader = DataLoader(imagenet_dataset, batch_size=256, shuffle=False, num_workers=4)

    num_correct = 0
    num_total = 0
    with torch.no_grad():
        for ii, sample in enumerate(dataloader):
            image, label = sample[0].cuda(), sample[1].numpy()
            # image, label = sample[0], sample[1].numpy()
            logits = model(image)

            pred = torch.max(logits, 1)[1].cpu().numpy()
            
            num_correct += np.sum(pred == label)
            num_total += image.shape[0]
            print(num_correct, num_total, num_correct/num_total)
            
    acc = num_correct / num_total
    return acc

class QuantModel(torch.nn.Module):
    def __init__(self, base_model, bits_activation=8, momentum=0.1):
        super(QuantModel, self).__init__()
        self.base_model = base_model
        self.quant = QuantMeasure(bits_activation, momentum)

    def forward(self, x):
        x = self.base_model(x)
        x = self.quant(x)

        return x

def main():
    args = get_argument()

    data = torch.ones((4, 3, 224, 224))#.cuda()

    if args.model == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif args.model == 'inceptionv3':
        model = models.inception_v3(pretrained=True)
    elif args.model == 'mobilenetv2':
        from modeling.classification import MobileNetV2
        model = MobileNetV2.mobilenet_v2(pretrained=True)
    else:
        assert False, 'Model type not supported'

    model = QuantModel(model, args.bits_activation)

    model.eval()

    transformer = TorchTransformer()
    module_dict = {}
    if args.quantize:
        module_dict[1] = [(nn.Conv2d, QuantNConv2d),\
                            (nn.Linear, QuantNLinear),\
                            (nn.AdaptiveAvgPool2d, QuantAdaptiveAvgPool2d),\
                            (nn.MaxPool2d, QuantMaxPool2d)]

    # transformer.summary(model, data)
    # transformer.visualize(model, data, 'graph_cls', graph_size=120)

    model, transformer = switch_layers(model, transformer, data, module_dict, ignore_layer=[QuantMeasure], quant_op=args.quantize)

    graph = transformer.log.getGraph()
    bottoms = transformer.log.getBottoms()
    if args.quantize:
        targ_layer = [QuantNConv2d, QuantNLinear]
    else:
        targ_layer = [nn.Conv2d, nn.Linear]

    model = merge_batchnorm(model, graph, bottoms, targ_layer)

    if args.quantize:
        set_layer_bits(graph, args.bits_weight, args.bits_activation, targ_type=targ_layer)

    if args.quantize:
        print("preparing data for computing activation min/max range")
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
        if not os.path.exists("_512_train.txt"):
            print("Creating _512_train.txt, this will take some time...")
            from utils import default_loader
            imagenet_dataset = datasets.ImageFolder(os.path.join(args.imagenet_path, 'train'), trans, loader=default_loader)

            np.random.seed(1000)
            perm_idx = np.random.permutation(len(imagenet_dataset))
            images = []
            for i in range(512):
                images.append(imagenet_dataset[perm_idx[i]][0].unsqueeze(0))
            
            del imagenet_dataset
        else:
            from PIL import Image
            images = []
            for line in open("_512_train.txt", 'r'):
                line = line.strip()
                with open(line, 'rb') as f:
                    img = Image.open(f)
                    img = img.convert('RGB')

                images.append(trans(img).unsqueeze(0))

        set_update_stat(model, True)
        model = set_quant_minmax_data(model, images, [QuantMeasure])
        set_update_stat(model, False)

        graph = quantize_targ_layer(graph, args.bits_weight, targ_type=targ_layer, quant_type=args.qtype)

    model = model.cuda()
    model.eval()

    acc = inference_all(model, os.path.join(args.imagenet_path, 'val'))
    print("Acc: {}".format(acc))

    if args.log:
        with open("cls_result.txt", 'a+') as ww:
            ww.write("model: {}, quant: {}, qtype: {}, bits_weight: {}, correction: {}\n".format(
                args.model, args.quantize, args.qtype, args.bits_weight, args.correction
            ))
            ww.write("Acc: {}\n\n".format(acc))


if __name__ == '__main__':
    main()