# piecewise-quantization
PyTorch implementation of Near-Lossless Post-Training Quantization of Deep Neural Networks via a Piecewise Linear Approximation


## Usage
There are 5 main arguments
  1. quantize: whether to quantize parameters(per-channel) and activations(per-tensor).  
  2. imagenet_path: path to folder contains train/val folder of  imagenet data
  3. model: the type of model, should be one of ['mobilenetv2', 'resnet50', 'inceptionv3'], default to mobilenetv2
  4. qtype: the type of quantization for weights, should be one of ['uniform', 'pws', 'pwg', 'pwl'], default to uniform
  5. bits_weight: number of bits for weight quantization, default to 8

run the 4-bits quantized pws mobilenetv2 model by:
```
python main_cls.py --quantize --qtype pws --model mobilenetv2 --bits_Weight 4
```


## TODO
- [x] Uniform quantization
- [x] PWS quantization
- [ ] PWG quantization
- [ ] PWL quantization
- [ ] detection model
- [ ] segmentation model