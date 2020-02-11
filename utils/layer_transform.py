import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

from utils.quantize import QuantNConv2d, QuantNLinear, quantize, QuantMeasure, QuantReLU6, QuantReLU


def switch_layers(model, transformer, data, module_dict, ignore_layer=[], ignore_op=['pad'], quant_op=True):
    # replace layers
    for key in module_dict:
        for source, target in module_dict[key]:
            transformer.register(source, target)
        model = transformer.trans_layers(model, update=True if key == 1 else False)
    transformer._build_graph(model, data, ignore_layer) # construt graph after all state_dict loaded
    # if not quant_op:
    return model, transformer


def merge_batchnorm(model, graph, bottoms, targ_type=[]):
    """!
    This function will merge params and stats of BatchNorm into targ_type like QuantConv2d.
    Once the values is merged, the values of layer will be set to default (as an identity layer),
    and it creates buffer named 'fake_weight' adn 'fake_bias' for latter usage of set_quant_minmax
    """
    with torch.no_grad():
        # merge bn params into QConv2d
        for layer_idx in graph:
            # print(bottoms[layer_idx])
            if bottoms[layer_idx] is None:
                continue
            for bot_idx in bottoms[layer_idx]:
                if type(graph[layer_idx]) == nn.BatchNorm2d and type(graph[bot_idx]) in targ_type:
                    # TODO: suppport gpu version
                    conv_weight = graph[bot_idx].weight.detach()
                    bn_weight = graph[layer_idx].weight.detach()
                    bn_var = graph[layer_idx].running_var.detach()
                    bn_eps = graph[layer_idx].eps

                    graph[bot_idx].weight.copy_(conv_weight.mul(bn_weight.view(-1, 1, 1, 1) / torch.sqrt(bn_var.view(-1, 1, 1, 1) + bn_eps)))

                    if graph[bot_idx].bias is None: # add a bias term to conv or linear layer
                        graph[bot_idx].bias = nn.Parameter(data=torch.zeros((graph[bot_idx].weight.size(0)), dtype=torch.float32), requires_grad=False)

                    conv_bias = graph[bot_idx].bias.detach()
                    bn_bias = graph[layer_idx].bias.detach()
                    bn_mean = graph[layer_idx].running_mean.detach()

                    graph[bot_idx].bias.copy_(conv_bias.mul(bn_weight.view(-1) / torch.sqrt(bn_var.view(-1) + bn_eps)).add(bn_bias.view(-1) -\
						 (bn_weight.view(-1) * bn_mean.view(-1)) / torch.sqrt(bn_var.view(-1) + bn_eps)))

                    # store values for later usage. ex: set_quant_min_max and bias correction
                    graph[layer_idx].register_buffer('fake_weight', torch.abs(bn_weight.clone()))
                    graph[layer_idx].register_buffer('fake_bias', bn_bias.clone())

                    # set batch norm layer to the same to an identity layer
                    graph[layer_idx].weight.fill_(1)
                    graph[layer_idx].running_var.fill_(1)
                    graph[layer_idx].bias.fill_(0)
                    graph[layer_idx].running_mean.fill_(0)
                    graph[layer_idx].eps = 0

                    break

    return model


def quantize_targ_layer(graph, bit_weight=8, targ_type=None, quant_type='uniform'):
    """
    quant_type should be one of ['uniform', 'pwg', 'pwl', 'pws']
    """
    print("Quantizing Layer parameters")
    assert quant_type in ['uniform', 'pwg', 'pwl', 'pws'], "quant_type not supported"
    assert targ_type != None, "targ_type cannot be None!"
    
    for layer_idx in graph:
        if type(graph[layer_idx]) in targ_type:
            with torch.no_grad():
                if quant_type == 'uniform':
                    # quantization behave differently on cpu and gpu
                    param = graph[layer_idx].weight.detach()#.cuda()
                    min_value = param.view(param.size(0), -1).min(-1)[0].view(-1, 1, 1, 1)
                    max_value = param.view(param.size(0), -1).max(-1)[0].view(-1, 1, 1, 1)
                    if len(param.shape) == 2:
                        min_value = min_value.view(-1, 1)
                        max_value = max_value.view(-1, 1)
                    tmp = quantize(param, bit_weight, min_value, max_value)

                    graph[layer_idx].weight.data.copy_(tmp.data.cpu())
                    if graph[layer_idx].bias is not None:
                        param = graph[layer_idx].bias.detach()#.cuda()
                        graph[layer_idx].bias.data.copy_(quantize(param, 8, param.min(), param.max()).data.cpu())
                else:
                    param = graph[layer_idx].weight.detach()#.cuda()
                    m = torch.abs(param).max()

                    if quant_type == 'pws':
                        import numpy as np
                        def ecdf(p, x):
                            idx = np.argwhere(np.sort(x)>=p)
                            if idx.shape[0] == 0:
                                return 1.
                            return (np.arange(1, len(x)+1)/float(len(x)))[idx[0]]
                            
                        expect_error = lambda b, m, p, x: (1/(12*((2 ** b - 1) ** 2))) * ((m - p)**2 + m * (2*p-m) * (2*ecdf(p, x) - 1))

                        e_best = 1e8
                        p1 = 0
                        param_np = param.view(-1).cpu().numpy()
                        m_np = m.cpu().numpy()
                        for rr in np.arange(0.1, 1.0, 0.1):
                            tmp = rr * m_np
                            e_cur = expect_error(bit_weight, m_np, tmp, param_np)

                            if e_cur < e_best:
                                e_best = e_cur
                                p1 = tmp
                        p2 = p1
                        for rr in np.arange((p1/m_np)*0.1, (p1/m_np)+0.1, 0.01):
                            tmp = rr * m_np
                            e_cur = expect_error(bit_weight, m_np, tmp, param_np)

                            if e_cur < e_best:
                                e_best = e_cur
                                p2 = tmp
                        p = p2
                        for rr in np.arange((p2/m_np)*0.1, (p2/m_np)+0.01, 0.001):
                            tmp = rr * m_np
                            e_cur = expect_error(bit_weight, m_np, tmp, param_np)

                            if e_cur < e_best:
                                e_best = e_cur
                                p = tmp
                    else:
                        if quant_type == 'pwg':
                            raise NotImplementedError
                            m = (m - torch.mean(param)) / torch.std(param.view(-1))
                            p = torch.log(0.8614*m+0.6079) * m
                            p = p * torch.std(param.view(-1)) + torch.mean(param)
                        else:
                            raise NotImplementedError
                            p = 0.8030*torch.sqrt(m) - 0.3167

                    # print(p, param.max(), param.min(), param.mean(), param.std())
                    r1 = torch.abs(param).clamp(max=p)
                    r2 = torch.abs(param).clamp(min=p)

                    for rr in [r1, r2]:
                        min_value = rr.view(rr.size(0), -1).min(-1)[0].view(-1, 1, 1, 1)
                        max_value = rr.view(rr.size(0), -1).max(-1)[0].view(-1, 1, 1, 1)
                        if len(rr.shape) == 2:
                            min_value = min_value.view(-1, 1)
                            max_value = max_value.view(-1, 1)

                        rr.data.copy_(quantize(param, bit_weight, min_value, max_value).data)

                    result = torch.zeros_like(param)
                    result[torch.abs(param) < p] = (r1 * torch.sign(param))[torch.abs(param) < p]
                    result[torch.abs(param) >= p] = (r2 * torch.sign(param))[torch.abs(param) >= p]

                    graph[layer_idx].weight.data.copy_(result)

    return graph


def set_quant_minmax_data(model, datas, target=[QuantMeasure]):
    from tqdm import tqdm
    print("SET QUANT MIN MAX USING DATA")
    class ForwardHook():
        def __init__(self):
            self.max_10 = []
            self.min_10 = []
            self.module = None


        def hook(self, module, input, output):
            self.module = module
            input = input[0]
            
            if len(self.max_10) < 10:
                self.max_10.append(input.detach().max())
            else:
                cur = input.detach().max()
                if torch.sum((torch.stack(self.max_10) >= cur).float()) != 10:
                    self.max_10.remove(torch.stack(self.max_10).min())
                    self.max_10.append(cur)

            if len(self.min_10) < 10:
                self.min_10.append(input.detach().min())
            else:
                cur = input.detach().min()
                if torch.sum((torch.stack(self.min_10) <= cur).float()) != 10:
                    self.min_10.remove(torch.stack(self.min_10).max())
                    self.min_10.append(cur)

        def get_results(self):
            return self.module, self.max_10, self.min_10

    hooks = []
    hook_handles = []
    model.eval().cuda()
    for n, m in model.named_modules():
        if type(m) in target:
            hook = ForwardHook()
            hooks.append(hook)
            hook_handles.append(m.register_forward_hook(hook.hook))

    with torch.no_grad():
        # replace_op()
        for data in tqdm(datas):
            _ = model(data.cuda())
            # _ = model(data)
        # restore_op()

    for hook in hooks:
        m, max10, min10 = hook.get_results()
        if m is None:
            continue
        
        m.running_max.copy_(torch.median(torch.stack(max10)))
        m.running_min.copy_(torch.median(torch.stack(min10)))

    for handle in hook_handles:
        handle.remove()

    return model

def mark_relu_relu6(graph, bottoms, model, targ_layer):

    def replace_module(model, source, target):
        for module_name in model._modules:
            if len(model._modules[module_name]._modules) > 0 and id(getattr(model, module_name)) != id(source):
                replace_module(model._modules[module_name], source, target)
            
            elif id(getattr(model, module_name)) == id(source):
                setattr(model, module_name, target)

        return model


    for idx in graph:
        if idx == 'Data':
            continue

        if type(graph[idx]) in [nn.ReLU, nn.ReLU6]:
            while True:
                bot = bottoms[idx][0]
                if bot == 'Data':
                    break
                if type(graph[bot]) in [nn.BatchNorm2d]:
                    pass
                elif type(graph[bot]) == str and 'add' in bot:
                    target = QuantReLU() if type(graph[idx]) == nn.ReLU else QuantReLU6()
                    model = replace_module(model, graph[idx], target)
                    graph[idx] = target
                    break
                elif type(graph[bot]) in targ_layer:
                    graph[bot].ac_fn = 1 if type(graph[idx]) == nn.ReLU else 2
                    break
                else:
                    print(bot, type(graph[bot]))
                    raise NotImplementedError

                idx = bot

    return model