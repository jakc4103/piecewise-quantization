def visualize_per_layer(param, title='test'):
    import matplotlib.pyplot as plt
    channel = 0
    param_list = []
    for idx in range(param.shape[channel]):
        # print(idx, param[idx].max(), param[idx].min())
        param_list.append(param[idx].cpu().numpy().reshape(-1))

    fig7, ax7 = plt.subplots()
    ax7.set_title(title)
    ax7.boxplot(param_list, showfliers=False)
    # plt.ylim(-70, 70)
    plt.show()

def pil_loader(path):
    from PIL import Image
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    """
    load images and output image file name to _512_train.txt
    """
    from torchvision import get_image_backend
    with open("_512_train.txt", "a+") as ww:
        ww.write(path+"\n")
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)