# original code: https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py

import torch
import random

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

__all__ = ["Compose", "Lighting", "ColorJitter"]


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class ColorJitter(object):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        self.transforms = []
        if self.brightness != 0:
            self.transforms.append(Brightness(self.brightness))
        if self.contrast != 0:
            self.transforms.append(Contrast(self.contrast))
        if self.saturation != 0:
            self.transforms.append(Saturation(self.saturation))

        random.shuffle(self.transforms)
        transform = Compose(self.transforms)
        # print(transform)
        return transform(img)

def get_pred_as_list(test_preds):

    result = []
    for i in range(len(test_preds)):
        result.append(int(test_preds[i].item()))

    return result

def make_prediction(net, class_names, loader, name_to_save):

    test_loss = 0
    correct = 0
    total = 0
    acc = 0
    all_preds = torch.tensor([]).cuda()
    ground_truths = torch.tensor([]).cuda()
    net.eval()

    for batch_idx, (inputs, targets) in enumerate(loader):
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)

            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            ground_truths = torch.cat(
                  (ground_truths, targets)
                  ,dim=0
              )

            all_preds = torch.cat(
                  (all_preds, predicted)
                  ,dim=0
              )

            acc = 100.*correct/total

    targets = get_pred_as_list(ground_truths)
    preds = get_pred_as_list(all_preds)
    cm = confusion_matrix(targets, preds)
    print(classification_report(targets, preds, target_names=class_names))
