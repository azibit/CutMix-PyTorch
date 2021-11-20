# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py
# python train.py --net_type resnet --dataset cifar10  --batch_size 64 --lr 0.1 --beta 1.0 --cutmix_prob 0.5 --no-verbose --epochs 2 --trials=2 --iterations=2 --dataset_dir=../Datasets2 --image_size 224
# python train.py --net_type resnet --dataset cifar10  --batch_size 64 --lr 0.1
# --beta 1.0 --cutmix_prob 0.5 --no-verbose --epochs 2 --trials=2 --iterations=2 --dataset_dir=../Datasets2 --image_size 224 -v2 cutmix version 2

import argparse, os, glob, sys, shutil, time, torch, csv

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import resnet as RN
import pyramidnet as PYRM
import utils
import numpy as np

import warnings

warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
parser.add_argument('--net_type', default='pyramidnet', type=str,
                    help='networktype: resnet, and pyamidnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')
parser.add_argument('--beta', default=0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0, type=float,
                    help='cutmix probability')
# Add dataset dir path and number of trials
parser.add_argument('--dataset_dir', default='Data', type=str,
                    help='The location of the datasets to be explored')
parser.add_argument('--trials', default=5, type=int,
                    help='Number of times to run the complete experiment')
parser.add_argument('--iterations', default=2, type=int,
                    help='Number of times to run the complete experiment')
parser.add_argument('--image_size', default=32, type=int,
                    help='input image size')
parser.add_argument('--cutmix_v2', '-v2', action='store_true',
                    help='Add a version of mixup that uses original dataset')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

args = parser.parse_args()

cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed(123)

if not os.path.exists(args.dataset_dir):
    os.makedirs(args.dataset_dir)

dataset_list = sorted(glob.glob(args.dataset_dir + "/*"))
print("Dataset List: ", dataset_list)

if len(dataset_list) == 0:
    print("ERROR: 1. Add the Datasets to be run inside of the", args.dataset_dir, "folder")
    sys.exit()

def main(dataset_dir, iteration, trial):
    global args

    best_err1 = 100
    best_err5 = 100

    # 1. Location to save the output for the given dataset
    current_dataset_file = dataset_dir.split("/")[-1] + '_.txt'
    current_exp = "_ite_" + str(iteration) + "_trial_" + str(trial) + "_dataset_" + dataset_dir.split("/")[-1] + "_"

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
        transforms.RandomCrop(args.image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])


    trainset = datasets.ImageFolder(os.path.join(dataset_dir, 'train'),
                                  transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valset = datasets.ImageFolder(os.path.join(dataset_dir, 'test'),
                                  transform_test)
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    numberofclass = len(trainset.classes)


    print("=> creating model '{}'".format(args.net_type))

    if args.image_size == 32:
        model = RN.ResNet18(num_classes = numberofclass)  # for ResNet
    else:
        model = models.densenet161()
        model.classifier = nn.Linear(model.classifier.in_features, len(trainset.classes))

    model = torch.nn.DataParallel(model).cuda()

    results = "results_" + dataset_dir.split("/")[-1]
    if not os.path.isdir(results):
        os.mkdir(results)
    logname = (results + '/log_' + current_exp + '_.csv')

    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'train loss', 'train acc',
                                'test loss', 'test acc'])

    # print(model)
    # print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    cudnn.benchmark = True

    for epoch in range(0, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss, train_err1, train_err5, = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        err1, err5, val_loss = validate(val_loader, model, criterion, epoch)

        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, 100 - train_err1, val_loss, 100 - err1])

        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best:
            best_err5 = err5

            print('Current best accuracy (top-1 and 5 accuracy):', round(100 - best_err1, 3), round(100 - best_err5, 3))
            save_checkpoint({
                'epoch': epoch,
                'arch': args.net_type,
                'state_dict': model.state_dict(),
                'best_err1': best_err1,
                'best_err5': best_err5,
                'optimizer': optimizer.state_dict(),
            }, is_best, current_exp)

        if epoch + 1 == args.epochs:
            with open(current_dataset_file, 'a') as f:
                # Test for the result of the best model
                checkpoint = torch.load('runs/%s/' % (args.expname) + current_exp + 'model_best.pth.tar')
                model.load_state_dict(checkpoint['state_dict'])

                print("Test result for iteration", iteration, "experiment:", trial, " for dataset ", dataset_dir, file = f)
                print(utils.make_prediction(model, valset.classes, val_loader, 'save'), file = f)

                print("Train result for iteration", iteration, "experiment:", trial, " for dataset ", dataset_dir, file = f)
                print(utils.make_prediction(model, valset.classes, train_loader, 'save'), file = f)

    print('Best accuracy (top-1 and 5 accuracy):', round(100 - best_err1, 3), round(100 - best_err5, 3))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        X = input.cuda()

        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:


            if args.cutmix_v2:
                # compute output
                output1 = model(X)
                # loss = criterion(output, target)

            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            output = model(input)
            if args.cutmix_v2:
                loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam) + criterion(output1, target)
            else:
                loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed timeinput
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-acc {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=100 - top1, top5=100 - top5))

    print('* Epoch: [{0}/{1}]\t Top 1-acc {top1:.3f}  Top 5-acc {top5:.3f}\t Train Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=(100 - top1.avg), top5=(100 - top5.avg), loss=losses))

    return losses.avg, top1.avg, top5.avg


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-acc {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-acc {top1:.3f}  Top 5-acc {top5:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=(100 - top1.avg), top5=(100 - top5.avg), loss=losses))
    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, current_exp, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + current_exp + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.expname) + current_exp + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.dataset.startswith('cifar'):
        lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))
    elif args.dataset == ('imagenet'):
        if args.epochs == 300:
            lr = args.lr * (0.1 ** (epoch // 75))
        else:
            lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


# if __name__ == '__main__':
#     main()
for dataset in dataset_list:
    for iteration in range(args.iterations):
        for trial in range(args.trials):
            print("Iteration ", iteration, "Experiment: ", trial, " Dataset: ", dataset)

            main(dataset, iteration, trial)
