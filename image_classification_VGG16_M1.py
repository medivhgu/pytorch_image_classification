import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import torchvision_datasets_lister

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))
model_names = model_names + ['VGG16_M', 'VGG16_M0', 'VGG16_M1']


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('rootdir', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7', type=str, metavar='GPUs',
                    help='which GPU devices will be used (default: 0,1,2,3,4,5,6,7)')
parser.add_argument('--train-list-file', '--tlf', default='./train_list.txt', type=str, metavar='PATH',
                    help='path of train_list_file (default: ./train_list.txt)')
parser.add_argument('--val-list-file', '--vlf', default='./val_list.txt', type=str, metavar='PATH',
                    help='path of val_list_file (default: ./val_list.txt)')
parser.add_argument('--train-batch-size', '--tbs', default=256, type=int, metavar='N',
                    help='mini-train batch size (default: 256)')
parser.add_argument('--val-batch-size', '--vbs', default=50, type=int, metavar='N',
                    help='mini-val batch size (default: 50)')
parser.add_argument('--resize', default='256,256', type=str, metavar='M[,N]',
                    help='input images need to be resized into M x M*L/S or M x N (default: 256,256)')
parser.add_argument('--cropsize', default='224,244', type=str, metavar='H[,W]',
                    help='crop_size is H x H or H x W (default: 224,224)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--optim-mode', default='SGD', type=str, metavar='TYPE',
                    help='the type of optimizer (default: SGD)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run (default: 90)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts) (default: 0)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                    help='initial learning rate (default: 0.1)')
parser.add_argument('--lr-policy', default='step', type=str, metavar='TYPE',
                    help='descend policy of learning rate (default: step)')
parser.add_argument('--stepsize', default='20', type=str, metavar='N[,N2,N3]',
                    help='the stepsize, only one-N is Step but multi-Ns is MultiStep (default: 20)')
parser.add_argument('--gamma', default=0.1, type=float, metavar='GAMMA',
                    help='descent coefficient of learning rate (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='WD',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N',
                    help='print frequency (default: 100)')
parser.add_argument('--snapshot-prefix', default='', type=str, metavar='N',
                    help='prefix of checkpoint files (default: None)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='fine tune pre-trained model')

best_prec1 = 0
best_epoch = -1

class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, num_classes):
        super(FineTuneModel, self).__init__()

        if arch.startswith('alexnet'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'alexnet'
        elif arch.startswith('resnet'):
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1]) #unknown
            in_features = list(original_model.children())[-1].in_features
            self.classifier = nn.Sequential(
                nn.Linear(in_features, num_classes)
            )
            self.modelName = 'resnet'
        elif arch.startswith('densenet'):
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            in_features = list(original_model.children())[-1].in_features
            self.classifier = nn.Sequential(
                nn.Linear(in_features, num_classes)
            )
            self.modelName = 'densenet'
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.classifier = nn.Sequential(*list(original_model.classifier)[:-1])
            self.classifier_fc8 = nn.Sequential(
                nn.Linear(4096, num_classes),
            )
            #self.classifier = nn.Sequential(
            #    nn.Linear(512 * 7 * 7, 4096),
            #    nn.ReLU(inplace=True),
            #    nn.Dropout(),
            #    nn.Linear(4096, 4096),
            #    nn.ReLU(inplace=True),
            #    nn.Dropout(),
            #    nn.Linear(4096, num_classes),
            #)
            #self.initialize_weights_local(self.classifier_fc8)
            self.modelName = 'vgg16'
        elif arch == 'VGG16_M':
            self.features = nn.Sequential(*list(original_model.features)[:-1])
            self.classifier = nn.Linear(512 * 28 * 28, num_classes)
            self.modelName = 'VGG16_M'
            #self.initialize_weights_local(self.classifier)
        elif arch == 'VGG16_M0':
            self.features = nn.Sequential(*list(original_model.features)[:-1])
            self.pool5_avg = nn.Sequential(
                nn.AvgPool2d(kernel_size=28, stride=28),
                nn.BatchNorm2d(512),
            )
            self.classifier = nn.Linear(512, num_classes)
            self.modelName = 'VGG16_M0'
            #self.initialize_weights_local(self.pool5_avg)
            #self.initialize_weights_local(self.classifier)
        elif arch == 'VGG16_M1':
            self.features = nn.Sequential(*list(original_model.features)[:-1])
            dim = 28
            self.left_pool = nn.Sequential(
                nn.AvgPool2d(kernel_size=dim, stride=dim),
                nn.BatchNorm2d(512),
            )
            self.right_pool = nn.Sequential(
                nn.MaxPool2d(kernel_size=dim, stride=dim),
                nn.BatchNorm2d(512),
            )
            self.classifier = nn.Linear(1024, num_classes)
            self.modelName = 'VGG16_M1'
            #self.initialize_weights_local(self.left_pool)
            #self.initialize_weights_local(self.right_pool)
            #self.initialize_weights_local(self.classifier)
        else:
            raise("Finetuning not supported on this architecture yet")

        ## Freeze those weights 
        #for p in self.features.parameters():
        #    p.requires_grad = false


    def forward(self, x):
        f = self.features(x)
        if self.modelName == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
        elif self.modelName == 'vgg16':
            f = f.view(f.size(0), -1)
        elif self.modelName == 'VGG16_M':
            f = f.view(f.size(0), -1)
        elif self.modelName == 'VGG16_M0':
            f = self.pool5_avg(f)
            f = f.view(f.size(0), -1)
        elif self.modelName == 'VGG16_M1':
            x1 = self.left_pool(f)
            x2 = self.right_pool(f)
            f = torch.cat([x1, x2], 1)
            f = f.view(f.size(0), -1)
        elif self.modelName == 'resnet':
            f = f.view(f.size(0), -1)
        elif self.modelName == 'densenet':
            out = F.relu(f, inplace=True)
            f = F.avg_pool2d(out, kernel_size=7, stride=1).view(f.size(0), -1)
        y = self.classifier(f)
        if self.modelName == 'vgg16':
            y = self.classifier_fc8(y)
        return y


    def initialize_weights_local(self, block):
        for m in block.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def main():
    global args, best_prec1
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print("Using GPUs {0}, PID = {1}".format(args.gpus, os.getpid()))

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_lister = torchvision_datasets_lister.ImageLister(args.rootdir, args.train_list_file,
        transforms.Compose([
            transforms.Scale(string2list(args.resize)),
            #transforms.RandomCrop(string2list(args.cropsize)),
            transforms.RandomSizedCrop(string2list(args.cropsize)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_lister, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    val_lister = torchvision_datasets_lister.ImageLister(args.rootdir, args.val_list_file,
        transforms.Compose([
            transforms.Scale(string2list(args.resize)),
            transforms.CenterCrop(string2list(args.cropsize)),
            transforms.ToTensor(),
            normalize,
        ])
    )
    val_loader = torch.utils.data.DataLoader(
        val_lister, batch_size=args.val_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    num_classes = train_lister.get_num_classes()
    print("num_classes = {}".format(num_classes))
    print("resize_image = {}, cropsize_image = {}".format(string2list(args.resize), string2list(args.cropsize)))
    print("train_batch_size = {0}, val_batch_size = {1}".format(args.train_batch_size, args.val_batch_size))

    # create model
    if args.finetune:
        print("=> using pre-trained model '{}'".format(args.arch))
        if args.arch.startswith('VGG16_M'):
            original_model = models.__dict__['vgg16'](pretrained=True)
        else:
            original_model = models.__dict__[args.arch](pretrained=True)
        model = FineTuneModel(original_model, args.arch, num_classes)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.pretrained:
            model = models.__dict__[args.arch](pretrained=True)
        else:
            model = models.__dict__[args.arch](num_classes=num_classes)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint for new initial weights or resume previous training
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.start_epoch == -1:
                args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}, Prec@1 {:.3f})".format(
                  args.resume, checkpoint['epoch'], best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.optim_mode == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optim_mode == 'RMSProp':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
                                args.lr)
    else:
        optimizer = None
        return
    print("Solver_Type = '{0}', lr = {1}, momentum = {2}, weight_decay = {3}".format(
            args.optim_mode, args.lr, args.momentum, args.weight_decay))

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    import torch.optim.lr_scheduler as lr_scheduler
    if args.lr_policy == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, string2list(args.stepsize), args.gamma)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, string2list(args.stepsize), args.gamma)
    else:
        scheduler = None
        return
    print("lr_policy = '{0}', stepsize = {1}, gamma = {2}".format(args.lr_policy, args.stepsize, args.gamma))
    print("start_epoch = {0}, total_epoch = {1}\n".format(args.start_epoch, args.epochs))
    
    for epoch in range(0, args.epochs):
        #adjust_learning_rate(optimizer, epoch)
        scheduler.step()
        if epoch < args.start_epoch:
            continue

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            best_epoch = epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, args.snapshot_prefix)
    print("Best_Prec@1: {:.3f} (at {} epoch)".format(best_prec1, best_epoch))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    print('Train: ' + format_time() + '  ' + str(os.getpid()))
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i == 0 or (i+1) % args.print_freq == 0:
            print('    Epoch: [{0}][{1}/{2}]\t'
                  'LR {3}\t'
                  'Time {batch_time.val:.3f}({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f}({data_time.avg:.3f})\n    '
                  'Loss {loss.val:.4f}({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f}({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f}({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), round(optimizer.state_dict()['param_groups'][0]['lr'], 8),
                      batch_time=batch_time, data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    print('Val:   ' + format_time() + '  ' + str(os.getpid()))
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        """
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time,
                      loss=losses, top1=top1, top5=top5))
        """
    batch_time.update(time.time() - end)
    print('    *** Time {batch_time.val:.3f} ***  Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} ***'.format(
          batch_time=batch_time, top1=top1, top5=top5))
    return top1.avg


def save_checkpoint(state, is_best, snapshot_prefix, filename='checkpoint.pth.tar'):
    filename = snapshot_prefix + '_' + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, snapshot_prefix + '_model_best.pth.tar')


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if len(output[0]) < topk[1]:
        topk = (1, len(output[0]))
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def string2list(ori_str='20,40,60,80'):
    list1 = [int(item) for item in ori_str.split(',')]
    if len(list1) > 1:
        return list1
    else:
        return list1[0]


def format_time(date=-1):
    if date == -1:
        date=time.time()
    time_formatted = time.strftime("%Y-%m-%d %H:%M:", time.localtime(date))
    return "{0}{1:0>9.6f}".format(time_formatted, date % 60)


if __name__ == '__main__':
    main()
