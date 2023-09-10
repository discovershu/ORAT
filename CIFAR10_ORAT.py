import os
import argparse
import torchvision
import torch.optim as optim
from torchvision import transforms
import datetime
from models import *
from earlystop import earlystop
import numpy as np
from utils import Logger
import attack_generator as attack
import itertools
from utils.convert_to_data_loader import dataloader_generation



parser = argparse.ArgumentParser(description='PyTorch AoRR Adversarial Training')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train, 120 for WRN, 100 for resnet, 50 for lenet')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epsilon', type=float, default=0.0078, help='perturbation bound， 0.0078, 0.031 for CIFAR, 0.1, 0.3 for MNIST')
parser.add_argument('--num_steps', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--step_size', type=float, default=0.0078/4, help='step size')
parser.add_argument('--seed', type=int, default=7, metavar='S', help='random seed')
parser.add_argument('--net', type=str, default="smallcnn",
                    help="decide which network to use,choose from smallcnn,smallcnn_for_mnist,resnet18,WRN,WRN_madry,lenet_mnist")
parser.add_argument('--dataset', type=str, default="mnist_noise_asym_40", help="choose from cifar10,svhn,mnist,mnist_noise_sym_20")
parser.add_argument('--rand_init', type=bool, default=True, help="whether to initialize adversarial sample with random noise")
parser.add_argument('--depth', type=int, default=32, help='WRN depth')
parser.add_argument('--width_factor', type=int, default=10, help='WRN width factor')
parser.add_argument('--drop_rate', type=float, default=0.0, help='WRN drop rate')
parser.add_argument('--out_dir', type=str, default='./AoRRAT_results', help='dir of output')
parser.add_argument('--resume', type=str, default='', help='whether to resume training, default: None')
parser.add_argument('--gpuid', type=str, default='2', help='GPU ID')
parser.add_argument('--k', type=int, default=50000, help='k')
parser.add_argument('--m', type=int, default=500, help='m')
parser.add_argument('--aorr', type=bool, default=True, help="use aorr or not")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
out_str = str(args)
print(out_str)

# training settings
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

lambda_k = Variable(torch.Tensor([0]), requires_grad=True).cuda()
lambda_hat = Variable(torch.Tensor([0]), requires_grad=True).cuda()
intial_loss=[]

def train(epoch, model, train_loader, optimizer):
    starttime = datetime.datetime.now()
    loss_sum = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        # Get adversarial training data via PGD
        output_adv = attack.pgd(model, data, target, epsilon=args.epsilon, step_size=args.step_size, num_steps=args.num_steps, loss_fn='cent', category="Madry",rand_init=True)
        # output_adv = data

        model.train()
        optimizer.zero_grad()
        output = model(output_adv)

        if args.aorr:
            if epoch==0:
                intial_loss.append(nn.CrossEntropyLoss(reduction='none')(output, target).cpu().detach().numpy().tolist())
                # calculate standard adversarial training loss
                loss = nn.CrossEntropyLoss(reduction='mean')(output, target)
                loss_sum += loss.item()
                loss.backward()
                optimizer.step()
            else:
                if epoch==1 and batch_idx==0:
                    lambda_k.data = torch.topk(torch.from_numpy(np.asarray(list(itertools.chain(*intial_loss)), dtype=np.float32)).cuda(), args.k, sorted=True, dim=0)[0][-1].data.flatten().cuda()
                    lambda_hat.data = torch.topk(torch.from_numpy(np.asarray(list(itertools.chain(*intial_loss)), dtype=np.float32)).cuda(), args.m, sorted=True, dim=0)[0][
                        -1].data.flatten().cuda()

                loss_term_1 = (args.k-args.m) * lambda_k/(len(train_loader.dataset.data))
                loss_term_2 = (len(train_loader.dataset.data) - args.m) * lambda_hat/(len(train_loader.dataset.data))
                cr_loss = nn.CrossEntropyLoss(reduction='none')(output, target)
                loss_term_3 = cr_loss - lambda_k
                loss_term_3[loss_term_3 < 0] = 0
                loss_term_3 = lambda_hat - loss_term_3
                loss_term_3[loss_term_3 < 0] = 0
                loss = loss_term_1 + loss_term_2 - loss_term_3
                loss = torch.mean(loss)
                loss_sum += loss.item()
                optimizer.zero_grad()
                lambda_k.retain_grad()
                lambda_hat.retain_grad()
                loss.backward()
                optimizer.step()
                lambda_k.data = lambda_k.data - args.lr *lambda_k.grad.data
                lambda_hat.data = lambda_hat.data + args.lr *lambda_hat.grad.data
                lambda_k.grad.data.zero_()
                lambda_hat.grad.data.zero_()
        else:
            # calculate standard adversarial training loss
            loss = nn.CrossEntropyLoss(reduction='mean')(output, target)

            loss_sum += loss.item()
            loss.backward()
            optimizer.step()

    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds

    return time, loss_sum


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 30:
        lr = args.lr * 0.1
    if epoch >= 60:
        lr = args.lr * 0.01
    if epoch >= 110:
        lr = args.lr * 0.005
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, checkpoint=out_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

print('==> Load Test Data')
if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
if args.dataset == "svhn":
    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
if args.dataset == "mnist":
    trainset = torchvision.datasets.MNIST(root='./data/MNIST', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, pin_memory=True)
    testset = torchvision.datasets.MNIST(root='./data/MNIST', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, pin_memory=True)
if 'mnist_noise' in args.dataset:
    trainset, testset = dataloader_generation(
        data_path='./data/mnist_noise_data/{}.mat'.format(args.dataset))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    test_loader_aa = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False)
if 'cifar_10_noise' in args.dataset:
    trainset, testset = dataloader_generation(
        data_path='./data/cifar_10_noise_data/{}.mat'.format(args.dataset))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    test_loader_aa = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False)

print('==> Load Model')
if args.net == "smallcnn":
    model = SmallCNN().cuda()
    net = "smallcnn"
if args.net == "smallcnn_for_mnist":
    model = SmallCNN_for_mnist().cuda()
    net = "smallcnn_for_mnist"
if args.net == "lenet_mnist":
    model = LeNet_mnist().cuda()
    net = "lenet_mnist"
if args.net == "resnet18":
    model = ResNet18().cuda()
    net = "resnet18"
if args.net == "WRN":
  # e.g., WRN-34-10
    model = Wide_ResNet(depth=args.depth, num_classes=10, widen_factor=args.width_factor, dropRate=args.drop_rate).cuda()
    net = "WRN{}-{}-dropout{}".format(args.depth, args.width_factor, args.drop_rate)
if args.net == 'WRN_madry':
  # e.g., WRN-32-10
    model = Wide_ResNet_Madry(depth=args.depth, num_classes=10, widen_factor=args.width_factor, dropRate=args.drop_rate).cuda()
    net = "WRN_madry{}-{}-dropout{}".format(args.depth, args.width_factor, args.drop_rate)
print(net)

model = torch.nn.DataParallel(model)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

start_epoch = 0
# Resume
title = 'AoRRAT train'
if args.resume:
    # resume directly point to checkpoint.pth.tar e.g., --resume='./out-dir/checkpoint.pth.tar'
    print('==> AoRR Adversarial Training Resuming from checkpoint ..')
    print(args.resume)
    assert os.path.isfile(args.resume)
    out_dir = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger_test = Logger(os.path.join(out_dir, 'log_results.txt'), title=title, resume=True)
else:
    print('==> AoRR Adversarial Training')
    logger_test = Logger(os.path.join(args.out_dir, 'log_results.txt'), title=title)
    logger_test.set_names(['Epoch', 'Natural Test Acc', 'FGSM Acc', 'PGD20 Acc', 'CW Acc', 'AA Acc'])

test_nat_acc = 0
fgsm_acc = 0
test_pgd20_acc = 0
cw_acc = 0
best_epoch = 0
best_natural=0
best_fsgm=0
best_pgd20=0
best_cw=0
best_aa=0
for epoch in range(start_epoch, args.epochs):
    adjust_learning_rate(optimizer, epoch + 1)
    train_time, train_loss= train(epoch, model, train_loader, optimizer)

    ## Evalutions the same as DAT.
    loss, test_nat_acc = attack.eval_clean(model, test_loader)
    loss, fgsm_acc = attack.eval_robust(model, test_loader, perturb_steps=1, epsilon=args.epsilon, step_size=args.epsilon,loss_fn="cent", category="Madry",rand_init=True)
    loss, test_pgd20_acc = attack.eval_robust(model, test_loader, perturb_steps=20, epsilon=args.epsilon, step_size=args.epsilon / 4,loss_fn="cent", category="Madry", rand_init=True)
    loss, cw_acc = attack.eval_robust(model, test_loader, perturb_steps=30, epsilon=args.epsilon, step_size=args.epsilon / 4,loss_fn="cw", category="Madry", rand_init=True)
    loss, aa_acc = attack.eval_robust_aa(model, test_loader_aa, epsilon=args.epsilon, step_size=args.epsilon)

    print(
        'Epoch: [%d | %d] | Train Time: %.2f s | train_loss: %.4f | Natural Test Acc %.4f | FGSM Test Acc %.4f | PGD20 Test Acc %.4f | CW Test Acc %.4f |AA Test Acc %.4f |\n' % (
            epoch + 1,
            args.epochs,
            train_time,
            train_loss,
            test_nat_acc,
            fgsm_acc,
            test_pgd20_acc,
            cw_acc,
            aa_acc)
    )

    if best_natural < test_nat_acc:
        best_natural = test_nat_acc
    if best_fsgm < fgsm_acc:
        best_fsgm = fgsm_acc
    if best_pgd20 < test_pgd20_acc:
        best_pgd20 = test_pgd20_acc
    if best_cw < cw_acc:
        best_cw = cw_acc
    if best_aa < aa_acc:
        best_aa = aa_acc
    if (epoch + 1) == args.epochs:
        print(
            'Best: | Natural Best Acc %.4f | FGSM Best Acc %.4f | PGD20 Best Acc %.4f | CW Best Acc %.4f |AA Best Acc %.4f |\n' % (
                best_natural,
                best_fsgm,
                best_pgd20,
                best_cw,
                best_aa)
        )

    logger_test.append([epoch + 1, test_nat_acc, fgsm_acc, test_pgd20_acc, cw_acc, aa_acc])

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'test_nat_acc': test_nat_acc,
        'test_pgd20_acc': test_pgd20_acc,
        'optimizer': optimizer.state_dict(),
    })
