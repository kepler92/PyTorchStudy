import torch
from torch import nn, optim
from torch.nn import functional
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

import os
import logging

import denception


total_epoch = 350
last_epoch = 0

batch_size = 48
learning_rate = 0.1
learning_milestones = [60, 120, 180, 240, 300, ]
learning_gamma = 0.5
weight_decay = 1e-4

test_per_epoch = 5

model_initial = False
if model_initial:
    input("Initialize?")

model_config = (6, 12, 36, 24)
model_growth_rate = 32
model_init_features = model_growth_rate * 2

model_dir = 'model'
model_name = 'densenet_k{}{}'.format(model_growth_rate, model_config)
data_dir = '../dataset/imagenet_smallset'


LOG = logging.getLogger("Training")
LOG.setLevel(logging.DEBUG)

LOG_Format = logging.Formatter('[%(levelname)s] %(asctime)s > %(message)s')

LOG_FileHandler = logging.FileHandler('./logfile_{}'.format(model_name))
LOG_FileHandler.setFormatter(LOG_Format)
LOG.addHandler(LOG_FileHandler)

# LOG_StreamHandler = logging.StreamHandler()
# LOG_StreamHandler.setFormatter(LOG_Format)
# LOG.addHandler(LOG_StreamHandler)

torch.cuda.set_device(1)

if os.path.exists(model_dir) is False:
    os.mkdir(model_dir)

model_path = os.path.join(model_dir, model_name)

if os.path.exists(model_path) is False:
    os.mkdir(model_path)


# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#
# transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.ToTensor(),
#     normalize,
# ])
#
# train_set = datasets.CIFAR10('dataset/cifar10', train=True, transform=transform, download=True)
# test_set = datasets.CIFAR10('dataset/cifar10', train=False, transform=transform, download=True)
#
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
# test_loader = DataLoader(test_set, batch_size=batch_size//3, shuffle=True, drop_last=False)

transforms_Normalize = transforms.Normalize(mean = [0.49139968, 0.48215827, 0.44653124],
                                            std = [0.24703233, 0.24348505, 0.26158768])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms_Normalize,
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms_Normalize,
])


train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=test_transform)

# train_set = datasets.CIFAR10('../dataset/cifar10', train=True, transform=train_transform, download=True)
# val_set = datasets.CIFAR10('../dataset/cifar10', train=False, transform=test_transform, download=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size//2, shuffle=True)

net = None
saved_model = []

for file in os.listdir(model_path):
    if file.endswith(".pth"):
        file_path = os.path.join(model_path, file)
        if model_initial:
            os.remove(file_path)
        else:
            saved_model.append(file_path)


if len(saved_model) > 0:
    sort_model = [int(os.path.split(os.path.splitext(model)[0])[-1]) for model in saved_model]
    sort_model.sort()
    last_epoch = sort_model[-1]
    latest_model = "{}.pth".format(os.path.join(model_path, str(last_epoch)))

    net = torch.load(latest_model)
    net = net.cuda()
    LOG.info("Load model: {}".format(latest_model))

else:
    net = models.DenseNet(num_init_features=model_init_features, growth_rate=model_growth_rate,
                              block_config=model_config, num_classes=10)
    # net = nn.DataParallel(net)
    net = net.cuda()
    LOG.info("Initialize model")


loss_func = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9, nesterov=True)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=learning_milestones, gamma=learning_gamma)
scheduler.last_epoch = last_epoch - 2


def train(model, loader, optm):
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = Variable(data), Variable(target)
        data, target = data.cuda(), target.cuda()

        optm.zero_grad()
        output = model.forward(data)
        loss = loss_func(output, target)
        loss.backward()
        optm.step()

        total_loss += loss.data[0]
        LOG.debug("\t- Batch: {}/{}, Loss: {}".format(int(batch_idx), len(loader), loss.data[0]))

        if batch_idx % 100 == 0:
            LOG.info("\t- Batch: {}/{}, Loss: {}".format(int(batch_idx), len(loader), loss.data[0]))
    LOG.info("\t> Average Loss: {}".format(total_loss / len(loader)))


def test(model, loader):
    incorrect = 0
    for test_data, test_target in loader:
        test_data, test_target = Variable(test_data), Variable(test_target)
        test_data, test_target = test_data.cuda(), test_target.cuda()
        test_output = model.forward(test_data)
        pred = test_output.data.max(1)[1]
        incorrect += pred.ne(test_target.data).cpu().sum()
    LOG.info("\t> Error: {0}".format(100.0 * incorrect / len(loader.dataset)))


for epoch in range(last_epoch+1, total_epoch+1):
    LOG.info("Epoch: {}".format(epoch))
    scheduler.step()

    for param_group in scheduler.optimizer.param_groups:
        LOG.info("\t> Learning Rate: {}".format(param_group['lr']))

    train(net, train_loader, optimizer)
    # test(val_loader, net)

    if epoch % test_per_epoch == 0:
        test(net, val_loader)

    save_model_name = "{}.pth".format(os.path.join(model_path, str(epoch)))
    torch.save(net, save_model_name)
    LOG.info("\t> Save model: {}".format(save_model_name))
