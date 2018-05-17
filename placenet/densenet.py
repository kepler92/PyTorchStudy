import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from place365 import Place365Dataset
from torch.utils.data import DataLoader


class Models:
    def __init__(self):
        self.log_init()

        self.log_print("Model Init")
        model_ft = models.densenet161(pretrained=True)

        for param in model_ft.parameters():
            param.requires_grad = False

        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, 365)

        model_ft = nn.DataParallel(model_ft)

        train_data_transforms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

        test_data_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])

        self.log_print("Data Loader")
        # train_dataset = datasets.ImageFolder(root='/workspace/dataset/awa/image', transform=train_data_transforms)
        # train_dataset = datasets.ImageFolder(root='/workspace/dataset/awa/image', transform=test_data_transforms)
        train_dataset = Place365Dataset(csv_file='/workspace/dataset/place365/places365_train_standard.txt',
                                        image_dir='/workspace/dataset/place365/data_large',
                                        transform=train_data_transforms)
        test_dataset = Place365Dataset(csv_file='/workspace/dataset/place365/places365_val.txt',
                                        image_dir='/workspace/dataset/place365/val_large',
                                        transform=test_data_transforms)

        self.train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=True)

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            model_ft= model_ft.cuda()

        self.model = model_ft

        self.learning_rate = 0.01
        self.learning_milestones = [60, 120, 180, 240, 300, ]
        self.learning_gamma = 0.5
        self.weight_decay = 1e-4

        self.loss_fun = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()),
                                   lr=self.learning_rate, weight_decay=self.weight_decay,
                                   momentum=0.9, nesterov=True)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=self.learning_milestones,
                                                        gamma=self.learning_gamma)

    def log_init(self):
        import logging
        LOG = logging.getLogger("Training")
        LOG.setLevel(logging.DEBUG)

        LOG_Format = logging.Formatter('[%(levelname)s] %(asctime)s > %(message)s')

        LOG_FileHandler = logging.FileHandler('./logfile')
        LOG_FileHandler.setFormatter(LOG_Format)
        LOG.addHandler(LOG_FileHandler)
        self.log = LOG

    def log_print(self, string):
        self.log.info(string)


    def train(self):
        self.scheduler.step()
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            # data, target = Variable(data), Variable(target)
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            self.optimizer.zero_grad()
            output = self.model.forward(data)
            loss = self.loss_fun(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.data.item()
            self.log_print("\t- Batch: {}/{}, Loss: {}".format(int(batch_idx), len(self.train_dataloader), loss.data.item()))
        self.log_print("\t> Average Loss: {}".format(total_loss / len(self.train_dataloader)))

    def test(self):
        incorrect = 0
        for test_data, test_target in self.test_dataloader:
            # test_data, test_target = Variable(test_data), Variable(test_target)
            if torch.cuda.is_available():
                test_data, test_target = test_data.cuda(), test_target.cuda()
            test_output = self.model.forward(test_data)
            pred = test_output.data.max(1)[1]
            incorrect += pred.ne(test_target.data).cpu().sum()
        self.log_print("\t> Error: {0}".format(100.0 * incorrect / len(self.test_dataloader.dataset)))
