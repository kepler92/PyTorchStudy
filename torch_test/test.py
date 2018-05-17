import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


class Models:
    def __init__(self):

        print("Model Init")
        model_ft = models.densenet161(pretrained=True)
        # model_ft = models.resnet18(pretrained=True)

        model_ft.parameters()

        for param in model_ft.parameters():
            param.requires_grad = False

        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, 50)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, 50)

        model_ft = nn.DataParallel(model_ft)

        data_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

        print("Data Loader")
        dataset = datasets.ImageFolder(root='/workspace/dataset/awa/image',
                                       transform=data_transforms)

        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

        self.train_dataloader = dataloader
        self.test_dataloader = dataloader

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            model_ft= model_ft.cuda()

        self.model = model_ft
        self.loss_fun = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.01, weight_decay=1e-4, momentum=0.9, nesterov=True)

    def train(self):
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
            print("\t- Batch: {}/{}, Loss: {}".format(int(batch_idx), len(self.train_dataloader), loss.data.item()))
        print("\t> Average Loss: {}".format(total_loss / len(self.train_dataloader)))

    def test(self):
        incorrect = 0
        for test_data, test_target in self.test_dataloader:
            # test_data, test_target = Variable(test_data), Variable(test_target)
            if torch.cuda.is_available():
                test_data, test_target = test_data.cuda(), test_target.cuda()
            test_output = self.model.forward(test_data)
            pred = test_output.data.max(1)[1]
            incorrect += pred.ne(test_target.data).cpu().sum()
        print("\t> Error: {0}".format(100.0 * incorrect / len(self.test_dataloader.dataset)))
