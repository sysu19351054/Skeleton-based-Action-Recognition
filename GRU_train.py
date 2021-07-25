import torch
from torch import optim, nn
import visdom
import torchvision
from torch.utils.data import DataLoader
import numpy as np

from dataset import Action
from model import ResNet18, ResNet3D, RNN
from torchvision.models import resnet18

from utils import Flatten

batchsz = 32
lr = 1e-3
epochs = 200

device = torch.device('cuda')
torch.manual_seed(1234)

train_db = Action('data\\train', mode='train', model='GRU')
test_db = Action('data\\test',  mode='test', model='GRU')

train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=4)
test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=2 )

viz = visdom.Visdom()

def evaluate(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)

    for x,y in loader:
        x = x.type(torch.FloatTensor)
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()

    return correct / total


def main():

    model=RNN('GRU',68,20,3).to(device)
    # model = resnet3D.generate_model(18).to(device)
    # model = ResNet18(5).to(device)
    # model = ResNet18(5)
    # trained_model = resnet18(pretrained=True)
    # model = nn.Sequential(*list(trained_model.children())[:-1],
    #                       Flatten(),
    #                       nn.Linear(512, 5)
    #                       ).to(device)

    # x=torch.randn(2, 3, 16, 16,16).to(device)
    # print(model(x).shape)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='GRU_test_acc', opts=dict(title='GRU_test_acc'))
    for epoch in range(epochs):
        print(epoch)
        for step, (x,y) in enumerate(train_loader):

            x = x.type(torch.FloatTensor)
            x, y = x.to(device), y.to(device)


            model.train()
            logits = model(x)
            loss = criteon(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1

        if epoch % 1 == 0:

            test_acc = evaluate(model, test_loader)
            viz.line([test_acc], [global_step], win='GRU_test_acc', update='append')
            if test_acc>best_acc:
                best_epoch = epoch
                best_acc = test_acc

                torch.save(model.state_dict(), 'GRU.mdl')

    print('best acc:', best_acc, 'best epoch:', best_epoch)

    model.load_state_dict(torch.load('GRU.mdl'))
    print('loaded from ckpt!')

    test_acc = evaluate(model, test_loader)
    print('GRU_test acc:', test_acc)


if __name__ == '__main__':
    main()