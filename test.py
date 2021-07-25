import torch
from torch.utils.data import DataLoader

from dataset import Action
from model import ResNet18, ResNet3D,generate_ResNet3D, RNN

device = torch.device('cuda')
batchsz = 32

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

    test_db = Action('data\\test', mode='test',model='LSTM')
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=2)
    model=RNN('LSTM',68,20,3).to(device)
    model.load_state_dict(torch.load('LSTM_best.mdl'))
    LSTM_test_acc = evaluate(model, test_loader)
    print('LSTM_test acc:', LSTM_test_acc)

    test_db = Action('data\\test', mode='test', model='GRU')
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=2)
    model = RNN('GRU', 68, 20, 3).to(device)
    model.load_state_dict(torch.load('GRU_best.mdl'))
    GRU_test_acc = evaluate(model, test_loader)
    print('GRU_test acc:', GRU_test_acc)

    # test_db = Action('data\\test', mode='test', model='ResNet3D')
    # test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=2)
    # model = generate_ResNet3D(18).to(device)
    # model.load_state_dict(torch.load('ResNet3D_best.mdl'))
    # ResNet3D_test_acc = evaluate(model, test_loader)
    # print('ResNet3D_test acc:', ResNet3D_test_acc)

if __name__ == '__main__':
    main()