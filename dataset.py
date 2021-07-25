import  torch
import  os, glob
import  random, csv

from    torch.utils.data import Dataset, DataLoader


import numpy as np




class Action(Dataset):

    def __init__(self, root,mode,model):
        super(Action, self).__init__()

        self.root = root
        self.mode = mode
        self.model=model

        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue

            self.name2label[name] = len(self.name2label.keys())

        self.images, self.labels = self.load_csv('images.csv')




    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.npy'))


            print(len(images), images)

            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img, label])
                print('writen into csv file:', filename)


        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)

        return images, labels



    def __len__(self):

        return len(self.images)


    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]

        img = np.load(img)

        img[:, 1, :, :, :] = 0

        if self.mode=='train':
            a=random.choice([0,1,2,3,4,5])
            if a == 1 or a==3:
                x = random.uniform(-10, 10)
                img[:, 0, :, :, :] = img[:, 0, :, :, :] + x
            if a == 2 or a==3:
                y = random.uniform(-10, 10)
                img[:, 2, :, :, :] = img[:, 2, :, :, :] + y

        img=img[0,:,:,:,:]


        if self.model=='ResNet18':
            img = np.resize(img, (3,224,224))

        if self.model=='ResNet3D' or self.model=='DenseNet':
            img = np.resize(img, (3,16,16,16))

        if self.model=='LSTM' or self.model=='GRU':
            img=img[0:3:2,:,:]
            img = img.transpose((1, 0, 2, 3))
            img = np.resize(img,(128,68))

        label = torch.tensor(label)


        return img, label





def main():



    db = Action('data//train', 64, mode='test')

    x,y = next(iter(db))
    print(x.shape)
    # print('sample:', x.shape, y.shape, y)

    # sample=db.denormalize(x)
    # sample=np.load(x)
    # sample[:, 1, :, :, :] = 0
    # visualise(x, graph=Graph(), is_3d=True)
    #
    # loader = DataLoader(db, batch_size=1, shuffle=True, num_workers=8)
    #
    # for x, y in loader:
    #     # sample = np.load(x)
    #     # sample[:, 1, :, :, :] = 0
    #     print(x.shape)
    #     visualise(x[0,:,:,:,:,:], graph=Graph(), is_3d=True)

if __name__ == '__main__':
    main()