import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

import time
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm
import visdom

from dataload import testDataset
from train import evalIdentification

img_testpath = '/path/to/lfw'
testpairs_txt = '/path/to/pairs.txt'
img_transform = transforms.Compose([
    transforms.Scale((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])
batch_size = 32

def main():
    test_data = testDataset(dir=img_testpath, transform=img_transform)
    test_dataloader = DataLoader(dataset=test_data, shuffle=False, num_workers=4, batch_size=batch_size)

    model_name = 'model/model_A_rank_2018122516.pkl'
    print model_name
    model = torch.load(model_name)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()
    model.eval()

    accuracy = []
    std_err = []
    thrange = np.arange(0, 1.01, 0.01)
    feature_map = {}
    for data in tqdm(test_dataloader):
        img, label = data
        if use_gpu:
            img = Variable(img).cuda()
        else:
            img = Variable(img)
        feature = model(img)
        feature = F.normalize(feature, p=2, dim=1).cpu().data.numpy()
        person_names, idx = label
        for p in range(len(person_names)):
            if person_names[p] not in feature_map:
                feature_map[person_names[p]] = {}
            feature_map[person_names[p]][idx[p]] = feature[p,:]
    mx_acc = 0.0
    for threshold in thrange:
        t_accuracy, t_std = evalIdentification(feature_map, testpairs_txt, threshold)
        if t_accuracy > mx_acc:
            mx_th = threshold
            mx_acc = t_accuracy
        accuracy.append(t_accuracy)
        std_err.append(t_std)
    print "%.4f: %.4f"% (mx_th, mx_acc)

    # vis = visdom.Visdom(env='face')
    # vis.line(
    #         X = thrange,
    #         Y = np.array(accuracy),
    #         opts = dict(
    #             title=time.strftime('%Y-%m-%d %H:%M:%S'),
    #             xlabel='threshold',
    #             ylabel='accuracy'
    #         )
    # )

if __name__ == '__main__':
    main()
