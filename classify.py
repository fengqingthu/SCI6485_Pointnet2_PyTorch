import fire
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from models.pointnet2_cls import pointnet2_cls_ssg, pointnet2_cls_msg
from models.pointnet2_seg import pointnet2_seg_ssg
from data.ModelNet40 import ModelNet40
from data.ShapeNet import ShapeNet
from utils.IoU import cal_accuracy_iou
from data.provider import pc_normalize
import os


def classify(model_id, data_dir, checkpoint, npoints, dims=6, nclasses=40):
    print('Loading..')
    Models = {
        'pointnet2_cls_ssg': pointnet2_cls_ssg,
        'pointnet2_cls_msg': pointnet2_cls_msg
    }
    Model = Models[model_id]
    device = torch.device('cuda')
    model = Model(dims, nclasses)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    print('Loading {} completed'.format(checkpoint))
    
    data = []
    
    order = []

    for file in os.listdir(data_dir):
        # skip non-obj files
        if not file.endswith(".txt"):
            continue
        order.append(file.split('_')[1].split('.')[0])
        xyz_points = np.loadtxt(data_dir+'/'+file, delimiter=',')

        if npoints > 0:
            inds = np.random.randint(0, len(xyz_points), size=(npoints, ))
        else:
            inds = np.arange(len(xyz_points))
            np.random.shuffle(inds)
        xyz_points = xyz_points[inds, :]
        # normalize
        xyz_points[:, :3] = pc_normalize(xyz_points[:, :3])

        # print(xyz_points.shape)
        data.append(xyz_points)
    
    data = torch.Tensor(np.stack(data))
    print(data.shape)
    test_loader = DataLoader(dataset=TensorDataset(data),
                             batch_size=1, shuffle=False,
                             num_workers=1)
    
    print('Constructing dataloader completed')

    i = 0
    m = nn.softmax(dim=1)
    for batch in tqdm(test_loader):
        # print(data.shape)
        data = batch[0]
        xyz = data[:, :, :3]
        points = torch.zeros_like(xyz) # do not use normals
        with torch.no_grad():
            pred = model(xyz.to(device), points.to(device))
            pred = m(pred)
            # print(cls2name[torch.max(pred, dim=-1)[1]])
        print('Sample {} plantness: {}'.format(order[i], pred[0,26]))
        i += 1

if __name__ == '__main__':
    fire.Fire()