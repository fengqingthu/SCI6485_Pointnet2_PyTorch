import fire
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.pointnet2_cls import pointnet2_cls_ssg, pointnet2_cls_msg
from models.pointnet2_seg import pointnet2_seg_ssg
from data.ModelNet40 import ModelNet40
from data.ShapeNet import ShapeNet
from utils.IoU import cal_accuracy_iou
from data.provider import pc_normalize


def classify(model_id, data_path, checkpoint, npoints, dims=6, nclasses=40):
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
    
    xyz_points = np.loadtxt(data_path, delimiter=',')
    if npoints > 0:
        inds = np.random.randint(0, len(xyz_points), size=(npoints, ))
    else:
        inds = np.arange(len(xyz_points))
        np.random.shuffle(inds)
    xyz_points = xyz_points[inds, :]
    # normalize
    xyz_points[:, :3] = pc_normalize(xyz_points[:, :3])
    xyz, points = xyz_points[:, :, :3], xyz_points[:, :, :3]
    with torch.no_grad():
        pred = model(xyz.to(device), points.to(device))
        pred = torch.max(pred, dim=-1)[1]
    print('Pred: {}'.format(pred))

    # total_correct, total_seen = 0, 0
    # for data, labels in tqdm(test_loader):
    #     labels = labels.to(device)
    #     xyz, points = data[:, :, :3], data[:, :, 3:]
    #     with torch.no_grad():
    #         pred = model(xyz.to(device), points.to(device))
    #         pred = torch.max(pred, dim=-1)[1]
    #         total_correct += torch.sum(pred == labels)
    #         total_seen += xyz.shape[0]
    # print("Evaluating completed!")
    # print('Corr: {}, Seen: {}, Acc: {:.4f}'.format(total_correct, total_seen, total_correct / float(total_seen)))

if __name__ == '__main__':
    fire.Fire()