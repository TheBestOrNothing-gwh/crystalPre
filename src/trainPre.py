import argparse
import os
import pytz
import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from data_prop import *
from model import CrystalGraphConvNet


class Normalizer(object):
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor, device):
        """tensor is taken as a sample to calculate the mean and std"""
        self.device = device
        self.mean = torch.mean(tensor).to(self.device)
        self.std = torch.std(tensor).to(self.device)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"].to(self.device)
        self.std = state_dict["std"].to(self.device)


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target
    计算平均绝对误差

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


# 累计求平均值的方法，用于计算每个epoch的平均误差，因为每个batch进行了一次计算，所以需要统计一下
class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 用来保存模型的函数
def save_models(epoch, model, is_best, path):
    torch.save(model, os.path.join(path, f"model{epoch}.pth"))
    if is_best:
        torch.save(model, os.path.join(path, "best_model.pth"))


def args_parse():
    parser = argparse.ArgumentParser(
        description="Crystal Explainable Property Predictor"
    )
    # region 机器配置
    parser.add_argument("--device", type=str, default="cpu", help="Set the Device")
    # endregion
    # region 数据集配置
    parser.add_argument("--data_path", type=str, default="../data/", help="Data Path")
    parser.add_argument(
        "--radius", type=int, default=8, help="Radius of the sphere along an atom"
    )
    parser.add_argument(
        "--max_nbr",
        type=int,
        default=12,
        help="Maximum Number of neighbours to consider",
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--workers", type=int, default=0, help="workers")
    parser.add_argument(
        "--pin_memory", type=bool, default=True, help="Set the pin_memory"
    )
    parser.add_argument(
        "--split", type=list, default=[0.4, 0.1, 0.5], help="Split the dataset"
    )
    # endregion
    # region 模型配置
    parser.add_argument(
        "--atom_fea_len",
        default=64,
        type=int,
        metavar="N",
        help="number of hidden atom features in conv layers",
    )
    parser.add_argument(
        "--h_fea_len",
        default=128,
        type=int,
        metavar="N",
        help="number of hidden features after pooling",
    )
    parser.add_argument(
        "--n_conv", default=3, type=int, metavar="N", help="number of conv layers"
    )
    parser.add_argument(
        "--n_h",
        default=1,
        type=int,
        metavar="N",
        help="number of hidden layers after pooling",
    )
    # endregion
    # region 训练配置
    parser.add_argument(
        "--optim",
        default="Adam",
        type=str,
        metavar="SGD",
        help="choose an optimizer, SGD or Adam, (default: SGD)",
    )
    parser.add_argument(
        "--epochs",
        default=300,
        type=int,
        metavar="N",
        help="number of total epochs to run (default: 30)",
    )
    parser.add_argument(
        "--lr",
        "--learning_rate",
        default=0.01,
        type=float,
        metavar="LR",
        help="initial learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--lr_milestones",
        default=[100],
        nargs="+",
        type=int,
        metavar="N",
        help="milestones for scheduler (default: [100])",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--weight_decay",
        "--wd",
        default=0,
        type=float,
        metavar="W",
        help="weight decay (default: 0)",
    )
    # endregion
    # region 其他参数
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="Path of the Pretrained CrysAE Model Path",
    )
    parser.add_argument(
        "--feature_selector",
        type=bool,
        default=True,
        help="Option for feature selector",
    )
    parser.add_argument(
        "--timezone", type=str, default="Asia/Shanghai", help="Set the timezone"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1000,
        help="sample some batch to compute the normalize",
    )
    # endregion
    args = parser.parse_args()
    return args


def main():
    args = args_parse()
    # region 设置时间
    eastern = pytz.timezone(args.timezone)
    current_time = datetime.datetime.now().astimezone(eastern).time()
    current_date = datetime.datetime.now().astimezone(eastern).date()
    # endregion
    # region 设置机器
    device = torch.device(args.device)
    # endregion
    # region 设置结果存放位置
    path = "../results/Prediction/" + str(current_date) + "/" + str(current_time)
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(os.path.join(path, "models")):
        os.makedirs(os.path.join(path, "models"))
    # endregion
    # region 初始的日志信息
    out = open(os.path.join(path, "out.txt"), "w")
    out.writelines("Pretrained Model Path " + str(args.pretrained_model) + "\n")
    out.writelines("Data path : " + str(args.data_path) + "\n")
    out.writelines("Learning Rate : " + str(args.lr) + "\n")
    out.writelines("Epochs " + str(args.epochs) + "\n")
    out.writelines("Batch size " + str(args.batch_size) + "\n")
    # endregion
    # region 数据集
    dataset = CIFData(args.data_path, args.max_nbr, args.radius)
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        split=args.split,
        collate_fn=collate_pool,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
    )
    # 选择2000个样本用来估计均值、方差
    target_list = []
    count = 0
    for _, (_, target, _) in enumerate(train_loader):
        count += 1
        if count > args.sample:
            break
        target_list.append(target)
    sample_target = torch.cat(target_list, 0)
    normalizer = Normalizer(sample_target, device)
    # endregion
    # region 构造模型
    structure, _, _ = dataset[0]
    orig_atom_fea_len = structure[0].shape[-1]
    nbr_fea_len = structure[1].shape[-1]
    model = CrystalGraphConvNet(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=args.atom_fea_len,
        n_conv=args.n_conv,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h,
        classification=False,
    )
    model.to(device)
    # endregion
    # region 导入预训练AE来初始化模型
    pretrained_path = args.pretrained_model
    if pretrained_path != None:
        model_name = pretrained_path
        # if args.cuda:
        #     ae_model = torch.load(model_name)
        # else:
        #     ae_model = torch.load(model_name, map_location=torch.device('cpu'))
        ae_model = torch.load(model_name).to(device)
        # Transfer Weights from Encoder of AutoEnc
        model_dict = ae_model.state_dict()
        model_dict.pop("embedding.weight")
        model_dict.pop("fc_adj.weight")
        model_dict.pop("fc_adj.bias")
        # model_dict.pop('fc_edge.weight')
        # model_dict.pop('fc_edge.bias')
        model_dict.pop("fc_atom_feature.weight")
        model_dict.pop("fc_atom_feature.bias")
        model_dict.pop("fc1.weight")
        model_dict.pop("fc1.bias")
        # model_dict.pop('fc2.weight')
        # model_dict.pop('fc2.bias')
        pmodel_dict = model.state_dict()
        pmodel_dict.update(model_dict)
        model.load_state_dict(pmodel_dict)
    else:
        print("No Pretrained Model.Property Predictor will be trained from Scratch!!")
    # endregion
    # region 定义损失函数和优化器
    criterion = nn.MSELoss()
    assert (
        args.optim == "SGD" or args.optim == "Adam"
    ), "Only SGD or Adam is allowed as --optim"
    if args.optim == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(), args.lr, weight_decay=args.weight_decay
        )
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
    # endregion
    # region 训练过程
    best_mae_error = 1e10
    train_loss_list, val_loss_list, train_mae_list, val_mae_list = [], [], [], []
    for epoch in tqdm(range(args.epochs)):
        # train for one epoch
        train_loss, train_mae = train(
            train_loader, model, criterion, optimizer, normalizer, device
        )
        # evaluate on test set
        val_loss, val_mae = validate(val_loader, model, criterion, normalizer, device)
        scheduler.step()
        # remember the best mae_eror and save checkpoint
        is_best = val_mae < best_mae_error
        if is_best:
            best_mae_error = val_mae
        save_models(epoch, model, is_best, os.path.join(path, "models"))
        train_loss_list.append(train_loss.cpu().detach().numpy())
        train_mae_list.append(train_mae.cpu().detach().numpy())
        val_loss_list.append(val_loss.cpu().detach().numpy())
        val_mae_list.append(val_mae.cpu().detach().numpy())
        out.writelines(
            f"Epoch Summary : Epoch : {epoch} Train Mean Loss : {train_loss} Train MAE : {train_mae} Val Mean Loss : {val_loss} Val MAE : {val_mae} Best Val MAE : {best_mae_error}\n"
        )
    # endregion
    test_mae = test(test_loader, model, criterion, normalizer, device)
    out.writelines(f"Test MAE : {test_mae}\n")


def train(train_loader, model, criterion, optimizer, normalizer, device):
    # 均方误差，用作损失函数
    losses = AverageMeter()
    # 平均绝对误差，用作性能参考
    mae_errors = AverageMeter()

    # switch to train mode
    model.train()
    for _, (input, target, _) in enumerate(train_loader):
        # 加载数据到指定空间上
        atom_fea, nbr_fea, nbr_fea_idx, crys_atom_idx = (
            input[0],
            input[1],
            input[2],
            input[3],
        )
        input_var = (
            atom_fea.to(device),
            nbr_fea.to(device),
            nbr_fea_idx.to(device),
            [crys_idx.to(device) for crys_idx in crys_atom_idx],
        )
        target = target.to(device)
        target_var = normalizer.norm(target)
        # compute output
        output = model(*input_var)
        # measure accuracy and record loss
        loss = criterion(output, target_var)
        mae_error = mae(normalizer.denorm(output), target)
        losses.update(loss, target.size(0))
        mae_errors.update(mae_error, target.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses.avg, mae_errors.avg


def validate(val_loader, model, criterion, normalizer, device):
    # 均方误差，用作损失函数
    losses = AverageMeter()
    # 平均绝对误差，用作性能参考
    mae_errors = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, target, _) in enumerate(val_loader):
            # 加载数据到指定空间上
            atom_fea, nbr_fea, nbr_fea_idx, crys_atom_idx = (
                input[0],
                input[1],
                input[2],
                input[3],
            )
            input_var = (
                atom_fea.to(device),
                nbr_fea.to(device),
                nbr_fea_idx.to(device),
                [crys_idx.to(device) for crys_idx in crys_atom_idx],
            )
            target = target.to(device)
            target_var = normalizer.norm(target)
            # compute output
            output = model(*input_var)
            # measure accuracy and record loss
            loss = criterion(output, target_var)
            mae_error = mae(normalizer.denorm(output), target)
            losses.update(loss, target.size(0))
            mae_errors.update(mae_error, target.size(0))
    return losses.avg, mae_errors.avg


def test(test_loader, model, criterion, normalizer, device):
    mae_errors = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, target, _) in enumerate(test_loader):
            # 加载数据到指定空间上
            atom_fea, nbr_fea, nbr_fea_idx, crys_atom_idx = (
                input[0],
                input[1],
                input[2],
                input[3],
            )
            input_var = (
                atom_fea.to(device),
                nbr_fea.to(device),
                nbr_fea_idx.to(device),
                [crys_idx.to(device) for crys_idx in crys_atom_idx],
            )
            target = target.to(device)
            # compute output
            output = model(*input_var)
            # measure accuracy and record loss
            mae_error = mae(normalizer.denorm(output), target)
            mae_errors.update(mae_error, target.size(0))
    return mae_errors.avg


if __name__ == "__main__":
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    main()
