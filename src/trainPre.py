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
from model import *
from tools import *


def train(train_loader, model, criterion, optimizer, normalizer, device):
    losses = AverageMeter()
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
    losses = AverageMeter()
    mae_errors = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for _, (input, target, _) in enumerate(val_loader):
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


def test(test_loader, model, normalizer, device):
    mae_errors = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for _, (input, target, _) in enumerate(test_loader):
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
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--workers", type=int, default=0, help="workers")
    parser.add_argument(
        "--pin_memory", type=bool, default=True, help="Set the pin_memory"
    )
    parser.add_argument(
        "--split", type=float, nargs='+', default=[0.6, 0.3, 0.1], help="Split the dataset"
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
        "--pretrained_path",
        type=str,
        default=None,
        help="Path of the Pretrained CrysAE Model Path",
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
    assert (
        abs(args.split[0] + args.split[1] + args.split[2] - 1) <= 1e-5
    ), "train + val + test == 1"
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
    out.writelines("Pretrained Model Path " + str(args.pretrained_path) + "\n")
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
        n_h=args.n_h
    )
    model.to(device)
    # endregion
    # region 导入预训练AE来初始化模型
    if args.pretrained_path != None:
        model.load_state_dict(torch.load(args.pretrained_path, map_location=device), strict=False)
        print("Loaded pretrained model!!!")
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
    best_model = model = CrystalGraphConvNet(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=args.atom_fea_len,
        n_conv=args.n_conv,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h
    )
    best_model.load_state_dict(torch.load(os.path.join(path, "models", "best_model.pth")))
    best_model.to(device)
    test_mae = test(test_loader,best_model, normalizer, device)
    out.writelines(f"Test MAE : {test_mae}\n")


if __name__ == "__main__":
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    main()
