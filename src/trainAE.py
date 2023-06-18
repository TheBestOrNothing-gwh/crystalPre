import pytz
import os
import datetime
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from data import *
from model import *
from tools import *
       

def train(loader, model, optimizer, device, split):
    total_losses = AverageMeter()
    adj_reconst_losses = AverageMeter()
    feature_reconst_losses = AverageMeter()

    # 批训练
    pos_weight = torch.Tensor([0.1, 1, 1, 1, 1, 1]).to(device)
    model.train()
    for _, (input, adj, _) in enumerate(loader):
        atom_fea, nbr_fea, nbr_fea_idx, crys_atom_idx = (
            input[0],
            input[1],
            input[2],
            input[3],
        )
        # 加载数据到指定空间
        input_var = (
            atom_fea.to(device),
            nbr_fea.to(device),
            nbr_fea_idx.to(device),
            [crys_idx.to(device) for crys_idx in crys_atom_idx],
        )
        output_var = adj.to(device)
        # compute output
        adj, atom_fea = model(*input_var)
        # 计算loss
        loss_adj = F.nll_loss(adj, output_var, weight=pos_weight)
        loss_atom_fea = F.binary_cross_entropy_with_logits(
                atom_fea, input_var[0]
            )
        loss = loss_adj * split[0] + loss_atom_fea * split[1]
        adj_reconst_losses.update(loss_adj, len(input_var[3]))
        feature_reconst_losses.update(
            loss_atom_fea, len(input_var[3])
        )
        total_losses.update(loss, len(input_var[3]))
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_losses.avg, adj_reconst_losses.avg, feature_reconst_losses.avg


def args_parse():
    parser = argparse.ArgumentParser()
    # region 机器配置
    parser.add_argument("--device", type=str, default="cpu", help="Set the Device")
    # endregion
    # region 数据集配置
    parser.add_argument(
        "--data_path", type=str, default="../data_1000/", help="Root Data Path"
    )
    parser.add_argument(
        "--radius", type=int, default=8, help="Radius of the sphere along an atom"
    )
    parser.add_argument(
        "--max_num_nbr",
        type=int,
        default=12,
        help="Maximum Number of neighbours to consider",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--workers", type=int, default=0, help="workers")
    parser.add_argument(
        "--pin_memory", type=bool, default=True, help="Set the pin_memory"
    )
    # endregion
    # region 模型配置
    parser.add_argument(
        "--atom_fea_len", type=int, default=64, help="Atom Feature Dimension"
    )
    parser.add_argument(
        "--n_conv", type=int, default=3, help="Number of Convolution Layers"
    )
    # endregion
    # region 训练配置
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of Training Epoch"
    )
    parser.add_argument(
        "--lr_milestones",
        type=int,
        default=[100],
        nargs="+",
        metavar="N",
        help="milestones for scheduler (default: [100])",
    )
    parser.add_argument("--gamma", type=float, default=0.9, help="Set gamma")
    parser.add_argument(
        "--optim",
        type=str,
        default="Adam",
        metavar="SGD",
        help="choose an optimizer, SGD or Adam, (default: SGD)",
    )
    parser.add_argument("--lr", type=float, default=0.003, help="Learning Rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Set the Weight_Decay, L2regular",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--global_local_split", type=float, nargs='+', default=[0.9, 0.1], help="global:local"
    )
    # endregion
    # region 其他参数
    parser.add_argument(
        "--timezone", type=str, default="Asia/Shanghai", help="Set the Timezone"
    )
    # endregion
    args = parser.parse_args()
    return args


def main():
    args = args_parse()
    # region 检查参数是否合法
    assert (
        abs(args.global_local_split[0] + args.global_local_split[1] - 1) <= 1e-5
    ), "global + local == 1"
    # endregion
    # region 设置时间
    eastern = pytz.timezone(args.timezone)
    current_date = datetime.datetime.now().astimezone(eastern).date()
    current_time = datetime.datetime.now().astimezone(eastern).time()
    # endregion
    # region 设置机器
    device = torch.device(args.device)
    # endregion
    # region 结果的存放位置
    path = os.path.join("../results/CrystalAE", str(current_date), str(current_time))
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(os.path.join(path, "models")):
        os.makedirs(os.path.join(path, "models"))
    # endregion
    # region 初始的日志信息
    out = open(os.path.join(path, "out.txt"), "w")
    out.writelines(f'data_path :{args.data_path}\n')
    out.writelines(f'radius :{args.radius}\n')
    out.writelines(f'max_num_nbr :{args.max_num_nbr}\n')
    out.writelines("***** Hyper-Parameters Details ********\n")
    out.writelines(f'atom_fea_len :{args.atom_fea_len}\n')
    out.writelines(f'n_conv :{args.n_conv}\n')
    out.writelines(f'epochs :{args.epochs}\n')
    out.writelines(f'batch_size :{args.batch_size}\n')
    # endregion
    # region 数据集
    full_dataset = CIFData(args.data_path, args.max_num_nbr, args.radius)
    loader = get_data_loader(
        dataset=full_dataset,
        collate_fn=collate_pool,
        data_size=len(full_dataset),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
    )
    # endregion
    # region 构造模型
    structure, _, _ = full_dataset[0]
    orig_atom_fea_len = structure[0].shape[-1]
    nbr_fea_len = structure[1].shape[-1]
    model = CrystalAE(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=args.atom_fea_len,
        n_conv=args.n_conv,
    )
    model.to(device)
    # endregion
    # region Optimizer
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
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.gamma)
    # endregion
    # region Training Process
    # 训练相关变量
    total_loss_list = []
    adj_reconst_loss_list = []
    feature_reconst_loss_list = []
    min_loss = 1000
    # 训练流程
    for epoch in tqdm(range(args.epochs)):
        total_loss, adj_reconst_loss, feature_reconst_loss = train(
            loader, model, optimizer, device, args.global_local_split
        )
        scheduler.step()
        # remember the best loss and save checkpoint
        is_best = total_loss < min_loss
        if is_best:
            min_loss = total_loss
        save_models(epoch, model, is_best, os.path.join(path, "models"))
        total_loss_list.append(total_loss.cpu().detach().numpy())
        adj_reconst_loss_list.append(adj_reconst_loss.cpu().detach().numpy())
        feature_reconst_loss_list.append(feature_reconst_loss.cpu().detach().numpy())
        out.writelines(
            f"Epoch Summary : Epoch : {epoch} Total Loss : {total_loss} Adj Reconst Loss : {adj_reconst_loss} Feature Reconst Loss : {feature_reconst_loss}\n"
        )
    show(os.path.join(path, "total_loss.png"), args.epochs, {"total_loss":np.array(total_loss_list)})
    show(os.path.join(path, "adj_reconst_loss.png"), args.epochs, {"adj_reconst_loss":np.array(adj_reconst_loss_list)})
    show(os.path.join(path, "feature_reconst_loss.png"), args.epochs, {"feature_reconst_loss":np.array(feature_reconst_loss_list)})
    # endregion
    

if __name__ == "__main__":
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    main()
