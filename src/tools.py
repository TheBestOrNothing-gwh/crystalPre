import torch
import matplotlib.pyplot as plt
import pandas as pd
import os


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


# 用来保存模型的函数
def save_models(epoch, model, is_best, path):
    torch.save(model, os.path.join(path, f"model{epoch}.pth"))
    if is_best:
        torch.save(model, os.path.join(path, "best_model.pth"))


def show(epochs, path, name, y):
    x = list(range(1, epochs + 1))
    df = pd.DataFrame({'epochs': x, name: y})
    plt.figure(figsize=(16, 14))
    plt.tight_layout()
    plt.plot('epochs', name, data=df, marker='', color='blue', linewidth=10)
    plt.savefig(path)


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
