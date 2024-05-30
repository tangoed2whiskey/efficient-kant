import torch
import torch.nn as nn
from tqdm import tqdm

from efficient_kant import KANT


def test_mul():
    kant = KANT([2, 1], cheby_order=10)
    optimizer = torch.optim.LBFGS(kant.parameters(), lr=1)
    with tqdm(range(1000)) as pbar:
        for i in pbar:
            loss, reg_loss = None, None

            def closure():
                optimizer.zero_grad()
                x = torch.rand(1024, 2) * 2 - 1
                y = kant(x)

                assert y.shape == (1024, 1)
                nonlocal loss, reg_loss
                u = x[:, 0]
                v = x[:, 1]
                loss = nn.functional.mse_loss(y.squeeze(-1), (u**2 + v).squeeze(-1))
                reg_loss = kant.regularization_loss(1e-5, regularization_scaling=0.1)
                (loss + reg_loss).backward()
                return loss + reg_loss

            optimizer.step(closure)
            pbar.set_postfix(mse_loss=loss.item(), reg_loss=reg_loss.item())
    for layer in kant.layers:
        print(layer.cheby_weight)
    print(kant(torch.tensor([[1.0, 0.5]])))


if __name__ == "__main__":
    test_mul()
