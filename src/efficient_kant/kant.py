import torch
import torch.nn.functional as F
import math


class KANTLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        cheby_order=10,
        scale_noise=0.1,
        scale_base=1.0,
        base_activation=torch.nn.SiLU,
        input_range=[-1, 1],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cheby_order = cheby_order
        self.input_range = input_range
        # TODO make this do something

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.cheby_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, cheby_order)
        )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.base_activation = base_activation()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base
        )
        torch.nn.init.kaiming_uniform_(
            self.cheby_weight, a=math.sqrt(5) * self.scale_base
        )

    def chebyshev_bases(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Chebyshev bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Chebyshev bases tensor of shape (batch_size, in_features, cheby_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        bases = torch.stack(
            [
                torch.special.chebyshev_polynomial_t(x, i)
                for i in range(self.cheby_order)
            ]
        )

        return bases.contiguous()

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = (
            F.linear(
                self.chebyshev_bases(x).view(x.size(0), -1),
                self.cheby_weight.view(self.out_features, -1),
            )
            / self.cheby_order
        )
        return base_output + spline_output

    def regularization_loss(self, regularize_activation=1.0):
        # """
        # Compute the regularization loss.

        # This is a dumb simulation of the original L1 regularization as stated in the
        # paper, since the original one requires computing absolutes and entropy from the
        # expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        # behind the F.linear function if we want an memory efficient implementation.

        # The L1 regularization is now computed as mean absolute value of the spline
        # weights. The authors implementation also includes this term in addition to the
        # sample-based regularization.
        # """
        # l1_fake = self.spline_weight.abs().mean(-1)
        # regularization_loss_activation = l1_fake.sum()
        # p = l1_fake / regularization_loss_activation
        # regularization_loss_entropy = -torch.sum(p * p.log())
        # return (
        #     regularize_activation * regularization_loss_activation
        #     + regularize_entropy * regularization_loss_entropy
        # )
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        I'm just using standard L1 on the weights for simplicity.

        It might also be worth including a term that increases the penalty on higher-order
        Chebyshev terms
        """
        l1_cheby = self.cheby_weight.abs().mean(-1)
        l1_base = self.base_weight.abs().mean(-1)
        return regularize_activation * (l1_cheby + l1_base)


class KANT(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        cheby_order=5,
        scale_noise=0.1,
        scale_base=1.0,
        base_activation=torch.nn.SiLU,
        input_range=[-1, 1],
    ):
        super().__init__()
        self.cheby_order = cheby_order
        self.input_range = input_range

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANTLinear(
                    in_features,
                    out_features,
                    cheby_order=cheby_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    base_activation=base_activation,
                    input_range=input_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0):
        return sum(
            layer.regularization_loss(regularize_activation) for layer in self.layers
        )
