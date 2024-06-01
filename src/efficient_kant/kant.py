import torch
import math


class KANTLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        cheby_order=10,
        scale_base=1.0,
        input_range=[-1, 1],
        final_layer=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cheby_order = cheby_order
        self.scale_base = scale_base
        self.input_range = input_range
        # TODO make this do something: probably scale everything to lie
        # between -1 and 1 if it doesn't already

        self.cheby_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, cheby_order)
        )

        self.final_layer = final_layer

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(
            self.cheby_weight, a=math.sqrt(5) * self.scale_base, mode="fan_out"
        )

    def chebyshev_bases(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Chebyshev bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Chebyshev bases tensor of shape (cheby_order, batch_size, in_features).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        acos = torch.arccos(x)
        bases = torch.stack([torch.cos(i * acos) for i in range(self.cheby_order)])

        return bases.contiguous()

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        ###
        # This is not the most efficient way of doing this calculation:
        # more efficient would be to use Clenshaw's algorithm (https://en.wikipedia.org/wiki/Clenshaw_algorithm)
        # to avoid numerical instabilities. However that is less clear in the code
        ###

        if not self.final_layer:
            # Ensure that Chebyshev polynomial is between -1 and 1 and does not explode
            _output = (
                torch.einsum(
                    "cbi,oic->bo",
                    self.chebyshev_bases(x),
                    self.cheby_weight
                    / torch.sum(torch.abs(self.cheby_weight), axis=-1)[:, :, None],
                )
                / self.out_features
            )
        else:
            _output = torch.einsum(
                "cbi,oic->bo", self.chebyshev_bases(x), self.cheby_weight
            )
        return _output.contiguous()

    def regularization_loss(
        self,
        regularize_activation=1.0,
        regularization_scaling=0.0,
    ):
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
        # l1_cheby = self.cheby_weight.abs().mean()
        # return regularize_activation * l1_cheby
        regularization_lambda = regularize_activation * torch.tensor(
            [(1.0 + regularization_scaling) ** i for i in range(self.cheby_order)],
            device=self.cheby_weight.device,
        )
        cheby_weight_reg = torch.einsum(
            "c,oic->", regularization_lambda, self.cheby_weight.abs()
        ) / math.prod(self.cheby_weight.size())
        return cheby_weight_reg


class KANT(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        cheby_order=5,
        scale_base=1.0,
        input_range=[-1, 1],
    ):
        super().__init__()
        self.cheby_order = cheby_order
        self.input_range = input_range

        self.layers = torch.nn.ModuleList()
        num_layers = len(layers_hidden)
        for layer_ind, (in_features, out_features) in enumerate(
            zip(layers_hidden, layers_hidden[1:])
        ):
            self.layers.append(
                KANTLinear(
                    in_features,
                    out_features,
                    cheby_order=cheby_order,
                    scale_base=scale_base,
                    input_range=input_range,
                    final_layer=layer_ind == num_layers - 2,
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(
        self, regularize_activation=1.0, regularization_scaling=0.0
    ):
        return sum(
            layer.regularization_loss(regularize_activation, regularization_scaling)
            for layer in self.layers
        )
