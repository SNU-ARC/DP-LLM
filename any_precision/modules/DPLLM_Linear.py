import torch
import torch.nn as nn

try:
    from any_precision_ext import matmul_kbit, dequant_kbit
except:
    matmul_kbit, dequant_kbit = None, None


class DPLLM_Linear(nn.Module):
    def __init__(self, in_features, out_features, supported_bits, bias=True, precisions=None, device=None,
                 dtype=None, prefill_by_decode=False, maxmem=6, 
                 est_linear=None, est_async=None, est_params=None, est_T=None, b_l=None, b_h=None,
                 residual_layernorm=None, my_layernorm=None):
        super().__init__()
        if dequant_kbit is None or matmul_kbit is None:
            raise ModuleNotFoundError('Please install any precision CUDA kernel extension from modules/kernels.')
        if precisions is None:
            precisions = supported_bits
        if not isinstance(precisions, list):
            raise RuntimeError('supported_bits must be a list of integers.')

        self.in_features = in_features
        self.out_features = out_features
        self.precisions = precisions
        self.precision = max(self.precisions)
        self.supported_bits = supported_bits

        self.register_buffer(
            'qweight',
            torch.empty((max(supported_bits), out_features, in_features // 32), dtype=torch.int32, device=device)
        )

        for bit in supported_bits:
            self.register_buffer(
                f'lut{bit}',
                torch.empty((out_features, 2 ** bit), dtype=dtype, device=device)
            )

        if bias:
            self.register_buffer(
                "bias",
                torch.empty((out_features,), dtype=dtype, device=device)
            )
        else:
            self.bias = None

        self.est_linear = est_linear
        self.est_async = est_async
        if self.est_linear:
            self.lin_slope, self.lin_inter = est_params
        else:
            self.register_buffer(
                "jl",
                est_params
            )
        self.est_T = est_T
        self.b_l = b_l
        self.b_h = b_h

        self.comp_count = {bit:0 for bit in self.precisions}

        self.prefill_by_decode = prefill_by_decode
        self.maxmem = maxmem

        self.residual_layernorm = residual_layernorm
        self.my_layernorm = my_layernorm

    def prune_precisions(self):
        self.qweight = self.qweight[:max(self.precisions)]
        for bit in self.supported_bits:
            if bit not in self.precisions:
                delattr(self, f'lut{bit}')

    def forward(self, x, **kwargs):

        # Use residual or real input
        if self.est_async:
            x_for_err = self.my_layernorm.orig_fn(self.residual_layernorm.recorded_hidden_states)
        else:
            x_for_err = x
        
        # Use linear regression or JL
        if self.est_linear:
            e = x_for_err.norm(dim=-1) * self.lin_slope + self.lin_inter
        else:
            e = (x_for_err @ self.jl.T).norm(dim=-1)
        mask = e > self.est_T

        if x.numel() // x.shape[-1] > 1: # Prefill phase
            if self.prefill_by_decode: # For perplexity measuring
                # Make output tensor filled with NaN
                y = torch.empty(tuple([*x.shape[:-1], self.out_features]), dtype=x.dtype, device=x.device).fill_(torch.nan)

                # Prefill low precision tokens
                w_l = dequant_kbit(self.qweight, self._buffers[f'lut{self.b_l}'], self.b_l)
                y_l = torch.matmul(x, w_l.T)
                y[~mask] = y_l[~mask]
                b_l_count = (~mask).count_nonzero().item()

                # Prefill high precision tokens
                w_h = dequant_kbit(self.qweight, self._buffers[f'lut{self.b_h}'], self.b_h)
                y_h = torch.matmul(x, w_h.T)
                y[mask] = y_h[mask]
                b_h_count = mask.count_nonzero().item()

                assert not y.isnan().any(), "output contains NaN, possibly some tokens were not prefilled"

                # Update tokens count for each precision
                self.comp_count[self.b_l] += b_l_count
                self.comp_count[self.b_h] += b_h_count
            else: # For downstream tasks
                # Use maxmem for prefill
                w = dequant_kbit(self.qweight, self._buffers[f'lut{self.maxmem}'], self.maxmem)
                y = torch.matmul(x, w.T)

        else: # Decode phase
            b_now = self.b_h if mask else self.b_l
            y = matmul_kbit(x, self.qweight, self._buffers[f'lut{b_now}'], b_now)
            self.comp_count[b_now] += 1

        if self.bias is not None:
            y += self.bias

        return y

    def set_precision(self, precision):
        self.precision = precision

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
