import torch
import torch.nn as nn
import math

try:
    from any_precision_ext import matmul_kbit, dequant_kbit
except:
    matmul_kbit, dequant_kbit = None, None


class deq_gemm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qweight: torch.Tensor, bl: torch.Tensor, bh: torch.Tensor, x: torch.Tensor, b_l: int, b_h: int):
        ctx.qweight = qweight
        ctx.bl = bl
        ctx.bh = bh
        ctx.b_l = b_l
        ctx.b_h = b_h
        with torch.no_grad():
            wl = dequant_kbit(qweight, bl, b_l)
            yl = torch.matmul(x, wl.T)
            wh = dequant_kbit(qweight, bh, b_h)
            yh = torch.matmul(x, wh.T)
            del wl, wh

        return yl, yh

    @staticmethod
    def backward(ctx, dyl, dyh):
        qweight = ctx.qweight
        bl = ctx.bl
        bh = ctx.bh
        b_l = ctx.b_l
        b_h = ctx.b_h
        with torch.no_grad():
            wl = dequant_kbit(qweight, bl, b_l)
            dxl = dyl @ wl
            wh = dequant_kbit(qweight, bh, b_h)
            dxh = dyh @ wh
            dx = dxl+dxh
            del wl, wh
        
        return None, None, None, dx, None, None

class DPLLM_Linear_Finetune(nn.Module):
    def __init__(self, in_features, out_features, supported_bits, bias=True, precisions=None, device=None,
                 dtype=None, z_init=0.5, maxmem=6):
        super().__init__()
        if dequant_kbit is None or matmul_kbit is None:
            raise ModuleNotFoundError('Please install any precision CUDA kernel extension from modules/kernels.')
        if precisions is None:
            precisions = supported_bits
        if not isinstance(precisions, list):
            raise RuntimeError('supported_bits must be a list of integers.')

        self.dtype = dtype

        self.in_features = in_features
        self.out_features = out_features
        self.precisions = precisions
        self.precision = min(self.precisions)
        self.min_prec = min(self.precisions)
        self.supported_bits = supported_bits

        self.maxmem = maxmem

        self.register_buffer(
            'qweight',
            torch.empty((max(supported_bits), out_features, in_features // 32), dtype=torch.int32, device=device)
        )

        for bit in supported_bits:
            self.register_buffer(
                f'lut{bit}',
                torch.empty((out_features, 2 ** bit), dtype=self.dtype, device=device)
            )

        if bias:
            self.register_buffer(
                "bias",
                torch.empty((out_features,), dtype=self.dtype, device=device)
            )
        else:
            self.bias = None

        self.z_inv = math.log(z_init / (1 - z_init))
        self.device = device
        self.sigmoid = torch.nn.Sigmoid()

    def create_z(self):
        self.z = torch.nn.Parameter(torch.tensor(self.z_inv, device=self.qweight.device, requires_grad=True))

    def prune_precisions(self):
        self.qweight = self.qweight[:max(self.precisions)]
        for bit in self.supported_bits:
            if bit not in self.precisions:
                delattr(self, f'lut{bit}')

    def forward(self, x, **kwargs):
        func = deq_gemm.apply

        prange_len = self.maxmem - self.min_prec
        if self.maxmem > self.min_prec:
            p = self.sigmoid(self.z)*prange_len + self.min_prec
            bl = math.floor(p.item())
            bh = math.ceil(p.item())
            r = 1 - (p - bl)
            yl, yh = func(self.qweight, self._buffers[f'lut{bl}'], self._buffers[f'lut{bh}'], x, bl, bh)
            y = yl * r + yh * (1-r)
        else:
            bl = self.min_prec
            y, _ = func(self.qweight, self._buffers[f'lut{bl}'], self._buffers[f'lut{bl}'], x, bl, bl)
        
        if self.bias is not None:
            y += self.bias

        return y

    def set_precision(self, precision):
        self.precision = precision

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'