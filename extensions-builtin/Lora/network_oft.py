import torch
import network
from lyco_helpers import factorization
from einops import rearrange
from modules import devices


class ModuleTypeOFT(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        if all(x in weights.w for x in ["oft_blocks"]) or all(x in weights.w for x in ["oft_diag"]):
            return NetworkModuleOFT(net, weights)

        return None

# adapted from kohya-ss' implementation https://github.com/kohya-ss/sd-scripts/blob/main/networks/oft.py
# and KohakuBlueleaf's implementation https://github.com/KohakuBlueleaf/LyCORIS/blob/dev/lycoris/modules/diag_oft.py
class NetworkModuleOFT(network.NetworkModule):
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):

        super().__init__(net, weights)

        self.lin_module = None
        self.org_module: list[torch.Module] = [self.sd_module]

        # kohya-ss
        if "oft_blocks" in weights.w.keys():
            self.is_kohya = True
            self.oft_blocks = weights.w["oft_blocks"] # (num_blocks, block_size, block_size)
            self.alpha = weights.w["alpha"]
            self.dim = self.oft_blocks.shape[0] # lora dim
            #self.oft_blocks = rearrange(self.oft_blocks, 'k m ... -> (k m) ...')
        elif "oft_diag" in weights.w.keys():
            self.is_kohya = False
            self.oft_blocks = weights.w["oft_diag"] # (num_blocks, block_size, block_size)

            # alpha is rank if alpha is 0 or None
            if self.alpha is None:
                pass
            self.dim = self.oft_blocks.shape[1] # FIXME: almost certainly incorrect, assumes tensor is shape [*, m, n]
        else:
            raise ValueError("oft_blocks or oft_diag must be in weights dict")

        is_linear = type(self.sd_module) in [torch.nn.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear]
        is_conv = type(self.sd_module) in [torch.nn.Conv2d]
        is_other_linear = type(self.sd_module) in [torch.nn.MultiheadAttention]

        if is_linear:
            self.out_dim = self.sd_module.out_features
        elif is_other_linear:
            self.out_dim = self.sd_module.embed_dim
        elif is_conv:
            self.out_dim = self.sd_module.out_channels
        else:
            raise ValueError("sd_module must be Linear or Conv")

        if self.is_kohya:
            #self.num_blocks = self.dim
            #self.block_size = self.out_dim // self.num_blocks
            #self.block_size = self.dim
            #self.num_blocks = self.out_dim // self.block_size
            self.constraint = self.alpha * self.out_dim
            self.num_blocks, self.block_size = factorization(self.out_dim, self.dim)
        else:
            self.constraint = None
            self.block_size, self.num_blocks = factorization(self.out_dim, self.dim)

        if is_other_linear:
            self.lin_module = self.create_module(weights.w, "oft_diag", none_ok=True)

    
    def create_module(self, weights, key, none_ok=False):
        weight = weights.get(key)

        if weight is None and none_ok:
            return None

        is_linear = type(self.sd_module) in [torch.nn.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear, torch.nn.MultiheadAttention]
        is_conv = type(self.sd_module) in [torch.nn.Conv2d]

        if is_linear:
            weight = weight.reshape(weight.shape[0], -1)
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif is_conv and key == "lora_down.weight" or key == "dyn_up":
            if len(weight.shape) == 2:
                weight = weight.reshape(weight.shape[0], -1, 1, 1)

            if weight.shape[2] != 1 or weight.shape[3] != 1:
                module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], self.sd_module.kernel_size, self.sd_module.stride, self.sd_module.padding, bias=False)
            else:
                module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        elif is_conv and key == "lora_mid.weight":
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], self.sd_module.kernel_size, self.sd_module.stride, self.sd_module.padding, bias=False)
        elif is_conv and key == "lora_up.weight" or key == "dyn_down":
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        else:
            raise AssertionError(f'Lora layer {self.network_key} matched a layer with unsupported type: {type(self.sd_module).__name__}')

        with torch.no_grad():
            if weight.shape != module.weight.shape:
                weight = weight.reshape(module.weight.shape)
            module.weight.copy_(weight)

        module.to(device=devices.cpu, dtype=devices.dtype)
        module.weight.requires_grad_(False)

        return module


    def merge_weight(self, R_weight, org_weight):
        R_weight = R_weight.to(org_weight.device, dtype=org_weight.dtype)
        if org_weight.dim() == 4:
            weight = torch.einsum("oihw, op -> pihw", org_weight, R_weight)
        else:
            weight = torch.einsum("oi, op -> pi", org_weight, R_weight)
        return weight

    def get_weight(self, oft_blocks, multiplier=None):
        if self.constraint is not None:
            constraint = self.constraint.to(oft_blocks.device, dtype=oft_blocks.dtype)

        block_Q = oft_blocks - oft_blocks.transpose(1, 2)
        norm_Q = torch.norm(block_Q.flatten())
        if self.constraint is not None:
            new_norm_Q = torch.clamp(norm_Q, max=constraint)
        else:
            new_norm_Q = norm_Q
        block_Q = block_Q * ((new_norm_Q + 1e-8) / (norm_Q + 1e-8))
        m_I = torch.eye(self.num_blocks, device=oft_blocks.device).unsqueeze(0).repeat(self.block_size, 1, 1)
        #m_I = torch.eye(self.block_size, device=oft_blocks.device).unsqueeze(0).repeat(self.num_blocks, 1, 1)
        block_R = torch.matmul(m_I + block_Q, (m_I - block_Q).inverse())

        block_R_weighted = multiplier * block_R + (1 - multiplier) * m_I
        R = torch.block_diag(*block_R_weighted)
        return R

    def calc_updown_kohya(self, orig_weight, multiplier):
        R = self.get_weight(self.oft_blocks, multiplier)
        merged_weight = self.merge_weight(R, orig_weight)

        updown = merged_weight.to(orig_weight.device, dtype=orig_weight.dtype) - orig_weight
        output_shape = orig_weight.shape
        orig_weight = orig_weight
        return self.finalize_updown(updown, orig_weight, output_shape)

    def calc_updown_kb(self, orig_weight, multiplier):
        is_other_linear = type(self.sd_module) in [torch.nn.MultiheadAttention]

        if not is_other_linear:
            if is_other_linear and orig_weight.shape[0] != orig_weight.shape[1]:
                orig_weight=orig_weight.permute(1, 0)

            R = self.oft_blocks.to(orig_weight.device, dtype=orig_weight.dtype)
            # if self.is_kohya:
            #     #R = R.transpose(1, 0)
            #     R = self.get_weight(self.oft_blocks, multiplier)
            merged_weight = rearrange(orig_weight, '(k n) ... -> k n ...', k=self.num_blocks, n=self.block_size)
            #if self.is_kohya:
            #    R = R * multiplier + torch.eye(self.num_blocks, device=orig_weight.device)
            #else:
            R = R * multiplier + torch.eye(self.block_size, device=orig_weight.device)

            merged_weight = torch.einsum(
                'k n m, k n ... -> k m ...',
                R,
                merged_weight
            )
            merged_weight = rearrange(merged_weight, 'k m ... -> (k m) ...')

            if is_other_linear and orig_weight.shape[0] != orig_weight.shape[1]:
                orig_weight=orig_weight.permute(1, 0)

            updown = merged_weight.to(orig_weight.device, dtype=orig_weight.dtype) - orig_weight
            output_shape = orig_weight.shape
        else:
            # FIXME: skip MultiheadAttention for now
            #up = self.lin_module.weight.to(orig_weight.device, dtype=orig_weight.dtype)
            updown = torch.zeros([orig_weight.shape[1], orig_weight.shape[1]], device=orig_weight.device, dtype=orig_weight.dtype)
            output_shape = (orig_weight.shape[1], orig_weight.shape[1])

        return self.finalize_updown(updown, orig_weight, output_shape)

    def calc_updown(self, orig_weight):
        multiplier = self.multiplier() * self.calc_scale()
        #if self.is_kohya:
        #    return self.calc_updown_kohya(orig_weight, multiplier)
        #else:
        return self.calc_updown_kb(orig_weight, multiplier)

    # override to remove the multiplier/scale factor; it's already multiplied in get_weight
    def finalize_updown(self, updown, orig_weight, output_shape, ex_bias=None):
        #return super().finalize_updown(updown, orig_weight, output_shape, ex_bias)

        if self.bias is not None:
            updown = updown.reshape(self.bias.shape)
            updown += self.bias.to(orig_weight.device, dtype=orig_weight.dtype)
            updown = updown.reshape(output_shape)

        if len(output_shape) == 4:
            updown = updown.reshape(output_shape)

        if orig_weight.size().numel() == updown.size().numel():
            updown = updown.reshape(orig_weight.shape)

        if ex_bias is not None:
            ex_bias = ex_bias * self.multiplier()

        return updown, ex_bias
