
import torch


def pack_inputs(*tensors, item_num_dims=3):
    shapes = []
    tensor_list = []
    for t in tensors:
        s = t.shape[:-item_num_dims]
        if len(s) > 1:  # batch is over two dims
            t = t.view(-1, *t.shape[len(s):])
        tensor_list.append(t)
        shapes.append(s)
    return torch.cat(tensor_list), shapes


def unpack_outputs(tensor, shapes):
    idx = 0
    out = []
    for shape in shapes:
        if len(shape) == 1:
            B = shape[0]
        else:
            B = torch.tensor(shape).prod()
        out_shape = shape + tensor.shape[1:]
        out.append(tensor[idx:idx + B].view(*out_shape))
        idx += B
    return tuple(out)
