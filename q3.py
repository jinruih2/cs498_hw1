###Q3: allreduce###
###please implement ring_allreduce method, using  pytorch's dist method is not allowed###
#test
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist

def reduce_scatter(chunks, tmp, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #                                                                   #
    for i in range(world-1):
        send_rank = (rank - i)%world
        recv_rank = (rank - i - 1)%world

        r = dist.irecv(tmp, src=left)
        s = dist.isend(chunks[send_rank],dst=right)
        r.wait(); s.wait()
        chunks[recv_rank] += tmp
    return
        
def all_gather(chunks, tmp, current, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #                                                                   #
    cur = (rank + 1) % world
    for _ in range(world - 1):
        # receive the next chunk from left (which should be (cur-1)%world),
        # while sending our current chunk to right
        rreq = dist.irecv(tmp, src=left)
        sreq = dist.isend(chunks[cur], dst=right)
        rreq.wait(); sreq.wait()

        next_idx = (cur - 1) % world   # the index of the chunk we just received
        chunks[next_idx].copy_(tmp)
        cur = next_idx
    return

def ring_allreduce_(tensor: torch.Tensor, world_size = None, rankid = None):
    """In-place ring all-reduce (SUM, optional average) using isend/irecv."""
    world = world_size
    if world == 1: return tensor
    rank = rankid
    left, right = (rank - 1) % world, (rank + 1) % world

    ##following steps try to fill blank to the tensor so that final tensor can be divided to 3 chunks evenly
    flat = tensor.contiguous().view(-1)
    n = flat.numel()
    chunk = (n + world - 1) // world
    padded_n = chunk * world

    padded_flat = torch.zeros(
        padded_n, dtype=flat.dtype, device=flat.device
    )
    padded_flat[:n] = flat
    #                                                                   #
    #                                                                   #
    # your code here: we cannot divide flat into 3 pieces evenly as the
    # flat lengh may not be able to divided exactly by 3....
    #
    #                                                                   #
    #                                                                   #
    #So, fill zeros at the end of flat to generate padded_flat
    #padded_flat = None # modify this line and fill correct value into padded_flat
    chunks = [padded_flat[i*chunk:(i+1)*chunk] for i in range(world)]

    #                                                                   #
    #                                                                   #
    # your code here: call reduce_scatter and all_gather
    #
    #                                                                   #
    #                                                                   #
    #we provide the reduce_scatter and all_gather func prototype for you
    # You may adjust the function signature (input structure) of `reduce_scatter` and `all_gather` if needed.
    tmp = torch.empty_like(chunks[0])
    reduce_scatter(chunks, tmp, world, rank, left, right)
    all_gather(chunks, tmp, world, rank, left, right)


    # stitch & unpad  
    flat /= world
    tensor.view(-1).copy_(flat[:n])
    return