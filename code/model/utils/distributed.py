""" Utility functions for distributed training setup"""

###### import libraries ######
# system libraries
import os
import re

# ML libraries
import torch
import torch.distributed as dist

def setup_distributed():
    """Initialize distributed training with SLURM environment variables"""
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ.get('SLURM_LOCALID', rank % torch.cuda.device_count()))
        
        # Set device BEFORE any distributed operations
        torch.cuda.set_device(local_rank)
        
        # MASTER_ADDR and MASTER_PORT are set by the bash script
        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ['MASTER_PORT']
        
        # Verify IPv4 format
        if not re.match(r'^\d+\.\d+\.\d+\.\d+$', master_addr):
            raise ValueError(f"MASTER_ADDR must be IPv4 format, got: {master_addr}")
        
        if rank == 0:
            print(f"✓ Master node: {master_addr}:{master_port} (IPv4)")
            print(f"✓ World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
        
        # Force IPv4 socket family
        os.environ['NCCL_SOCKET_FAMILY'] = 'AF_INET'
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        # init
        dist.init_process_group('nccl', device_id=local_rank)
        
        # Verify initialization
        if rank == 0:
            print(f"✓ Distributed initialization successful")
            print(f"✓ Backend: {dist.get_backend()}")
            print(f"✓ World size from DDP: {dist.get_world_size()}")
            print(f"✓ Rank from DDP: {dist.get_rank()}")
        
    else:
        rank = 0
        local_rank = 0
        world_size = 1
    
    return rank, local_rank, world_size

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()