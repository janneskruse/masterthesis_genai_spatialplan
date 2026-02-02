""" Utility functions for distributed training setup"""

###### import libraries ######
# system libraries
import os
import re

# ML libraries
import torch
import torch.distributed as dist

def setup_distributed():
    """Initialize distributed training with SLURM or torchrun environment variables"""
    
    # Debug: print all relevant env vars
    print(f"[DDP DEBUG] RANK={os.environ.get('RANK')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}")
    print(f"[DDP DEBUG] SLURM_PROCID={os.environ.get('SLURM_PROCID')}, SLURM_NTASKS={os.environ.get('SLURM_NTASKS')}, SLURM_LOCALID={os.environ.get('SLURM_LOCALID')}")
    print(f"[DDP DEBUG] CUDA device count: {torch.cuda.device_count()}")
    
    # Check for torchrun environment variables FIRST (preferred for interactive runs)
    # torchrun sets LOCAL_RANK which SLURM batch jobs don't set directly
    if 'LOCAL_RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        print(f"[DDP DEBUG] Using torchrun path: rank={rank}, local_rank={local_rank}, world_size={world_size}")
        
        # Set device BEFORE any distributed operations
        torch.cuda.set_device(local_rank)
        print(f"[DDP DEBUG] Set CUDA device to {local_rank}, current device: {torch.cuda.current_device()}")
        
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '29500')
        
        if rank == 0:
            print(f"✓ [torchrun] Master node: {master_addr}:{master_port}")
            print(f"✓ [torchrun] World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
        
        # Force IPv4 socket family
        os.environ['NCCL_SOCKET_FAMILY'] = 'AF_INET'
        
        # init (torchrun already sets up the rendezvous, just call init)
        dist.init_process_group('nccl', device_id=local_rank)
        
        if rank == 0:
            print(f"✓ Distributed initialization successful (torchrun)")
            print(f"✓ Backend: {dist.get_backend()}")
            print(f"✓ World size from DDP: {dist.get_world_size()}")
    
    # Check for SLURM environment variables (batch jobs via srun)
    elif 'SLURM_PROCID' in os.environ and 'SLURM_NTASKS' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ.get('SLURM_LOCALID', rank % torch.cuda.device_count()))
        
        print(f"[DDP DEBUG] Using SLURM path: rank={rank}, local_rank={local_rank}, world_size={world_size}")
        
        # Set device BEFORE any distributed operations
        torch.cuda.set_device(local_rank)
        print(f"[DDP DEBUG] Set CUDA device to {local_rank}, current device: {torch.cuda.current_device()}")
        
        # MASTER_ADDR and MASTER_PORT are set by the bash script
        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ['MASTER_PORT']
        
        # Verify IPv4 format
        if not re.match(r'^\d+\.\d+\.\d+\.\d+$', master_addr):
            raise ValueError(f"MASTER_ADDR must be IPv4 format, got: {master_addr}")
        
        if rank == 0:
            print(f"✓ [SLURM] Master node: {master_addr}:{master_port} (IPv4)")
            print(f"✓ [SLURM] World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
        
        # Force IPv4 socket family
        os.environ['NCCL_SOCKET_FAMILY'] = 'AF_INET'
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        # init
        dist.init_process_group('nccl', device_id=local_rank)
        
        # Verify initialization
        if rank == 0:
            print(f"✓ Distributed initialization successful (SLURM)")
            print(f"✓ Backend: {dist.get_backend()}")
            print(f"✓ World size from DDP: {dist.get_world_size()}")
            print(f"✓ Rank from DDP: {dist.get_rank()}")
    else:
        # Single GPU mode
        rank = 0
        local_rank = 0
        world_size = 1
    
    return rank, local_rank, world_size

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()