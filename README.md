Unofficial implementation in PyTorch of the DION optimizer.

Paper Credit! 
Dion: Distributed Orthonormalized Updates
Kwangjun Ahn, Byron Xu, Natalie Abreu, John Langford

https://arxiv.org/abs/2504.05295
https://doi.org/10.48550/arXiv.2504.05295

Note that it is integrated into TorchTitan for distributed training with this PR:
https://github.com/pytorch/torchtitan/pull/1417



And may be easier to test it at scale there. 

Otherwise, there are some basic tests working here as well with simple models.

