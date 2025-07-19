Unofficial implementation in PyTorch of the DION optimizer.

Paper Credit! 
Dion: Distributed Orthonormalized Updates
Kwangjun Ahn, Byron Xu, Natalie Abreu, John Langford

https://arxiv.org/abs/2504.05295
https://doi.org/10.48550/arXiv.2504.05295

Note that it is integrated into TorchTitan for distributed training with this PR:  
https://github.com/pytorch/torchtitan/pull/1417

<img width="1196" height="539" alt="Screenshot 2025-07-18 at 2 14 57 PM" src="https://github.com/user-attachments/assets/e885f74a-4136-4048-9746-bda952caa611" />


And may be easier to test it at scale there. 

Otherwise, there are some basic tests working here as well with simple models.

Code reviews, PRs, errors, or other feedback is appreciated. 
