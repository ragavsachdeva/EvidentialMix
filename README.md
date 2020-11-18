# EvidentialMix: Learning with Combined Open-set and Closed-set Noisy Labels

**Conference:** Accepted at WACV'21

**Paper:** [Arxiv](https://arxiv.org/abs/2011.05704), [Blog](https://ragavsachdeva.github.io/research/2020/evidentialmix/)

**Authors:** Ragav Sachdeva, Filipe R. Cordeiro, Vasileios Belagiannis, Ian Reid, Gustavo Carneiro

#### Usage:

```
python Train_cifar.py --r [total_noise] --on [proportion_of_openset_noise] --data_path [path_to_cifar10] --noisy_dataset [cifar100/imagenet32] --noise_data_dir [path_to_cifar100/imagenet32]
```

Note: ```r``` is the same as _ρ_ in the paper and ```on``` is the same as _(1-ω)_.

#### Acknowledgements:
Thanks to Li et al. for publishing their code for [DivideMix](https://github.com/LiJunnan1992/DivideMix). Our implementation is heavily based on their work.
