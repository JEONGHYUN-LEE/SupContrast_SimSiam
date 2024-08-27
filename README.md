# SupSiam

This repository provides a PyTorch implementation that modifies the **formulation of [supervised contrastive learning](https://arxiv.org/abs/2004.11362) in a manner similar to [SimSiam](https://arxiv.org/abs/2011.10566)**. It offers the following advantages over original supervised contrastive learning:

- It can prevent representation collapse **without the use of negative samples**, enabling stable training even with small batch sizes.
- It demonstrates **higher test accuracy** compared to traditional supervised contrastive learning (based on ResNet-18 and CIFAR-10 benchmarks).




## Contributors

- Jeonghyun Lee (nomar0107@korea.ac.kr)
- Sungmin Han (sungmin_15@korea.ac.kr)

- Main frame of this repository is based on the work: 
    - [SimSiam-91.9-top1-acc-on-CIFAR10](https://github.com/Reza-Safdari/SimSiam-91.9-top1-acc-on-CIFAR10).

## Usage

To train representation function with supervised contrastive learning, run:
    
```bash
    python main.py --data_root Dataset_Path  --arch resnet18 --learning_rate 0.06 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --exp_dir Save_Path
```



To train linear classifier, run:

```bash
    python main_lincls.py --arch resnet18 --num_cls 10 --batch_size 256 --lr 30.0 --weight_decay 0.0 --pretrained Pretrained_CKPT Dataset_Path
```

## Results

Here are the benchmarking results for CIFAR-10 and ResNet-18 on a single `RTX A5000`:



|                                                    | Pre-train<br/>epochs | Linear fine-tuning<br/>epochs   | Batch size | Top-1 Test Acc. | Time per Epoch <br/> (Sec) | GPU Memory (MiB) |  Weights |
|----------|:----:|:---:|:---:|:---:|:---:|:---:|:---:|
|[SupCon](https://github.com/HobbitLong/SupContrast) | 1000           |  100   |            1024                |          94.32             |  20.31  |  15620  | [download](https://drive.google.com/file/d/12sWzXKu6mHzMyhMb2biLsdCBlGqCou0s/view?usp=drive_link) |
|**SupSiam**                                             | **800**             |**100**    | **512** | **95.58**          |  **20.05**  |                **6508**                       |[download](https://drive.google.com/file/d/1CGzZhE-k-5SK-tQt5x9nkHWwrMVuwIJ4/view?usp=sharing) |
