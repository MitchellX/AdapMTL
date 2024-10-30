# AdapMTL: Adaptive Pruning Framework for Multitask Learning Models

AdapMTL is an adaptive pruning framework tailored for multitask learning (MTL) models. Model compression in MTL introduces unique challenges due to the need for balanced sparsity allocation and accuracy across multiple tasks. AdapMTL addresses these challenges by applying independently learnable soft thresholds to the shared backbone and task-specific heads, capturing sensitivity nuances within each component.

During training, AdapMTL co-optimizes soft thresholds and model weights to automatically determine optimal sparsity for each component, achieving both high task accuracy and overall sparsity. Additionally, it integrates an adaptive weighting mechanism that dynamically adjusts the importance of each task’s loss based on robustness to pruning. Extensive experiments on popular multitask datasets, such as NYU-v2 and Tiny-Taskonomy, across various architectures, demonstrate AdapMTL’s superior performance compared to state-of-the-art pruning methods.


## Environment Setup

Duplicate the environment from the provided YAML file:

```bash
conda env create -f environment.yml
```

## Model Training

### Option 1: Training from Scratch

Example:

```bash
python baseline.py \
    --dataset NYUV2 \
    --architecture mobilenetv2 \
    --iters 100000 \
    --lr 0.0001 \
    --decay_lr_freq 12000 \
    --decay_lr_rate 0.3 \
    --sInit_value -1000 \
    --ratio 0.9 \
    --save_dir ./outputs/test2/
```


### Option 2: Training from a Dense Model or Pruned Model
This is what we do for the comparison methods too.
Please check the scripts in the `Scripts` for more detail.

```
for ratio in 0.9 0.93 0.95 0.97 0.99;do
    python finetune.py \
    --prune_ratio ${ratio} \
    --architecture resnet34 \
    --save_dir ./outputs/unity/resnet34_0.99
done
```

## Model Evaluation

Example:

```bash
python test.py
```

# Project Structure
`main`: Contains all core source code. See main/trainer.py for key training, pruning, and testing code.

`scripts`: Includes useful scripts for training, pruning, and evaluation.
logs: Contains sample results for baseline comparison.

`SOTA_methods`: Houses all comparative methods mentioned in the paper.

`./args.py`: Store the default arguments.



# Citation
If you use our dataset or methods, please cite our work:

```
@article{xiang2024adapmtl,
  title={AdapMTL: Adaptive Pruning Framework for Multitask Learning Model},
  author={Xiang, Mingcan and Tang, Steven Jiaxun and Yang, Qizheng and Guan, Hui and Liu, Tongping},
  journal={arXiv preprint arXiv:2408.03913},
  year={2024}
}

@inproceedings{10.1145/3664647.3681426,
  author = {Xiang, Mingcan and Tang, Jiaxun and Yang, Qizheng and Guan, Hui and Liu, Tongping},
  title = {AdapMTL: Adaptive Pruning Framework for Multitask Learning Model},
  year = {2024},
  booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
  series = {MM '24}
}
```
