import os
import click
import torch

import torch.distributed as dist
import torch.multiprocessing as mp
from trainers import RewardModelTrainer, FSDPRewardModelTrainer, AcceleratorRewardModelTrainer
from config.configs import get_configs
from models.gpt import GPTRewardModel
from utils.dataset import DahoasRMStaticDataset


def setup(rank, world_size):
    pass

def cleanup():
    pass

def train(pretrain, batch_size, exp_name):
    device = "cuda"
    cfg = get_configs("gpt2-medium/lora")
    cfg.batch_size = batch_size
    cfg.pretrain = pretrain
    cfg.total_epochs = 1
    cfg.exp_name = exp_name

    if pretrain == "huggingface":
        rm = GPTRewardModel.from_pretrained(cfg)
    else:
        rm = GPTRewardModel.from_backbone_checkpoint(cfg, pretrain)
    
    train_ds = DahoasRMStaticDataset(block_size=1024,
                                     split='train',
                                     max_examples=None,
                                     tokenizer_name="tiktoken/gpt2")
    test_ds = DahoasRMStaticDataset(block_size=1024,
                                    split='test',
                                    max_examples=None,
                                    tokenizer_name="tiktoken/gpt2")
    trainer = RewardModelTrainer(cfg, device, rm, train_ds, test_ds)
    trainer.fit()

@click.command()
@click.option('--strategy', '-s', default="naive")
@click.option('--pretrain', '-p', default="huggingface")
@click.option('--batch-size', '-b', default=1)
@click.option('--exp-name', '-n', default="default")
def main(strategy, pretrain, batch_size, exp_name):
    torch.manual_seed(1234)

    if strategy == "naive":
        train(pretrain, batch_size, exp_name)


if __name__ == "__main__":
    main()