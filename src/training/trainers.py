import functools
from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from loss import KPairwiseLoss, CrossEntropyLoss, ValueLoss, PolicyLoss
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
import statistics
# from llama import LLaMA
# from gpt import GPTRewardModel, GPT, GPTCritic, TransformerDecoderBlock, GPTActor
from gpt import GPT, GPTRewardModel
from tqdm import tqdm, trange
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import json
import random
from typing import Union
from accelerate import Accelerator
from torchinfo import summary
from configs import TrainingConfig
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload, )
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy, )
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from tokenizer import TiktokenTokenizer


# import bitsandbytes as bnb


class Trainer:

    def __init__(self) -> None:
        self.model = None
        self.optimizer = None
        random.seed(1)

    def save_hyperparams(self, hp):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')

        with open(f'./runs/{self.run_name}/hyperparams.json', 'w') as fp:
            json.dump(hp, fp, indent=4)

    def save_metrics(self, metrics):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')
        with open(f'./runs/{self.run_name}/metrics.json', 'w') as fp:
            json.dump(metrics, fp, indent=4)

    def save_states(self, step, is_last=False):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')
        file_name = f'{self.run_name}_final.pt' if is_last else f'{self.run_name}_step{step}.pt'
        torch.save(
            {
                'step': step,
                'model_state_dict':
                    self.model.state_dict(),  # Save the unoptimized model
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            f'./runs/{self.run_name}/{file_name}')

class SFTTrainer(Trainer):

    def __init__(self, cfg: TrainingConfig, device, model: nn.Module,
                 train_dataset, test_dataset) -> None:
        super().__init__()
        self.cfg = cfg
        self.run_name = f"sft_{cfg.exp_name}_{datetime.now().strftime('%Y%m%d%H%M')}"
        self.device = device
        assert self.device == 'cuda'
        self.max_steps = cfg.max_steps
        self.eval_freq = 1
        self.save_freq = 20000
        self.train_dataloader = iter(
            DataLoader(train_dataset,
                       batch_size=cfg.batch_size,
                       num_workers=6,
                       pin_memory=True))
        self.test_dataloader = iter(
            DataLoader(test_dataset,
                       batch_size=cfg.batch_size,
                       num_workers=6,
                       pin_memory=True))
        self.model = model
        self.criterion = CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.grad_clip = cfg.grad_clip
        self.dtype = torch.float16

        self.finetune_method = cfg.finetune_method

        hp = {
            "dtype": str(self.dtype),
            "train_dataset": type(train_dataset).__name__,
            "train_dataset_len": len(train_dataset),
            "test_dataset": type(test_dataset).__name__,
            "test_dataset_len": len(test_dataset),
            **cfg.dict(),
        }
        self.save_hyperparams(hp)

    def fit(self):
        if self.finetune_method:
            self.model.freeze_weights(self.finetune_method)
        summary(self.model, input_data=torch.ones(1, 1024).long())

        opt_model = torch.compile(self.model)
        opt_model.to(self.device)
        writer = SummaryWriter(f'./runs/{self.run_name}/logs', max_queue=40)
        scaler = GradScaler(enabled=self.dtype != torch.float32)

        opt_model.train()
        step = 0

        t0 = time.time()
        while step < self.max_steps:
            x, y = next(self.train_dataloader)
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.autocast(device_type=self.device, dtype=self.dtype):
                y_hat = opt_model(x)  # (B, 1)
                loss = self.criterion(y_hat, y)  # (B, 1)

            if self.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(opt_model.parameters(),
                                               self.grad_clip)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            lossf = loss.item()

            iter_time = time.time() - t0
            t0 = time.time()
            print(
                f"step {step}, batch loss {round(lossf, 3)}, {round(1.0 / iter_time, 2)} iters/s"
            )
            writer.add_scalar('Loss/train/step', lossf, step)

            if step != 0 and step % self.save_freq == 0:
                self.save_states(step)

            step += 1

        self.save_states(step, True)


class RewardModelTrainer(Trainer):

    def __init__(self, cfg: TrainingConfig, device, model: nn.Module,
                 train_dataset, test_dataset) -> None:
        super().__init__()
        self.run_name = f"rm_{cfg.exp_name}_{datetime.now().strftime('%Y%m%d%H%M')}"
        self.device = device
        assert self.device == 'cuda'
        self.total_epochs = cfg.total_epochs
        self.eval_freq = 1
        self.save_freq = 30000
        self.model = model
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=cfg.batch_size,
                                           num_workers=8,
                                           shuffle=True,
                                           pin_memory=True)
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=cfg.batch_size,
                                          num_workers=8,
                                          pin_memory=True)
        self.model = model
        self.criterion = KPairwiseLoss()
        self.finetune_method = cfg.finetune_method
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.grad_clip = cfg.grad_clip
        self.dtype = torch.float16

        hp = {
            "dtype": str(self.dtype),
            "train_dataset": type(train_dataset).__name__,
            "train_dataset_len": len(train_dataset),
            "test_dataset": type(test_dataset).__name__,
            "test_dataset_len": len(test_dataset),
            **cfg.dict(),
        }
        self.save_hyperparams(hp)

    
    def fit(self):
        
        #check if LoRA is enabled
        if self.finetune_method:
            self.model.freeze_weights(self.finetune_method)
        summary(self.model, input_data=torch.ones(1,1024).long())

        opt_model = torch.compile(self.model)
        opt_model.to(self.device)
        writer = SummaryWriter(f'./runs/{self.run_name}/logs', max_queue=40)
        scaler = GradScaler(enabled=self.dtype != torch.float32)

        opt_model.train()
        for epoch in range(self.total_epochs):
            for step, (completions, attention_masks) in enumerate(pbar := tqdm(self.train_dataloader)):
                total_steps = step + epoch * len(self.train_dataloader)
                completions = completions.to(self.device)
                attention_masks = attention_masks.to(self.device)

                with torch.autocast(device_type=self.device, dtype=self.dtype):
                    #autocast: operations that benefit from lower precision are executed in float16, while others remain in float32.

                    positive_scores = opt_model(completions[:,0,:],
                                                attention_masks[:,0,:],) # (B, 1)
                    negative_scores = opt_model(completions[:,1,:],
                                                attention_masks[:,1,:],) # (B, 1)
                    loss = self.criterion((positive_scores, negative_scores), dim=-1)    # (B, 2)

                if self.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(opt_model.parameters(),
                                                   self.grad_clip)
                
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                lossf = loss.item()
                writer.add_scalar('Loss/train/step', lossf, total_steps)
                pbar.set_description(f"Epoch {epoch}, step {step}, batch loss {round(lossf, 3)}")

                if total_steps != 0 and total_steps % self.save_freq == 0:
                    self.save_states(total_steps)
            
            if epoch % self.eval_freq == 0:
                opt_model.eval()
                with torch.no_grad():
                    tp = 0
                    total = 0
                    losses = []
                    for step, (completions, attention_masks) in enumerate(
                        self.test_dataloader):
                        completions = completions.to(self.device)
                        attention_masks = attention_masks.to(self.device)

                        positive_scores = opt_model(
                            completions[:, 0, :],
                            attention_masks[:, 0, :])  # (B, 1)
                        negative_scores = opt_model(
                            completions[:, 1, :],
                            attention_masks[:, 1, :])  # (B, 1)
                        loss = self.criterion(
                            torch.cat((positive_scores, negative_scores),
                                      dim=-1))  # (B, 2)
                        lossf = loss.item()
                        losses.append(lossf)
                        writer.add_scalar(
                            'Loss/test/step', lossf,
                            step + epoch * len(self.test_dataloader))
                        tp += torch.count_nonzero(
                            positive_scores > negative_scores)
                        total += positive_scores.shape[0]

                    acc = tp / total
                    epoch_loss = statistics.mean(losses)

                writer.add_scalar('Loss/test/epoch', epoch_loss, epoch)
                writer.add_scalar('Acc/test/epoch', acc, epoch)
                print(f'Epoch: {epoch + 1}, Test Loss: {lossf}, Acc: {acc}')

        self.save_states(total_steps, True)