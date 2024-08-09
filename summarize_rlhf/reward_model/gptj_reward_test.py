import random

import numpy as np
import torch
from datasets import load_dataset
from reward_model import GPTRewardModel
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from train_rewards_model_gptj import create_comparison_dataset, PairwiseDataset, DataCollatorReward

def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.pad_token = tokenizer.eos_token
    PAD_ID = tokenizer(tokenizer.pad_token)["input_ids"][0]

    model = GPTRewardModel("CarperAI/openai_summarize_tldr_sft")
    model.load_state_dict(torch.load("rm_checkpoint/pytorch_model.bin"))
    max_length = 550
    val_pairs = create_comparison_dataset("CarperAI/openai_summarize_comparisons", "test")
    dev_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_length)

    from torch.utils.data import DataLoader

    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=6, collate_fn=DataCollatorReward())
    model.cuda()
    model.eval()
    model.half()
    correct = 0
    chosen_list = []
    reject_list = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader)):
            for x in batch:
                batch[x] = batch[x].cuda()
            outputs = model(**batch)
            correct += sum(outputs["chosen_end_scores"] > outputs["rejected_end_scores"])
            chosen_list.append(outputs["chosen_end_scores"].cpu())
            reject_list.append(outputs["rejected_end_scores"].cpu())
    print("Total accuracy: ", correct / len(dev_dataset))
