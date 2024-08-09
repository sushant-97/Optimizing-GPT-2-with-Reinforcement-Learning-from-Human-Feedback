# Learning to summarize from Human Feedback using `trlx`

This is the reimplementation of how to utilize trlx to train a summarization model with human feedback, following the fine-tuning methods outlined in Stiennon et al.'s, "[Learning to Summarize from human feedback](https://arxiv.org/abs/2009.01325)".

first we need to install dependencies using `requirement.txt` file

```bash
pip install -r requirements.txt
```

### Training Process

1. Train SFT:
    ```bash
    cd sft/ && deepspeed train_gptj_summarize.py
    ```

2. Train Reward Model:
    ```bash
    cd reward_model/ && deepspeed train_reward_model_gptj.py
    ```

3. PPO Training:
    ```bash
    accelerate launch --config_file configs/default_accelerate_config.yaml trlx_gptj_text_summarization.py
    ```

### Results

1. SFT vs PPO
    __ROUGE scores__
    | Model | Rouge-1 | Rouge-2 | Rouge-L | Average |
    | --- | --- | --- | --- |   --- |
    | SFT | 0.334 | 0.125 | 0.261 | 0.240 |
    | PPO | 0.323 | 0.109 | 0.238 | 0.223 |

    __Reward scores__

    | Model | Average Reward | Reward $\Delta$ |
    | --- | --- | --- |
    | SFT | 2.729 | -0.181 |
    | PPO | 3.291 | +0.411 |

## References

1. Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano, "[Learning to Summarize from human feedback](https://arxiv.org/abs/2009.01325)", Neural Information Processing Systems, 2020.
2. Check our blog post for metric logs and other results [here](http://wandb.me/summarize-rlhf-trlx)