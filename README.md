# Optimizing-GPT-2-with-Reinforcement-Learning-from-Human-Feedback

ChatGPT has three stages, which is based on InstructGPT which was the last open publication on this topic from OpenAI. The first state is Supervised Finetuning stage or Instruct Tuning stage where model is traning on different taks(InstructGPT was trained on 30K task) in supervised autoregressive way. 
In second stage we perform reward model traning and in last stage we perform RL with human feeback with Proximal Policy Gradient. Here's a diagram from the InstructGPT paper:

![InstructGPT](assets/instructgpt.png)
 

# Directory
```bash
src
  |_train_ppo.py # training script for PPO 
  |_train_rm.py # trianing script for Reward Model
  |_train_sft.py # training script for SFT model
  |_tariners.py # the actual training loops and other trainer utilities, such as saving states
  |_loss.py # loss functions used in different training
  |_main.py # some scratch code to quickly test something
  |_gpt.py # GPT-2 implementation with LoRA
  |_evaluate.py # evaluate the generation with ChatGPT
  |_dataset.py # multiple datasets definition
  |_tokenizer.py # tokenizers in a unified class
  |_llama.py # wish I could have more time to test with LLaMA
init_debian.sh # in case you need to initialize a debian system from scratch
requirements.txt # dependencies without PyTorch! Install your own pytorch 2.0 nightly.
```