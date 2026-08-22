# LLM-RL-ZERO

implementation of REINFORCE, PPO, GRPO from scratch on Countdown tasks. Core code in [algorithms/](algorithms/)

the environment and training scaffold is from [GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero.git), original [README.md](README.old.md)

## Training(I just finished the core and haven't debug, so maybe not runnable.)

We use the `Qwen2.5-3B-Instruct` model for training. To train the model, run the following commands:

```bash
# initialize the environment
pip install uv
uv sync

# install git-lfs
apt update; apt install git-lfs -y; git lfs install

# download the dataset
git clone https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4

# download the pretrained model
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
# train the model
uv run train.py
# train the model with a 24GB VRAM GPU (e.g., an RTX 4090 GPU)
uv run train.py --config config_24GB.yaml
```
