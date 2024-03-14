import click
import torch
import torch.nn as nn
from src.trainers import PPOTrainer
from src.configs import get_configs
from src.gpt import GPT, GPTActor, GPTCritic
from src.dataset import PPO_Dataset
import pdb


def train(batch_size, exp_name, actor_weights, critic_weights,epoch,card,num_images_per_prompt):
    cfg = get_configs("gpt2-medium")
    
    device=f"cuda:{card}"
    cfg.actor_weights = actor_weights
    cfg.critic_weights = critic_weights
    cfg.reward_model_weights = cfg.critic_weights
    cfg.sft_model_weights = cfg.actor_weights
    cfg.batch_size = batch_size
    cfg.total_epochs = epoch
    cfg.exp_name = exp_name
   
    actor = GPTActor.from_checkpoint(cfg, cfg.actor_weights)
    actor.to(device)


    sft_model = GPTActor.from_checkpoint(cfg, cfg.sft_model_weights)
    sft_model.to(device)


    cfg2 = get_configs("gpt2-medium/lora")
    critic = GPTCritic.from_checkpoint(cfg2, cfg.critic_weights)
    critic.to(device)

    critic.freeze_weights("lora")
    dataset = PPO_Dataset(device=device)
    trainer = PPOTrainer(cfg, actor, critic,  sft_model, dataset,num_images_per_prompt=num_images_per_prompt,device=device)
  
    trainer.fit()


@click.command()
@click.option('--batch-size', '-b', default=2)
@click.option('--exp-name', '-n', default="default")
@click.option('--actor', '-a',default="./runs/stage1_steppt")
@click.option('--critic', '-c',default="./runs/stage1_step.pt")
@click.option('--epoch', '-e', default=1)
@click.option('--card', '-card', default="0") 
@click.option('--num_images_per_prompt', '-num_images_per_prompt', default=2) 
def main( batch_size, exp_name, actor, critic,epoch,card,num_images_per_prompt):
    train(batch_size, exp_name, actor, critic,epoch,card,num_images_per_prompt)


if __name__ == "__main__":
    main()
