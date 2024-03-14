import click
from src.trainers import SFTTrainer_head
from src.gpt import GPT,GPTActor
from src.dataset import SFT_Datasets
from src.configs import get_configs
import time
tic=time.time()
def train(batch_size, exp_name,step,card):
    device =f'cuda:{card}'
    cfg = get_configs("gpt2-medium")
    cfg.max_steps = step 
    cfg.batch_size = batch_size

    cfg.exp_name = exp_name
    model = GPTActor.from_pretrained(cfg)
    train_ds = SFT_Datasets(block_size=256,device=device)
    trainer = SFTTrainer_head(cfg, device, model, train_ds)
    trainer.fit()


@click.command()
@click.option('--batch-size', '-b', default=1)
@click.option('--exp-name', '-n', default="default")
@click.option('--step', '-t', default=5e5)
@click.option('--card', '-card', default=0)



def main( batch_size, exp_name,step,card):
    train(batch_size, exp_name,step,card)


if __name__ == "__main__":
    main()
    toc=time.time()
    print(f"time:{toc-tic}")
