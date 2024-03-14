
from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.loss import CrossEntropyLoss
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler


from src.gpt import  GPT, GPTCritic, GPTActor
from src.dynamicpipeline import StableDiffusionDynamicPromptPipeline
from tqdm import tqdm
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import json
import random
from typing import Union
import numpy as np


from src.configs import TrainingConfig
import clip

from transformers import AutoProcessor, AutoModel


from transformers import GPT2Tokenizer
from src.loss import  CrossEntropyLoss, ValueLoss, PolicyLoss
import pdb
import re

import json
import pytorch_lightning as pl
import clip
import numpy as np
import torch
import os

from torch import autocast, nn
from diffusers import UniPCMultistepScheduler


class Trainer:
    def __init__(self) -> None:
        self.model = None
        self.optimizer = None
        random.seed(1)

    def save_hyperparams(self, hp):
        if not os.path.exists(f"./runs/{self.run_name}"):
            os.makedirs(f"./runs/{self.run_name}")

        with open(f"./runs/{self.run_name}/hyperparams.json", "w") as fp:
            json.dump(hp, fp, indent=4)

    def save_metrics(self, metrics):
        if not os.path.exists(f"./runs/{self.run_name}"):
            os.makedirs(f"./runs/{self.run_name}")
        with open(f"./runs/{self.run_name}/metrics.json", "w") as fp:
            json.dump(metrics, fp, indent=4)

    def save_states(self, step, is_last=False):
        if not os.path.exists(f"./runs/{self.run_name}"):
            os.makedirs(f"./runs/{self.run_name}")
        file_name = (
            "final.pt" if is_last else f"step{step}.pt"
        )
        torch.save(
            {
                'step': step,
                "model_state_dict": self.model.state_dict(), 
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            f"./runs/{self.run_name}/{file_name}",
        )


@dataclass
class Experience:
    completion: torch.Tensor
    actor_log_probs: torch.Tensor
    w_log_probs: torch.Tensor
    step_log_probs: torch.Tensor
    attention_mask: torch.Tensor
    kl_penalized_reward: torch.Tensor
    advantage: torch.Tensor
    w_advantage: torch.Tensor
    step_advantage: torch.Tensor
    num_actions: int
    estimated_kl: torch.Tensor
    w_estimated_kl: torch.Tensor
    step_estimated_kl: torch.Tensor
    values: torch.Tensor
    action_mask: torch.Tensor





class SFTTrainer(Trainer):
    def __init__(
        self, cfg: TrainingConfig, device, model: nn.Module, train_dataset
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.run_name = (
            f"sft_{cfg.exp_name}_{datetime.now().strftime('%m%d%H%M')}"  # %Y%m%d%H%M
        )
        self.device = device
        # assert self.device == "cuda"
        self.max_steps = cfg.max_steps
        self.eval_freq = 1
        self.save_freq = 1e3  #
        self.train_dataloader = iter(
            DataLoader(
                train_dataset, batch_size=cfg.batch_size, num_workers=6, pin_memory=True
            )
        )

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
            **cfg.dict(),
        }
        self.save_hyperparams(hp)

    def fit(self):
        if self.finetune_method:
            self.model.freeze_weights(self.finetune_method)

        opt_model = torch.compile(self.model)
        opt_model.to(self.device)
        writer = SummaryWriter(f"./runs/{self.run_name}/logs", max_queue=40)
        scaler = GradScaler(enabled=self.dtype != torch.float32)

        opt_model.train()
        step = 0

        t0 = time.time()
        while step < self.max_steps:
            x, y = next(self.train_dataloader)
            x = x.to(self.device) 
            y = y.to(self.device) 

            with torch.autocast(device_type="cuda", dtype=self.dtype):
      
                y_hat = opt_model(x)  # (B, 1) 
                loss = self.criterion(y_hat, y)  # (B, 1)

            if self.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(opt_model.parameters(), self.grad_clip)

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
            writer.add_scalar("Loss/train/step", lossf, step)

            if step != 0 and step % self.save_freq == 0:
                self.save_states(step)

            step += 1


        self.save_states(step, True)




class SFTTrainer_head(Trainer):
    def __init__(
        self, cfg: TrainingConfig, device, model: nn.Module, train_dataset
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.run_name = (
            f"sft_{cfg.exp_name}_{datetime.now().strftime('%m%d%H%M')}"  # %Y%m%d%H%M
        )
        print(f"self.run_name:{self.run_name}")
        self.device = device
        # assert self.device == "cuda"
        self.max_steps = cfg.max_steps
        self.eval_freq = 1
        self.save_freq = 2e4  #
        self.train_dataloader = iter(
            DataLoader(
                train_dataset, batch_size=cfg.batch_size, num_workers=6, pin_memory=True
            )
        )

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
            **cfg.dict(),
        }
        self.save_hyperparams(hp)

    def generate_tensor(self, mean, values, shape):

        random_values = np.random.normal(mean,scale=0.5, size=shape)
        random_values = np.clip(random_values, min(values), max(values))
        random_values = np.round(random_values)
        tensor = torch.tensor(random_values,device=self.device)

        return tensor


    def fit(self):
        if self.finetune_method:
            self.model.freeze_weights(self.finetune_method)

        opt_model =self.model
        opt_model.to(self.device)
        writer = SummaryWriter(f"./runs/{self.run_name}/logs", max_queue=40)
        scaler = GradScaler(enabled=self.dtype != torch.float32)

        opt_model.train()
        step = 0

        t0 = time.time()
        while step < self.max_steps:
            x, y = next(self.train_dataloader)
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.autocast(device_type="cuda", dtype=self.dtype):
 
                y_hat,diffw,diffstep = opt_model(x)  # (B, 1) 

                loss = self.criterion(y_hat, y)  # (B, 1)
                loss_w = self.criterion(diffw,self.generate_tensor(2, [0, 1, 2, 3, 4], y.shape).long())
                loss_step = self.criterion(diffstep, self.generate_tensor(1, [0, 1, 2], y.shape).long())

                loss = loss + loss_w + loss_step




            if self.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(opt_model.parameters(), self.grad_clip)

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
            writer.add_scalar("Loss/train/step", lossf, step)
            writer.add_scalar("loss_w/train/step", loss_w.item(), step)
            writer.add_scalar("loss_step/train/step", loss_step.item(), step)
            if step != 0 and step % self.save_freq == 0 or step==5e4:
                self.save_states(step)

            step += 1



        self.save_states(step, True)





class AestheticMlp(pl.LightningModule):
    def __init__(self, input_size, xcol="emb", ycol="avg_rating"):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class PromptScorer:
    def __init__(self,device="cuda:7",num_images_per_prompt=2,seed=None):
        # init scorer hparams
        self.lambda_aes = 0.05
        self.lambda_clip = 5.0
        self.num_images_per_prompt = num_images_per_prompt
        self.seed=seed
        # init models

        self.device = device
        self.init_clip_model()
        self.init_aesthetic_model()
        self.init_diffusion_model()

        self.init_pickscore_model()


        
        self.eval_data_res = []

    def init_pickscore_model(self):
        self.pick_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.pick_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(self.device)           

     
    def init_diffusion_model(self):

        device = self.device
        
        self.sdmodel_name= "CompVis/stable-diffusion-v1-4"
        dpm_scheduler = UniPCMultistepScheduler.from_pretrained(
            self.sdmodel_name, subfolder="scheduler"
        )

        pipe = StableDiffusionDynamicPromptPipeline.from_pretrained(
            self.sdmodel_name,
            revision="fp16",
            torch_dtype=torch.float16,
            scheduler=dpm_scheduler,
        )
 
        # Disable NSFW detect
        pipe.safety_checker = None
        pipe.feature_extractor = None

        pipe = pipe.to(device)
        pipe.enable_xformers_memory_efficient_attention()
        self.diffusion_pipe = pipe

    def init_clip_model(self):
        # wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
        self.clip_model, self.clip_preprocess = clip.load("ckpt/CLIP_ViT/ViT-L-14.pt", device=self.device, jit=False)

    def init_aesthetic_model(self):
        model = AestheticMlp(768)
        s = torch.load("ckpt/aesthetic/sac+logos+ava1-l14-linearMSE.pth")

        model.load_state_dict(s)
        model.to(self.device)
        model.eval()
        self.aes_model = model



    def get_pick_score(self, prompt, images):
        # device = "cuda:7"
        # preprocess
        if len(images) != len(prompt):
            assert len(images) % len(prompt) == 0
            copied_strings = []
            for pmt in prompt:
                copied_strings.extend([pmt] * 3)
            prompt = copied_strings


        image_inputs = self.pick_processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        
        text_inputs = self.pick_processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)



        with torch.no_grad():
            # embed
            image_embs = self.pick_model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
            text_embs = self.pick_model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
            
            # score
            scores = self.pick_model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            
        
        return scores.cpu().tolist()



    def get_clip_features(self, pil_image, is_batched=False):
        if not is_batched:
            image = self.clip_preprocess(pil_image).unsqueeze(0)
        else:
            images = [self.clip_preprocess(i) for i in pil_image]
            image = torch.stack(images)
        
        image = image.to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
        return image_features

    def get_clip_score(self, image_features, prompt):
        
        tokens = clip.tokenize([prompt], truncate=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            logit = image_features @ text_features.t()
            score = logit.item()
        return score

    def get_clip_score_batched(self, image_features, prompts):
        
        tokens = clip.tokenize(prompts, truncate=True).to(self.device)

        with torch.no_grad():
            if len(image_features) != len(prompts):
                assert len(image_features) % len(prompts) == 0
                tokens = (
                    tokens.unsqueeze(1)
                    .expand(-1, self.num_images_per_prompt, -1)
                    .reshape(-1, tokens.shape[-1])
                )

            text_features = self.clip_model.encode_text(tokens)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            logit = image_features @ text_features.t()
        scores = logit.diag().tolist()
        return scores

    def get_aesthetic_score(self, image_features, is_batched=False):
        features = image_features.cpu().detach().numpy()
        order = 2
        axis = -1
        l2 = np.atleast_1d(np.linalg.norm(features, order, axis))
        l2[l2 == 0] = 1
        im_emb_arr = features / np.expand_dims(l2, axis)
        prediction = self.aes_model(
            torch.from_numpy(im_emb_arr)
            .to(self.device)
            .type(torch.cuda.FloatTensor)
        )
        if is_batched:
            return prediction[:, 0].tolist()
        else:
            return prediction.item()



    def gen_image_batched(self, prompts,save=None,save_path="./image"):
        images = []
        bsz=8
        if self.seed != None:
            for i in range(0, len(prompts), bsz):
                pmpts = prompts[i : i + bsz]
                with autocast("cuda"):
             
                    sub_images = self.diffusion_pipe(
                        pmpts,
                        num_images_per_prompt=self.num_images_per_prompt,
                        num_inference_steps=10,
                        generator=torch.Generator().manual_seed(int(self.seed))
                    ).images
                    images.extend(sub_images)
               
        else:
            for i in range(0, len(prompts), bsz):
                pmpts = prompts[i : i + bsz]
                try:
                    with autocast("cuda"):
                        sub_images = self.diffusion_pipe(
                            pmpts,
                            num_images_per_prompt=self.num_images_per_prompt,
                            num_inference_steps=10,
                        ).images
                        images.extend(sub_images)
                except:
                    print("!!!",pmpts)
                    exit()
        if save!=None:
            os.makedirs(save_path,exist_ok=True)
            [images[i].save(os.path.join(save_path,f'{save[i]:05}.png')) for i in range(len(images))]
        return images



    def get_score_batched(self, prompts, plain_texts, plain_aes_score=None):

        images = self.gen_image_batched(prompts)
        image_features = self.get_clip_features(images, is_batched=True)
        aes_scores = self.get_aesthetic_score(image_features, is_batched=True)

       
        clip_scores = self.get_clip_score_batched(image_features, plain_texts)
        clip_scores = torch.Tensor(clip_scores)
        clip_scores = torch.maximum(clip_scores, torch.zeros_like(clip_scores))

        pick_score = self.get_pick_score(plain_texts,images)
        pick_score = torch.Tensor(pick_score)

        aes_scores = torch.Tensor(aes_scores)


        final_scores =  aes_scores + torch.where(clip_scores > 0.28, 0, 20 * clip_scores - 5.6) +torch.where(pick_score > 18, 0, pick_score-18)
        



        if random.random() < 0.001: 
            print(f"prompt:{prompts}")
            print(f"final_scores:{final_scores}")


        final_scores = final_scores.reshape(-1, self.num_images_per_prompt).mean(1).to(self.device)

        return final_scores


class PPOTrainer(Trainer):
    def __init__(
        self,
        cfg: TrainingConfig,
        actor: GPTActor,
        critic: GPTCritic,
        sft_model: GPTActor,
        train_dataset,
        device,

        num_images_per_prompt=None
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.run_name = f"ppo_{cfg.exp_name}_{datetime.now().strftime('%m%d%H')}"
        print(f"self.run_name:{self.run_name}")
        self.device = device
        self.max_new_tokens = 77
        self.pattern = r'\[([^]]*):0-1:1\.0\]'#r'\[(\s*\w+):0-1:1\.0\]'
        self.orig_actor = actor
        self.orig_critic = critic
        self.orig_sft_model = sft_model


        self.actor = torch.compile(self.orig_actor)#self.orig_actor#
        self.critic = torch.compile(self.orig_critic)#self.orig_critic#
        self.sft_model = torch.compile(self.orig_sft_model)#self.orig_sft_model#t

        self.scorer =PromptScorer(device=device,num_images_per_prompt=num_images_per_prompt)


        # Separate actor loss from critic loss to save optimizer memory
        self.actor_criterion = PolicyLoss()
        self.critic_criterion = ValueLoss()

        self.step_dict={
            0:self.actor.tokenizer.encode("0-0.5"),
            1:self.actor.tokenizer.encode("0-1"), 
            2:self.actor.tokenizer.encode("0.5-1"),
        }

        self.w_dict={
            0:self.actor.tokenizer.encode("0.5"),
            1:self.actor.tokenizer.encode("0.75"), 
            2:self.actor.tokenizer.encode("1"),
            3:self.actor.tokenizer.encode("1.25"), 
            4:self.actor.tokenizer.encode("1.5"),
        }

        self.token_dict={
            ",":self.actor.tokenizer.encode(","),
            ".":self.actor.tokenizer.encode("."),
            ":":self.actor.tokenizer.encode(":"),
            " [":self.actor.tokenizer.encode(" ["),
            "[":self.actor.tokenizer.encode("["),
            "]":self.actor.tokenizer.encode("]"),
            " ":self.actor.tokenizer.encode(" "),
        }

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            num_workers=12,
            prefetch_factor=4,
            pin_memory=True,
        )
        
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=cfg.actor_lr,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta1),
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=cfg.critic_lr,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta1),
        )

        self.step=0

        self.writer = SummaryWriter(f"./runs/{self.run_name}/logs", max_queue=50)
        self.total_epochs = cfg.total_epochs
        self.debug = False
        self.save_freq = 1000
        self.dtype = torch.float16



        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.finetune_method = cfg.finetune_method

        hp = {
            "max_new_tokens": self.max_new_tokens,
            "train_dataset": type(train_dataset).__name__,
            "train_dataset_len": len(train_dataset),
            "dtype": str(self.dtype),
            **cfg.dict(),
        }
        self.save_hyperparams(hp)
        print("Initialized PPO Trainer")




    def trans_token(self,bef_list,diffw_list,diffstep_list):
        if len(bef_list)==0:
            return bef_list
        aft_list=torch.tensor([],device=bef_list.device)

        ind=0
        token = bef_list[ind]
        if not (token==self.token_dict[","] or token==self.token_dict["."]): 
            special_token_ind_list=[]
            while not (token==self.token_dict[","] or token==self.token_dict["."] or token==self.token_dict[" "]):
                token = bef_list[ind]
                aft_list=torch.cat([aft_list,token.unsqueeze(0)])
                ind+=1
                if ind>=(len(bef_list)):
                    break
            if ind<(len(bef_list)):
                token = bef_list[ind] 
            while ind<(len(bef_list)) and  not (token==self.token_dict[","] or token==self.token_dict["."] or self.tokenizer.decode([token.long()]).startswith(" ")): 
   
                if token==self.token_dict[" "]:
                    aft_list=torch.cat([aft_list,token.unsqueeze(0)])
                    ind+=1
                    if ind>=(len(bef_list)):
                        break
                    token = bef_list[ind]
                else:
                    special_token_ind_list.append(ind)
                    
                    ind+=1
                    if ind>=(len(bef_list)):
                        break
                    token = bef_list[ind]
                    

                    if token==self.token_dict[","] or token==self.token_dict["."]:
                        break


            
            try:
                w_counts = torch.bincount(diffw_list[special_token_ind_list])
                w_mode=int(torch.argmax(w_counts).item())
            except:
                w_mode=2

            try:
                counts = torch.bincount(diffstep_list[special_token_ind_list])
                mode=int(torch.argmax(counts).item())
            except:
                mode=1
            

            for ind in special_token_ind_list:
                aft_list=torch.cat([aft_list,self.token_dict["["].unsqueeze(0)])
                s_token = bef_list[ind]
                aft_list=torch.cat([aft_list,s_token.unsqueeze(0)])
                aft_list=torch.cat([aft_list,self.token_dict[":"].unsqueeze(0)])
                aft_list=torch.cat([aft_list,self.step_dict[mode]])
                aft_list=torch.cat([aft_list,self.token_dict[":"].unsqueeze(0)])
                aft_list=torch.cat([aft_list,self.w_dict[w_mode]])
                aft_list=torch.cat([aft_list,self.token_dict["]"].unsqueeze(0)])
            ind+=1

            while ind < len(bef_list):
                
                token = bef_list[ind]    

                if not (token==self.token_dict[","] or token==self.token_dict["."]): 
                    aft_list=torch.cat([aft_list,token.unsqueeze(0)])
                    ind+=1
                else: 
                    ind+=1
                    if ind >= len(bef_list):
                        break
                    token = bef_list[ind]
                    special_token_ind_list=[]
                    while not (token==self.token_dict[","] or token==self.token_dict["."]):
                        special_token_ind_list.append(ind)
                        ind+=1
                        if ind>=(len(bef_list)):
                            break
                        token = bef_list[ind]
                        
                        if token==self.token_dict[","] or token==self.token_dict["."]:
                            break


                    aft_list=torch.cat([aft_list,self.token_dict[","].unsqueeze(0)])
                    
                    try:
                        w_counts = torch.bincount(diffw_list[special_token_ind_list])
                        w_mode=int(torch.argmax(w_counts).item())
                    except:
                        w_mode=2

                    try:
                        counts = torch.bincount(diffstep_list[special_token_ind_list])
                        mode=int(torch.argmax(counts).item())
                    except:
                        mode=1
                    

                    for ind in special_token_ind_list:
                        aft_list=torch.cat([aft_list,self.token_dict["["].unsqueeze(0)])
                        s_token = bef_list[ind]
                        aft_list=torch.cat([aft_list,s_token.unsqueeze(0)])
                        aft_list=torch.cat([aft_list,self.token_dict[":"].unsqueeze(0)])
                        aft_list=torch.cat([aft_list,self.step_dict[mode]])
                        aft_list=torch.cat([aft_list,self.token_dict[":"].unsqueeze(0)])
                        aft_list=torch.cat([aft_list,self.w_dict[w_mode]])
                        aft_list=torch.cat([aft_list,self.token_dict["]"].unsqueeze(0)])
                    ind+=1

            
        else:
            while ind < len(bef_list):
                
                token = bef_list[ind]    

                if not (token==self.token_dict[","] or token==self.token_dict["."]): 
                    aft_list=torch.cat([aft_list,token.unsqueeze(0)])
                    ind+=1
                else:
                    ind+=1
                    if ind >= len(bef_list):
                        break
                    token = bef_list[ind]
                    special_token_ind_list=[]
                    while not (token==self.token_dict[","] or token==self.token_dict["."]): 
                        special_token_ind_list.append(ind)
                        ind+=1
                        if ind>=(len(bef_list)):
                            break
                        token = bef_list[ind]
                        

                        if token==self.token_dict[","] or token==self.token_dict["."]:
                            break

                    aft_list=torch.cat([aft_list,self.token_dict[","].unsqueeze(0)])
                    
                    try:
                        w_counts = torch.bincount(diffw_list[special_token_ind_list])
                        w_mode=int(torch.argmax(w_counts).item())
                    except:
                        w_mode=2

                    try:
                        counts = torch.bincount(diffstep_list[special_token_ind_list])
                        mode=int(torch.argmax(counts).item())
                    except:
                        mode=1
                    

                    for ind in special_token_ind_list:

                        aft_list=torch.cat([aft_list,self.token_dict["["].unsqueeze(0)])
                        s_token = bef_list[ind]
                        aft_list=torch.cat([aft_list,s_token.unsqueeze(0)])
                        aft_list=torch.cat([aft_list,self.token_dict[":"].unsqueeze(0)])
                        aft_list=torch.cat([aft_list,self.step_dict[mode]])
                        aft_list=torch.cat([aft_list,self.token_dict[":"].unsqueeze(0)])
                        aft_list=torch.cat([aft_list,self.w_dict[w_mode]])
                        aft_list=torch.cat([aft_list,self.token_dict["]"].unsqueeze(0)])
                    ind+=1

            
        return aft_list





    def save_states(self, step, is_last=False):
        file_name = (
            "actor_final.pt"
            if is_last
            else f"actor_step{step}.pt"
        )
        torch.save(
            {
                "step": step,
                "model_state_dict": self.orig_actor.state_dict(),  # Save the unoptimized model
                "optimizer_state_dict": self.actor_optimizer.state_dict(),
            },
            f"./runs/{self.run_name}/{file_name}",
        )
        file_name = (
            f"critic_final.pt"
            if is_last
            else f"critic_step{step}.pt"
        )
        torch.save(
            {
                "step": step,
                "model_state_dict": self.orig_critic.state_dict(),
                "optimizer_state_dict": self.critic_optimizer.state_dict(),
            },
            f"./runs/{self.run_name}/{file_name}",
        )

    def kl_penalized_reward(
        self,
        reward: torch.Tensor,
        log_prob_rl: torch.Tensor,
        log_prob_sft: torch.Tensor,
        action_mask: torch.Tensor = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        # log(π_RL(y|x) / π_SFL(y|x)) = log(π_RL(y|x)) - log(π_SFL(y|x))
        ratio = log_prob_rl - log_prob_sft
        # k3 in http://joschu.net/blog/kl-approx.html
        estimated_kl = (torch.exp(ratio) - 1) - ratio
        if action_mask:
            estimated_kl = estimated_kl * action_mask
            estimated_kl.sum(dim=1) / action_mask.sum(dim=1)
        estimated_kl = estimated_kl.mean(dim=1, keepdim=True)  # estimated_kl -> (B, 1)
        return reward - self.cfg.kl_beta * estimated_kl, estimated_kl

    @torch.no_grad()
    def make_experience(self, idx, input_masks, input_lengths):
        # self.reward_model.eval()
        self.sft_model.eval()
        self.actor.eval()
        self.critic.eval()


        (
            completion, 
            attention_mask,
            num_actions, 
            action_mask,diffw_list,diffstep_list
        ) = self.actor.batch_generate(
            idx,
            input_masks,
            input_lengths,
            self.max_new_tokens,
            temperature=1.0,
            top_k=50,
        )
        
        if self.debug:
            print(" --- Make Experience --- ")
            print("completion", completion.shape)
            print("input_masks", input_masks.shape)
            print("num_actions", num_actions)
            print("action_mask", action_mask.shape)
            print("idx", idx.shape)
            print("input_masks", input_masks.shape)
  
        actor_log_probs,w_log_probs,step_log_probs = self.actor.forward_actor(
            completion, attention_mask, num_actions  # (B, num_actions)
        )
        sft_log_probs,sft_w_log_probs,sft_step_log_probs = self.sft_model.forward_actor(
            completion, attention_mask, num_actions
        )  # (B, num_actions)

        values = self.critic.forward_critic(completion, attention_mask, num_actions).view(
            -1, 1
        )  # (B, 1)


        input_prompt=[ self.tokenizer.decode(completion[i,:input_lengths[i]]) for i in range(completion.size(0))]

        
        output_prompt=[]
        target =  [torch.tensor(220,device=completion.device), torch.tensor(50256,device=completion.device)]
        target_value = torch.tensor(50256,device=completion.device)
        for i in range(completion.size(0)):
            
            res=completion[i,input_lengths[i]:]
            
            input_w=diffw_list[i,input_lengths[i]:]
            input_step=diffstep_list[i,input_lengths[i]:]
            indices = [i for i, sublist in enumerate(zip(res, res[1:])) if list(sublist) == target]
            if len(indices) > 0:
                
                end=int(indices[0])
                res=res[:end]   
                input_w=input_w[:end]
                input_step=input_step[:end]
            
            
            if target_value in res:
                end = res.cpu().numpy().tolist().index(target_value)
                res=res[:end]   
                input_w=input_w[:end]
                input_step=input_step[:end]

   
            output_tokens=self.trans_token(res,input_w,input_step)
            res=self.tokenizer.decode(torch.cat([completion[i,:input_lengths[i]], output_tokens]))

            end = res.find("[<|endoftext|>")
            if end > 0:
                res=res[:end]        
            end = res.find("<|endoftext|>")
            if end > 0:
                res=res[:end]  
    
            res=re.sub(self.pattern, r'\1', res)      
            output_prompt.append(res)
      
        
        
        reward=self.scorer.get_score_batched(prompts=output_prompt,plain_texts=input_prompt).unsqueeze(1) #(B,1)



        if self.debug:
            print("actor_log_probs", actor_log_probs.shape)
            print("sft_log_probs", sft_log_probs.shape)
            print("values", values.shape)
            print("reward", reward.shape)
        
        kl_penalized_reward, estimated_kl = self.kl_penalized_reward(
            reward, actor_log_probs, sft_log_probs
        )

        w_kl_penalized_reward, w_estimated_kl = self.kl_penalized_reward(
            reward, w_log_probs, sft_w_log_probs
        )

        step_kl_penalized_reward, step_estimated_kl = self.kl_penalized_reward(
            reward, step_log_probs, sft_step_log_probs
        )
        

        advantage = kl_penalized_reward - values 
        w_advantage = w_kl_penalized_reward - values
        step_advantage = step_kl_penalized_reward - values


        if self.debug:
            print("kl_penalized_reward", kl_penalized_reward)
            print("advantage", advantage.shape) #[B, 1]

        return Experience(
            completion,
            actor_log_probs,
            w_log_probs,
            step_log_probs,
            attention_mask,
            kl_penalized_reward,
            advantage,
            w_advantage,
            step_advantage,
            num_actions,
            estimated_kl,
            w_estimated_kl,
            step_estimated_kl,
            values,
            action_mask,
        )

    def fit(self):
        scaler = GradScaler(enabled=self.dtype != torch.float32)
        
        print(f"self.total_epochs: {self.total_epochs} self.train_dataloader:{len(self.train_dataloader)}")
        for epoch in range(self.total_epochs):
            for step, (prompt, input_masks, input_lengths) in enumerate(
                pbar := tqdm(self.train_dataloader)
            ):
          
                step=step+self.step

                if len(prompt.shape)==3:
                    prompt=prompt.squeeze(1)
                    input_masks=input_masks.squeeze(1)
                prompt, input_masks, input_lengths = (
                    prompt.to(self.device),
                    input_masks.to(self.device),
                    input_lengths.to(self.device),
                )
                
 
                if self.debug:
                    print("prompt", prompt.shape)
         
                max_input_length = torch.max(input_lengths)
                prompt = prompt[:, :max_input_length]
                if self.debug:
                    print("input_lengths", input_lengths)
                    print("prompt after", prompt.shape)

                total_steps = step + epoch * len(self.train_dataloader)

                with torch.autocast(
                    device_type="cuda",
                    dtype=self.dtype,
                    enabled=self.dtype != torch.float32,
                ):
                    experience = self.make_experience(
                        prompt, input_masks, input_lengths
                    )
      
                    self.actor.train()
                    curr_actor_log_probs, diffw_log_probs, diffstep_log_probs = self.actor.forward_actor(
                        experience.completion, 
                        experience.attention_mask,
                        experience.num_actions,
                    )

                    if self.debug:
                        print("curr_actor_log_probs", curr_actor_log_probs.shape)
                        print("actor_log_probs", experience.actor_log_probs.shape)

                    actor_loss_token = self.actor_criterion( 
                        curr_actor_log_probs,
                        experience.actor_log_probs,
                        experience.advantage, 
                        experience.action_mask,
                    ) 

                    actor_loss_w=self.actor_criterion( 
                        diffw_log_probs,
                        experience.w_log_probs,
                        experience.w_advantage,
                        experience.action_mask,
                    )
                    
                    actor_loss_step = self.actor_criterion(
                        diffstep_log_probs,
                        experience.step_log_probs,
                        experience.step_advantage,
                        experience.action_mask,
                    )
                    actor_loss = actor_loss_token + actor_loss_w + actor_loss_step

                    scaler.scale(actor_loss).backward()
                    scaler.step(self.actor_optimizer)
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    actor_lossf = actor_loss.item()

                    self.critic.train()
                    new_values = self.critic.forward_critic(
                        experience.completion,
                        experience.attention_mask,
                        experience.num_actions,
                    ).view(-1, 1)

                    if self.debug:
                        print("new_value", new_values.shape)
                        print("reward", experience.kl_penalized_reward.shape)

                    critic_loss = self.critic_criterion(
                        new_values,
                        experience.kl_penalized_reward,
                        experience.values,
                        experience.action_mask,
                    )

                    scaler.scale(critic_loss).backward()
                    scaler.step(self.critic_optimizer)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    critic_lossf = critic_loss.item()

                    scaler.update()

                self.writer.add_scalar(
                    "KL", experience.estimated_kl.mean(), total_steps
                )
                self.writer.add_scalar(
                    "mean_advantage", experience.advantage.mean(), total_steps
                )
                self.writer.add_scalar(
                    "mean_reward", experience.kl_penalized_reward.mean(), total_steps
                )
                self.writer.add_scalar("mean_value", new_values.mean(), total_steps)
                self.writer.add_scalar("Loss/actor/step", actor_lossf, total_steps)
                self.writer.add_scalar("Loss/token/step", actor_loss_token.item(), total_steps)
                self.writer.add_scalar("Loss/w/step", actor_loss_w.item(), total_steps)
                self.writer.add_scalar("Loss/step/step", actor_loss_step.item(), total_steps)
                self.writer.add_scalar("Loss/critic/step", critic_lossf, total_steps)
                
                pbar.set_description(
                    f"actor loss {round(actor_lossf, 3)}, critic loss {round(critic_lossf, 3)}"
                )

                if (
                    (total_steps==500 or (total_steps != 0 and total_steps % self.save_freq == 0))
                ):
                    self.save_states(total_steps)

   
        self.save_states(None, True)




