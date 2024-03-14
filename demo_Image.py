import torch 
import pdb


from diffusers import UniPCMultistepScheduler
from src.dynamicpipeline import StableDiffusionDynamicPromptPipeline


if __name__=="__main__":
    pipe = StableDiffusionDynamicPromptPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    torch.cuda.empty_cache()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.to("cuda")


    prompt_list=["a red horse on the yellow grass, anime style",
    "a red horse on the yellow grass, [anime:0-1:1.5] style",
    "a red horse on the yellow grass, detailed",
    "a red horse on the yellow grass, [detailed:0.85-1:1]"]
    for i in range(len(prompt_list)):
        prompt=prompt_list[i]
        image0 = pipe(prompt,generator=torch.Generator().manual_seed(1), num_inference_steps=10).images[0]
        image0.save(f"{i:05}.png")

