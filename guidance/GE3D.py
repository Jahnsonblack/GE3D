from audioop import mul
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DDPMScheduler, DDIMScheduler, EulerDiscreteScheduler, \
                      EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, ControlNetModel, \
                      DDIMInverseScheduler, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
from pathlib import Path
import os
import random

import cv2
import torchvision.transforms as T
# suppress partial model loading warning
logging.set_verbosity_error()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.cuda.amp import custom_bwd, custom_fwd
from .perpneg_utils import weighted_perpendicular_aggregator

from .sd_step import *

def rgb2sat(img, T=None):
    max_ = torch.max(img, dim=1, keepdim=True).values + 1e-5
    min_ = torch.min(img, dim=1, keepdim=True).values
    sat = (max_ - min_) / max_
    if T is not None:
        sat = (1 - T) * sat
    return sat

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, vram_O, t_range=[0.02, 0.98], max_t_range=0.98, num_train_timesteps=None, 
                 ddim_inv=False, use_control_net=False, textual_inversion_path = None, 
                 LoRA_path = None, guidance_opt=None):
        super().__init__()

        self.device = device
        self.precision_t = torch.float16 if fp16 else torch.float32

        print(f'[INFO] loading stable diffusion...')

        model_key = guidance_opt.model_key
        assert model_key is not None

        is_safe_tensor = guidance_opt.is_safe_tensor
        base_model_key = "stabilityai/stable-diffusion-v1-5" if guidance_opt.base_model_key is None else guidance_opt.base_model_key # for finetuned model only

        if is_safe_tensor:
            pipe = StableDiffusionPipeline.from_single_file(model_key, use_safetensors=True, torch_dtype=self.precision_t, load_safety_checker=False)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)

        self.ism = not guidance_opt.sds
        self.scheduler = DDIMScheduler.from_pretrained(model_key if not is_safe_tensor else base_model_key, subfolder="scheduler", torch_dtype=self.precision_t)
        self.sche_func = ddim_step

        if use_control_net:
            controlnet_model_key = guidance_opt.controlnet_model_key
            self.controlnet_depth = ControlNetModel.from_pretrained(controlnet_model_key,torch_dtype=self.precision_t).to(device)

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            pipe.enable_model_cpu_offload()

        pipe.enable_xformers_memory_efficient_attention()

        pipe = pipe.to(self.device)
        if textual_inversion_path is not None:
            pipe.load_textual_inversion(textual_inversion_path)
            print("load textual inversion in:.{}".format(textual_inversion_path))
        
        if LoRA_path is not None:
            from lora_diffusion import tune_lora_scale, patch_pipe
            print("load lora in:.{}".format(LoRA_path))
            patch_pipe(
                pipe,
                LoRA_path,
                patch_text=True,
                patch_ti=True,
                patch_unet=True,
            )
            tune_lora_scale(pipe.unet, 1.00)
            tune_lora_scale(pipe.text_encoder, 1.00)

        self.pipe = pipe
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        
        self.num_train_timesteps = num_train_timesteps if num_train_timesteps is not None else self.scheduler.config.num_train_timesteps        
        self.scheduler.set_timesteps(self.num_train_timesteps, device=device)

        self.timesteps = torch.flip(self.scheduler.timesteps, dims=(0, ))
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.warmup_step = int(self.num_train_timesteps*(max_t_range-t_range[1]))

        self.noise_temp = None
        self.noise_gen = torch.Generator(self.device)
        self.noise_gen.manual_seed(guidance_opt.noise_seed)

        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        self.rgb_latent_factors = torch.tensor([
                    # R       G       B
                    [ 0.298,  0.207,  0.208],
                    [ 0.187,  0.286,  0.173],
                    [-0.158,  0.189,  0.264],
                    [-0.184, -0.271, -0.473]
                ], device=self.device)
        

        print(f'[INFO] loaded stable diffusion!')
        self.w = self.cauculate_w(guidance_opt.iterations+1,guidance_opt.n)

    def augmentation(self, *tensors):
        augs = T.Compose([
                        T.RandomHorizontalFlip(p=0.5),
                    ])
        
        channels = [ten.shape[1] for ten in tensors]
        tensors_concat = torch.concat(tensors, dim=1)
        tensors_concat = augs(tensors_concat)

        results = []
        cur_c = 0
        for i in range(len(channels)):
            results.append(tensors_concat[:, cur_c:cur_c + channels[i], ...])
            cur_c += channels[i]
        return (ten for ten in results)


    def editing_process(self, latents, noise, jumping_list, text_embeddings=None, cfg=1.0, eta=0.0):
        text_embeddings = text_embeddings.to(self.precision_t)
        neg_text_embeddings = text_embeddings[0:4] 
        unet = self.unet

        cur_noisy_lat  = self.scheduler.add_noise(latents, noise, self.timesteps[jumping_list[0]])
        cur_ind_t = jumping_list[0]
        add_scores = []
        del_scores = []

        for next_ind_t in jumping_list[1:]:
            cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(self.precision_t)
            latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_])
            timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
            unet_output = unet(latent_model_input, timestep_model_input, 
                                encoder_hidden_states=text_embeddings).sample
            add_scores.append((cur_ind_t, cur_noisy_lat, unet_output))
            neg, cond = torch.chunk(unet_output, chunks=2)
            unet_output = neg
            delta_t_ =  next_ind_t - cur_ind_t
            cur_noisy_lat = self.sche_func(self.scheduler, unet_output, cur_ind_t, cur_noisy_lat, -delta_t_, eta).prev_sample
            cur_ind_t = next_ind_t

            del unet_output
            torch.cuda.empty_cache()
        #反转    
        inverse_list = list(reversed(jumping_list))

        for next_ind_t in inverse_list[1:]:
    
            cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(self.precision_t)
            latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_])
            timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
            unet_output = unet(latent_model_input, timestep_model_input, 
                            encoder_hidden_states=text_embeddings).sample
            
            del_scores.append((cur_ind_t, cur_noisy_lat, unet_output))
            neg, cond= torch.chunk(unet_output, chunks=2)
            unet_output = neg + cfg * (cond - neg) 
            
            delta_t_ =  next_ind_t-cur_ind_t

            cur_noisy_lat = self.sche_func(self.scheduler, unet_output, cur_ind_t, cur_noisy_lat, -delta_t_, eta).prev_sample
            cur_ind_t = next_ind_t

            del unet_output
            torch.cuda.empty_cache()
        
        cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(self.precision_t)
        latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_])
        timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
        unet_output = unet(latent_model_input, timestep_model_input, 
                        encoder_hidden_states=text_embeddings).sample
        del_scores.append((cur_ind_t, cur_noisy_lat, unet_output))
        
        del unet_output
        torch.cuda.empty_cache()
           

        return add_scores, del_scores
    
    def editing_process_perpneg(self, latents, noise, jumping_list, B, weights,text_embeddings=None, cfg=1.0, eta=0.0, middle_result = False):
        
        text_embeddings = text_embeddings.to(self.precision_t) #neg(1) + cond(3) 
        neg_text_embeddings = text_embeddings[0:4] 
        unet = self.unet

        cur_noisy_lat  = self.scheduler.add_noise(latents, noise, self.timesteps[jumping_list[0]])
        cur_ind_t = jumping_list[0]
        add_scores = []
        del_scores = []

        for next_ind_t in jumping_list[1:]:
            if   middle_result:
                cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(self.precision_t)
                latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_, cur_noisy_lat_, cur_noisy_lat_])  #neg + 3 cond
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                unet_output = unet(latent_model_input, timestep_model_input, 
                                encoder_hidden_states=text_embeddings).sample
                
                add_scores.append((cur_ind_t, cur_noisy_lat, unet_output))
                neg, cond1, cond2, cond3 = torch.chunk(unet_output, chunks=4)
                unet_output = neg
            else: 
                cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(self.precision_t)
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(cur_noisy_lat_.shape[0], 1).reshape(-1)
                unet_output = unet(cur_noisy_lat_, timestep_model_input, 
                                encoder_hidden_states=neg_text_embeddings).sample
                add_scores.append((cur_ind_t, cur_noisy_lat, unet_output))
                
            delta_t_ =  next_ind_t - cur_ind_t
            cur_noisy_lat = self.sche_func(self.scheduler, unet_output, cur_ind_t, cur_noisy_lat, -delta_t_, eta).prev_sample
            cur_ind_t = next_ind_t

            del unet_output
            torch.cuda.empty_cache()
        #反转    
        inverse_list = list(reversed(jumping_list))

        for next_ind_t in inverse_list[1:]:
    
            cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(self.precision_t)
            latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_, cur_noisy_lat_, cur_noisy_lat_])  #neg + 3 cond
            timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
            unet_output = unet(latent_model_input, timestep_model_input, 
                            encoder_hidden_states=text_embeddings).sample
            
            del_scores.append((cur_ind_t, cur_noisy_lat, unet_output))
            neg, cond1, cond2, cond3 = torch.chunk(unet_output, chunks=4)
            cond = torch.cat([cond1, cond2, cond3])
            delta_noise_preds = cond - neg.repeat(3, 1, 1, 1)
            unet_output = neg + cfg * weighted_perpendicular_aggregator(delta_noise_preds, weights, B)


            #unet_output = uncond + cfg * (cond - uncond) # reverse cfg to enhance the distillation
            # unet_output = neg + cfg * (cond - neg) 
            
            delta_t_ =  next_ind_t-cur_ind_t

            cur_noisy_lat = self.sche_func(self.scheduler, unet_output, cur_ind_t, cur_noisy_lat, -delta_t_, eta).prev_sample
            cur_ind_t = next_ind_t

            del unet_output
            torch.cuda.empty_cache()
        
        cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(self.precision_t)
        latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_, cur_noisy_lat_, cur_noisy_lat_])
        timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
        unet_output = unet(latent_model_input, timestep_model_input, 
                        encoder_hidden_states=text_embeddings).sample
        del_scores.append((cur_ind_t, cur_noisy_lat, unet_output))
        
        del unet_output
        torch.cuda.empty_cache()
           

        return add_scores, del_scores



    @torch.no_grad()
    def get_text_embeds(self, prompt, resolution=(512, 512)):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

   
    def gaussian(self,t, delta, sigma=1.0):
        return np.exp(-(t - delta)**2 / (2 * sigma**2))
    
    def cauculate_w(self,T,n):
        sigma = 1000  # 高斯宽度
        delta_t = T / (n - 1)  # 每个系数的时间平移间隔
        # 计算每个系数的时间平移量
        deltas = [i * delta_t for i in range(n - 1, -1, -1)]  # delta从T到0递减
        # 生成时间序列
        t_values = np.linspace(0, T, T)

        # 计算每个w(t, i)
        w_values = np.zeros((n, len(t_values)))  # 初始化存储所有w(t, i)的数组

        for i in range(n):
            # 对每个系数计算高斯函数
            w_values[i, :] = self.gaussian(t_values, deltas[i], sigma)
            
        w_sums = np.sum(w_values, axis=0)
        w_normalized = w_values / w_sums 
        
        w = w_normalized
        
        return w
    
    def train_step_perpneg(self, text_embeddings, pred_rgb, pred_depth=None, pred_alpha=None,
                           grad_scale=1,use_control_net=False,
                           save_folder:Path=None, iteration=0, warm_up_rate = 0, weights = 0, 
                           resolution=(512, 512), guidance_opt=None, opt=None,as_latent=False):
        
        # flip aug
        pred_rgb, pred_depth, pred_alpha = self.augmentation(pred_rgb, pred_depth, pred_alpha)

        B = pred_rgb.shape[0]
        K = text_embeddings.shape[0] - 1

        if as_latent:      
            latents,_ = self.encode_imgs(pred_depth.repeat(1,3,1,1).to(self.precision_t))
        else:
            latents,_ = self.encode_imgs(pred_rgb.to(self.precision_t))
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        
        weights = weights.reshape(-1)
        noise = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)
        

        #text_embeddings  neg(1) + cond(3)
        #embedding_inverse neg(1) + uncond(1)

       
        text_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-2], text_embeddings.shape[-1]) # make it k+1, c * t, ...
      
        
        
        
        def Random_Sampling():
            jumping_list = []
            jumping_list.append(torch.zeros([1],dtype=torch.long).to(self.device))


            jump_min = guidance_opt.jump_min_begin - int((guidance_opt.jump_min_begin - guidance_opt.jump_min_end)* iteration/opt.iterations)
            jump_max = guidance_opt.jump_max_begin -  int((guidance_opt.jump_max_begin - guidance_opt.jump_max_end)* iteration/opt.iterations)
            max_step = guidance_opt.max_step_begin -int((guidance_opt.max_step_begin - guidance_opt.max_step_end) * iteration/opt.iterations)
            for i in range(guidance_opt.n):
                rand_jump = torch.randint(jump_min, jump_max, (1,))[0].to(self.device)
                if not jumping_list:
                    jumping_list.append(rand_jump)
                elif (jumping_list[-1] + rand_jump) < max_step:
                    jumping_list.append(jumping_list[-1] + rand_jump)
                else:
                    break
            return jumping_list, max_step
        

        with torch.no_grad():
            # step unroll via ddim inversion
            if iteration < opt.iterations+1:
                jumping_list, max_step = Random_Sampling()
                add_scores, del_scores = self.editing_process_perpneg(latents, noise, jumping_list,B, weights,text_embeddings, 7.5, middle_result=iteration % guidance_opt.vis_interval == 0)

 
        grad = 0
        t_len = len(add_scores) 
        for i in range(1,t_len):
            add_t, add_noisy_lat, _ = add_scores[-i] 
            del_t, del_noisy_lat, _ = del_scores[i]
            if add_t != del_t:
                raise ValueError("add_t != del_t") 
            
            if guidance_opt.use_w:
                grad += self.w[t_len-i][iteration]*(add_noisy_lat - del_noisy_lat)
            else:
                grad += (add_noisy_lat - del_noisy_lat)
        
        cur_ind_t, cur_noisy_lat, _ = del_scores[-1]
        if guidance_opt.use_w:
            grad += self.w[0][iteration]*(latents - cur_noisy_lat)
        else:
            grad += (latents - cur_noisy_lat)
        grad = torch.nan_to_num(grad_scale * grad)
        loss = SpecifyGradient.apply(latents, grad)


        if iteration % guidance_opt.vis_interval == 0:
            t_cols, eps_t_cols, prev_latents_noisy_cols = [], [], []
            for cur_ind_t, cur_noisy_lat, unet_output in add_scores[1:]:
                neg, cond1, cond2, cond3 = torch.chunk(unet_output, chunks=4)
                cond = torch.cat([cond1, cond2, cond3])
                delta_noise_preds = cond - neg.repeat(3, 1, 1, 1)
                pred_noise = neg + 7.5 * weighted_perpendicular_aggregator(delta_noise_preds, weights, B)
                t_cols.append(cur_ind_t)
                eps_t_cols.append(pred_noise)
                prev_latents_noisy_cols.append(cur_noisy_lat)
            
            for cur_ind_t, cur_noisy_lat, unet_output in del_scores[:-1]:
                neg, cond1, cond2, cond3 = torch.chunk(unet_output, chunks=4)
                cond = torch.cat([cond1, cond2, cond3])
                delta_noise_preds = cond - neg.repeat(3, 1, 1, 1)
                pred_noise = neg + 7.5 * weighted_perpendicular_aggregator(delta_noise_preds, weights, B)
                t_cols.append(cur_ind_t)
                eps_t_cols.append(pred_noise)
                prev_latents_noisy_cols.append(cur_noisy_lat)
        

        
        if iteration % guidance_opt.vis_interval == 0:
            eta_text = np.zeros((512, 512 * 4, 3))
            eta_text_line = 1
            
            cv2.putText(
                eta_text,
                f"max_step={max_step},t_list=" + ",".join([str(t.item()) for t in jumping_list]),
                (8, 48 * eta_text_line),
                cv2.FONT_HERSHEY_COMPLEX,
                1.5,
                (255, 255, 255),
                2,
            )
            eta_text_line += 1
            # noise_pred_post = noise_pred_uncond + 7.5* delta_DSD    
            lat2rgb = lambda x: torch.clip((x.permute(0,2,3,1) @ self.rgb_latent_factors.to(x.dtype)).permute(0,3,1,2), 0., 1.)
            save_path_iter = os.path.join(save_folder,"iter_{}_step.jpg".format(iteration))
            with torch.no_grad():
                pred_x0_pos_cols = []
                for t_, eps_t_, prev_latents_noisy_ in zip(
                    t_cols, eps_t_cols, prev_latents_noisy_cols
                ):
                    pred_x0_pos_cols.append(
                        self.decode_latents(
                            pred_original(
                                self.scheduler, eps_t_, t_, prev_latents_noisy_
                            ).type(self.precision_t)
                        )
                    )
                grad_abs = torch.abs(grad.detach())
                norm_grad  = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1,keepdim=True), (resolution[0], resolution[1]), mode='bilinear', align_corners=False).repeat(1,3,1,1)

                latents_rgb = F.interpolate(lat2rgb(latents), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)
                #latents_sp_rgb = F.interpolate(lat2rgb(pred_x0_latent_sp), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)

                viz_images = torch.cat([pred_rgb, 
                                        pred_depth.repeat(1, 3, 1, 1), 
                                        pred_alpha.repeat(1, 3, 1, 1), 
                                        rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                                        latents_rgb, norm_grad]
                                        +  pred_x0_pos_cols + 
                                        [
                                            torch.from_numpy(eta_text / 255.0)
                                            .clip(0.0, 1.0)
                                            .to(self.device)
                                            .reshape(
                                                512,
                                                4,
                                                512,
                                                3,
                                            )
                                            .permute(1, 3, 0, 2)
                                        ]
                                        ,dim=0) 
                save_image(viz_images, save_path_iter)

        return loss
    

   
        

       
    def train_step(self, text_embeddings, pred_rgb, pred_depth=None, pred_alpha=None,
                    grad_scale=1,use_control_net=False,
                    save_folder:Path=None, iteration=0, warm_up_rate = 0,
                    resolution=(512, 512), guidance_opt=None, opt=None, as_latent=False):
        if iteration==1 and guidance_opt.use_w:
            self.cauculate_w(opt.iterations+1,guidance_opt.n)

        pred_rgb, pred_depth, pred_alpha = self.augmentation(pred_rgb, pred_depth, pred_alpha)

        B = pred_rgb.shape[0]
        K = text_embeddings.shape[0] - 1

        if as_latent:      
            latents,_ = self.encode_imgs(pred_depth.repeat(1,3,1,1).to(self.precision_t))
        else:
            latents,_ = self.encode_imgs(pred_rgb.to(self.precision_t))
        
        noise = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)
        #text_embeddings  neg + cond
        #inverse_text_embeddings neg + uncond
        text_embeddings = text_embeddings[:, :, ...]
        text_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-2], text_embeddings.shape[-1]) # make it k+1, c * t, ...
        
        
        
        def Random_Sampling():
            jumping_list = []
            jumping_list.append(torch.zeros([1],dtype=torch.long).to(self.device))


            jump_min = guidance_opt.jump_min_begin - int((guidance_opt.jump_min_begin - guidance_opt.jump_min_end)* iteration/opt.iterations)
            jump_max = guidance_opt.jump_max_begin -  int((guidance_opt.jump_max_begin - guidance_opt.jump_max_end)* iteration/opt.iterations)
            max_step = guidance_opt.max_step_begin -int((guidance_opt.max_step_begin - guidance_opt.max_step_end) * iteration/opt.iterations)
            for i in range(guidance_opt.n):
                rand_jump = torch.randint(jump_min, jump_max, (1,))[0].to(self.device)
                if not jumping_list:
                    jumping_list.append(rand_jump)
                elif (jumping_list[-1] + rand_jump) < max_step:
                    jumping_list.append(jumping_list[-1] + rand_jump)
                else:
                    break
            return jumping_list, max_step
        
        

        with torch.no_grad():
            if iteration < opt.iterations + 1:
                jumping_list, max_step = Random_Sampling()
                add_scores, del_scores = self.editing_process(latents, noise, jumping_list, text_embeddings, 7.5)
                
        grad = 0
        t_len = len(add_scores) 
        for i in range(1,t_len):
            add_t, add_noisy_lat, _ = add_scores[-i] 
            del_t, del_noisy_lat, _ = del_scores[i]
            if add_t != del_t:
                raise ValueError("add_t != del_t") 
            if guidance_opt.use_w:
                grad += self.w[t_len-i][iteration]*(add_noisy_lat - del_noisy_lat)
            else:
                grad += (add_noisy_lat - del_noisy_lat)
                
        cur_ind_t, cur_noisy_lat, unet_output = del_scores[-1]
       
        if guidance_opt.use_w:
            grad += self.w[0][iteration]*(latents - cur_noisy_lat)
        else:
            grad += (latents - cur_noisy_lat)
        grad = torch.nan_to_num(grad_scale * grad)
        loss = SpecifyGradient.apply(latents, grad)


        if iteration %  guidance_opt.vis_interval == 0:
            t_cols, eps_t_cols, prev_latents_noisy_cols = [], [], []
            for cur_ind_t, cur_noisy_lat, unet_output in add_scores[1:]:
                neg, cond = torch.chunk(unet_output, chunks=2)
                pred_noise = neg + 7.5 * (cond - neg)
                t_cols.append(cur_ind_t)
                eps_t_cols.append(pred_noise)
                prev_latents_noisy_cols.append(cur_noisy_lat)
            
            for cur_ind_t, cur_noisy_lat, unet_output in del_scores[:-1]:               
                neg, cond= torch.chunk(unet_output, chunks=2)
                pred_noise = neg + 7.5 * (cond - neg)
                t_cols.append(cur_ind_t)
                eps_t_cols.append(pred_noise)
                prev_latents_noisy_cols.append(cur_noisy_lat)
        

        if iteration % guidance_opt.vis_interval == 0:
            eta_text = np.zeros((512, 512 * 4, 3))
            eta_text_line = 1
            
            cv2.putText(
                eta_text,
                f"max_step={max_step},t_list=" + ",".join([str(t.item()) for t in jumping_list]),
                (8, 48 * eta_text_line),
                cv2.FONT_HERSHEY_COMPLEX,
                1.5,
                (255, 255, 255),
                2,
            )
            eta_text_line += 1
            lat2rgb = lambda x: torch.clip((x.permute(0,2,3,1) @ self.rgb_latent_factors.to(x.dtype)).permute(0,3,1,2), 0., 1.)
            save_path_iter = os.path.join(save_folder,"iter_{}_step.jpg".format(iteration))
            with torch.no_grad():
                pred_x0_pos_cols = []
                for t_, eps_t_, prev_latents_noisy_ in zip(
                    t_cols, eps_t_cols, prev_latents_noisy_cols
                ):
                    pred_x0_pos_cols.append(
                        self.decode_latents(
                            pred_original(
                                self.scheduler, eps_t_, t_, prev_latents_noisy_
                            ).type(self.precision_t)
                        )
                    )
                grad_abs = torch.abs(grad.detach())
                norm_grad  = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1,keepdim=True), (resolution[0], resolution[1]), mode='bilinear', align_corners=False).repeat(1,3,1,1)

                latents_rgb = F.interpolate(lat2rgb(latents), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)
                #latents_sp_rgb = F.interpolate(lat2rgb(pred_x0_latent_sp), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)

                viz_images = torch.cat([pred_rgb, 
                                        pred_depth.repeat(1, 3, 1, 1), 
                                        pred_alpha.repeat(1, 3, 1, 1), 
                                        rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                                        latents_rgb, norm_grad]
                                        +  pred_x0_pos_cols + 
                                        [
                                            torch.from_numpy(eta_text / 255.0)
                                            .clip(0.0, 1.0)
                                            .to(self.device)
                                            .reshape(
                                                512,
                                                4,
                                                512,
                                                3,
                                            )
                                            .permute(1, 3, 0, 2)
                                        ]
                                        ,dim=0) 
                save_image(viz_images, save_path_iter)

        return loss
    

    

    def decode_latents(self, latents):
        target_dtype = latents.dtype
        latents = latents / self.vae.config.scaling_factor

        imgs = self.vae.decode(latents.to(self.vae.dtype)).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs.to(target_dtype)

    def encode_imgs(self, imgs):
        target_dtype = imgs.dtype
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs.to(self.vae.dtype)).latent_dist
        kl_divergence = posterior.kl()

        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents.to(target_dtype), kl_divergence