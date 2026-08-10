import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
import wandb
import numpy as np
import torchvision

import argparse
from omegaconf import OmegaConf
from tools.util import CustomProgressBar, CustomModelCheckpoint
from tools.util import masks_like, resolve_strategy

from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from models.wan2.custom_pipeline import CustomWanPipeline
from models.wan2.transformer_wan import CustomWanTransformer3DModel

from datasets.custom_dataset import CustomTrainDataset
from tools.my_schedule import FlowMatchScheduler
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler

from transformers import AutoTokenizer, UMT5EncoderModel
import torch
import random
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WorldWanderTrainSystem(L.LightningModule):
    def __init__(self, opt):
        super().__init__()
        # save save_hyperparameters，access by self.hparams
        self.save_hyperparameters(opt)   
        self.is_configured = False

    def configure_model(self):
        if not self.is_configured:
            self.is_configured = True
            #
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_id, subfolder="tokenizer")
            self.text_encoder = UMT5EncoderModel.from_pretrained(
                self.hparams.model_id,
                subfolder="text_encoder",
                torch_dtype=torch.float32
            )
            self.vae = AutoencoderKLWan.from_pretrained(
                self.hparams.model_id,
                subfolder="vae",
                torch_dtype=torch.float32
            )
            self.train_scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
            self.train_scheduler.set_timesteps(1000, training=True) # reset training scheduler
            self.transformer = CustomWanTransformer3DModel.from_pretrained(self.hparams.model_id, subfolder="transformer", torch_dtype=torch.float32)
            # unable the grad
            self.text_encoder.requires_grad_(False)
            self.vae.requires_grad_(False)
            if self.hparams.use_lora:
                self.transformer.requires_grad_(False)
            # enable the gradient_checkpointing
            if self.hparams.training.gradient_checkpointing:
                self.transformer.gradient_checkpointing = True
                self.transformer.enable_gradient_checkpointing()
            # register buffer
            self.register_buffer('latents_mean', torch.tensor(self.vae.config.latents_mean).float().view(1, self.vae.config.z_dim, 1, 1, 1))
            self.register_buffer('latents_std', torch.tensor(self.vae.config.latents_std).float().view(1, self.vae.config.z_dim, 1, 1, 1))

            # now we will add LoRA weights to the specific layers
            if self.hparams.use_lora:
                from peft import LoraConfig
                transformer_lora_config = LoraConfig(
                    r=self.hparams.training.rank,
                    lora_alpha=self.hparams.training.rank,
                    init_lora_weights=True,
                    target_modules=["to_k", "to_q", "to_v", "to_out.0", "ffn.net.0.proj", "ffn.net.2"],
                )
                self.transformer.add_adapter(transformer_lora_config) # freeze all and enbale LoRA

    def encode_prompt(self, prompt):
        max_sequence_length = 512
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()
        text_embeds = self.text_encoder(ids.to(self.device), mask.to(self.device)).last_hidden_state
        text_embeds = [u[:v] for u, v in zip(text_embeds, seq_lens)]
        text_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in text_embeds], dim=0
        )
        return text_embeds

    def process_data(self, batch, batch_idx):
        first_pixel_values = batch["first_pixel_values"] # B, C, F, H, W
        third_pixel_values = batch["third_pixel_values"] # B, C, F, H, W
        ref_pixel_values = batch['ref_pixel_values'].unsqueeze(2) if self.hparams.dataset.is_one2three else None # # B, C, 1, H, W
        prompts = batch["prompts"]
        drop_ratio = 0.1
        if self.hparams.use_drop_text:
            prompts = [prompt if random.random() > drop_ratio else '' for prompt in prompts]
        # ---------------------------------------------------------------------------------------
        # first-persion video
        first_pixel_values = self.vae.encode(first_pixel_values).latent_dist.sample() # [B, C, F, H, W]
        first_pixel_values = (first_pixel_values - self.latents_mean) / self.latents_std # scaling
        # third-persion video
        third_pixel_values = self.vae.encode(third_pixel_values).latent_dist.sample() # [B, C, F, H, W]
        third_pixel_values = (third_pixel_values - self.latents_mean) / self.latents_std # scaling
        # ref frame
        if self.hparams.dataset.is_one2three:
            ref_pixel_values = self.vae.encode(ref_pixel_values).latent_dist.sample() # [B, C, 1, H, W]
            ref_pixel_values = (ref_pixel_values - self.latents_mean) / self.latents_std # scaling
        # encode prompts
        prompt_embeds = self.encode_prompt(prompts)

        return first_pixel_values, third_pixel_values, ref_pixel_values, prompt_embeds

    # training for-loop
    def training_step(self, batch, batch_idx):
        first_pixel_values, third_pixel_values, ref_pixel_values, prompt_embeds = self.process_data(batch, batch_idx)        
        # ---------------------------------------------------------------------------------------
        batch_size, num_channels, num_frames, height, width = third_pixel_values.shape
        noise = torch.randn_like(third_pixel_values)
        timestep_id = torch.randint(0, self.train_scheduler.num_train_timesteps, (batch_size,))
        timestep = self.train_scheduler.timesteps[timestep_id].to(dtype=third_pixel_values.dtype)
        #
        if self.hparams.dataset.is_one2three:
            latent_noisy = self.train_scheduler.add_noise(third_pixel_values, noise, timestep)
            mask1, mask2 = masks_like(noise, zero=False)
            v_target = self.train_scheduler.training_target(third_pixel_values, noise, timestep)
            # condition
            attention_kwargs = {
                'encoder_condition_states': first_pixel_values,
                'encoder_ref_states': ref_pixel_values,
                'use_collaborative_position_encoding': self.hparams.use_collaborative_position_encoding,
            }
            timestep_all = (timestep.view(batch_size, 1, 1, 1) * mask2[0][:, 0, :, ::2, ::2]).flatten(1)
            timestep_all = torch.concat([torch.zeros_like(mask2[0][:, 0, 0, ::2, ::2].flatten(1)), torch.zeros_like(timestep_all), timestep_all], dim=-1)
        else:
            latent_noisy = self.train_scheduler.add_noise(first_pixel_values, noise, timestep)
            mask1, mask2 = masks_like(noise, zero=False)
            #
            v_target = self.train_scheduler.training_target(first_pixel_values, noise, timestep)
            # condition
            attention_kwargs = {
                'encoder_condition_states': third_pixel_values,
                'use_collaborative_position_encoding': self.hparams.use_collaborative_position_encoding,
            }
            timestep_all = (timestep.view(batch_size, 1, 1, 1) * mask2[0][:, 0, :, ::2, ::2]).flatten(1)
            timestep_all = torch.concat([torch.zeros_like(timestep_all), timestep_all], dim=-1)
        #
        v_pred = self.transformer(
            hidden_states=latent_noisy, # B, C, F, H, W
            encoder_hidden_states=prompt_embeds,
            timestep=timestep_all,
            return_dict=False,
            attention_kwargs=attention_kwargs,
        )[0]
        # cal loss
        loss = torch.nn.functional.mse_loss(v_pred.float(), v_target.float(), reduction='none')
        weight = self.train_scheduler.training_weight(timestep).to(loss.device)
        loss = (loss * weight[:, None, None, None, None]).mean()
        # record information
        self.log("train/loss", loss, prog_bar=True, on_step=True,
                logger=True, sync_dist=True if self.trainer.world_size > 1 else False)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True,
                on_step=True, logger=True, sync_dist=True if self.trainer.world_size > 1 else False)

        return loss

    def on_validation_epoch_start(self):
        self.val_pipeline = CustomWanPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            scheduler=UniPCMultistepScheduler.from_config(
                FlowMatchEulerDiscreteScheduler.from_pretrained(self.hparams.model_id, subfolder="scheduler").config,
                flow_shift=5,
            ),
        )
        self.print(f"validation at step: {self.global_step}.")
        self.val_path = os.path.join(self.hparams.output_root, self.hparams.experiment_name, 'val_samples')
        os.makedirs(self.val_path, exist_ok=True)

    def validation_step(self, batch, batch_idx):
        # data process
        first_pixel_values = batch["first_pixel_values"] # B, C, F, H, W
        third_pixel_values = batch["third_pixel_values"] # B, C, F, H, W
        ref_pixel_values = batch['ref_pixel_values'].unsqueeze(2) if self.hparams.dataset.is_one2three else None # # B, C, 1, H, W
        prompts = batch["prompts"]
        # ---------------------------------------------------------------------------------
        if self.hparams.dataset.is_one2three:
            video_gt = third_pixel_values.squeeze(0).permute(1, 0, 2, 3)
            video_gt = ((video_gt + 1) * 0.5).clamp(0, 1)
            video_gt = video_gt.permute(0, 2, 3, 1).cpu().numpy()
            #
            meta_ref = ref_pixel_values.squeeze(0).repeat(1, self.hparams.dataset.sample_n_frames, 1, 1).permute(1, 2, 3, 0)
            meta_ref = ((meta_ref + 1) * 0.5).clamp(0, 1).cpu().numpy()
            ref_pixel_values = self.vae.encode(ref_pixel_values).latent_dist.sample()
            ref_pixel_values = (ref_pixel_values - self.latents_mean) / self.latents_std
            #
            meta_video = first_pixel_values.squeeze(0).permute(1, 2, 3, 0)
            meta_video = ((meta_video + 1) * 0.5).clamp(0, 1).cpu().numpy()
            first_pixel_values = self.vae.encode(first_pixel_values).latent_dist.sample() # [B, C, F, H, W]
            first_pixel_values = (first_pixel_values - self.latents_mean) / self.latents_std # scaling
            #
            attention_kwargs = {
                'encoder_condition_states': first_pixel_values,
                'encoder_ref_states': ref_pixel_values,
                'use_collaborative_position_encoding': self.hparams.use_collaborative_position_encoding,
            }
        else:
            video_gt = first_pixel_values.squeeze(0).permute(1, 0, 2, 3)
            video_gt = ((video_gt + 1) * 0.5).clamp(0, 1)
            video_gt = video_gt.permute(0, 2, 3, 1).cpu().numpy()
            #
            meta_video = third_pixel_values.squeeze(0).permute(1, 2, 3, 0)
            meta_video = ((meta_video + 1) * 0.5).clamp(0, 1).cpu().numpy()
            third_pixel_values = self.vae.encode(third_pixel_values).latent_dist.sample() # [B, C, F, H, W]
            third_pixel_values = (third_pixel_values - self.latents_mean) / self.latents_std # scaling
            #
            attention_kwargs = {
                'encoder_condition_states': third_pixel_values,
                'use_collaborative_position_encoding': self.hparams.use_collaborative_position_encoding,
            }
        #
        video_generate = self.val_pipeline(
            prompt=prompts,
            height=self.hparams.dataset.height,
            width=self.hparams.dataset.width,
            num_frames=self.hparams.dataset.sample_n_frames,
            guidance_scale=5.0,
            attention_kwargs=attention_kwargs,
        )
        video_generate = video_generate.frames[0]   
        #
        if self.hparams.dataset.is_one2three:
            concatenated_video = np.concatenate([meta_ref, meta_video, video_generate, video_gt], axis=1)
        else:
            concatenated_video = np.concatenate([meta_video, video_generate, video_gt], axis=1)
        val_video_path = os.path.join(self.val_path, f"val_{self.global_step}step-batch_{batch_idx}-rank{self.trainer.global_rank}.mp4")
        export_to_video(concatenated_video, output_video_path=val_video_path, fps=self.hparams.dataset.fps)
        # upload to logger
        if self.trainer.is_global_zero and isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({
                f"val/video_{self.global_step}step-batch_{batch_idx}": wandb.Video(
                    val_video_path,
                    caption=f"Validation video - step {self.global_step}, batch {batch_idx}",
                    format="mp4"
                )
            })

    def configure_optimizers(self):
        # configure paras
        params_and_lrs = []
        modules = [self.transformer]
        for module in modules:
            params = [p for p in module.parameters() if p.requires_grad]
            learning_rate = self.hparams.training.learning_rate * (self.hparams.training.accumulate_grad_batches * self.hparams.num_gpus * self.hparams.num_nodes) ** 0.5
            params_and_lrs.append(
                {
                    "params": params, 
                    "lr": learning_rate
                }
            )
        # configrue optimizer
        optimizer = torch.optim.AdamW(
            params_and_lrs,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=self.hparams.training.weight_decay,  # 默认 0.01
        )
        # configrue scheduler
        def lr_fn(step, warmup_steps):
            if warmup_steps <= 0:
                return 1
            else:
                return min(step / warmup_steps, 1)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: lr_fn(step, warmup_steps=self.hparams.training.warmup_steps),
        )
        # return
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            }
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        # only load lora
        self.transformer.load_state_dict(state_dict['lora'], strict=False)

    def on_save_checkpoint(self, checkpoint):
        del checkpoint['hparams_name']
        del checkpoint['hparams_type']
        # reset model_state_dict
        model_state_dict = {}
        # transformer_processor
        tmp_dict = {}
        for name, param in self.transformer.state_dict().items():
            if "lora" in name:
                tmp_dict[name] = param.cpu()
        model_state_dict["lora"] = tmp_dict 
        # reset the model_state_dict
        checkpoint['state_dict'] = model_state_dict


def main(opt):
    # set seed
    L.seed_everything(opt.seed)
    # dataset && dataloader
    # train_dataset = CustomTrainDataset(
    #     first_video_root=opt.dataset.first_video_root,
    #     third_video_root=opt.dataset.third_video_root,
    #     ref_image_root=opt.dataset.ref_image_root,
    #     #
    #     height=opt.dataset.height,
    #     width=opt.dataset.width,
    #     resize_long=opt.dataset.resize_long,
    #     sample_n_frames=opt.dataset.sample_n_frames,
    #     stride=opt.dataset.stride,
    #     is_one2three=opt.dataset.is_one2three,
    #     training_len=opt.num_nodes * opt.num_gpus * opt.training.accumulate_grad_batches * opt.training.max_steps * opt.training.batch_size # 自动计算样本数
    # )
    train_dataset = CustomTrainDataset(
        # 新增参数: 支持 JSON 索引
        json_index_path=opt.dataset.get('json_index_path', None),
        video_root_prefix=opt.dataset.get('video_root_prefix', ''),
        followbench_root=opt.dataset.get('followbench_root', None),
        warped_video_root=opt.dataset.get('warped_video_root', None),
        
        # 兼容原有参数 (如果 yaml 里写了 root 也不影响，Dataset 内部会优先 JSON)
        first_video_root=opt.dataset.get('first_video_root', None),
        third_video_root=opt.dataset.get('third_video_root', None),
        ref_image_root=opt.dataset.get('ref_image_root', None),
        
        # 其他参数
        height=opt.dataset.height,
        width=opt.dataset.width,
        resize_long=opt.dataset.resize_long,
        sample_n_frames=opt.dataset.sample_n_frames,
        stride=opt.dataset.stride,
        is_one2three=opt.dataset.is_one2three
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opt.training.batch_size,
        num_workers=opt.dataset.num_workers,
        drop_last=opt.dataset.drop_last,
        pin_memory=opt.dataset.pin_memory,
        shuffle=opt.dataset.shuffle,
    )
    #
    val_dataset = CustomTrainDataset(
        # 新增参数: 支持 JSON 索引
        json_index_path=opt.dataset.get('json_index_path', None),
        video_root_prefix=opt.dataset.get('video_root_prefix', ''),
        followbench_root=opt.dataset.get('followbench_root', None),
        warped_video_root=opt.dataset.get('warped_video_root', None),
        
        first_video_root=opt.dataset.first_video_root,
        third_video_root=opt.dataset.third_video_root,
        ref_image_root=opt.dataset.ref_image_root,
        # 
        height=opt.dataset.height,
        width=opt.dataset.width,
        resize_long=opt.dataset.resize_long,
        sample_n_frames=opt.dataset.sample_n_frames,
        stride=opt.dataset.stride,
        is_one2three=opt.dataset.is_one2three,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=opt.dataset.num_workers,
        drop_last=opt.dataset.drop_last,
        pin_memory=opt.dataset.pin_memory,
        shuffle=False,
    )
    # custom system
    system = WorldWanderTrainSystem(opt)
    # custom logger
    wandb_logger = WandbLogger(
        project=opt.experiment_project,
        name=opt.experiment_name,
        save_dir=os.path.join(opt.output_root, opt.experiment_name),
        log_model=False,
        offline=True,
    )
    # define trainer
    trainer = L.Trainer(
        logger=wandb_logger,
        # logger=False,
        max_steps=opt.training.max_steps,
        precision=opt.training.precision,
        num_sanity_val_steps=1,
        limit_val_batches=1,
        val_check_interval=opt.training.save_val_interval_steps * opt.training.accumulate_grad_batches,
        accumulate_grad_batches=opt.training.accumulate_grad_batches,
        gradient_clip_val=opt.training.gradient_clip_val,
        gradient_clip_algorithm='value',
        log_every_n_steps=1,
        accelerator=opt.training.accelerator,
        strategy=resolve_strategy(opt.training.strategy), # specially for fsdp
        benchmark=opt.training.benchmark,
        callbacks=[
            CustomProgressBar(),
            CustomModelCheckpoint(
                dirpath=os.path.join(opt.output_root, opt.experiment_name, 'checkpoints'),
                filename="{step}",
                every_n_train_steps=opt.training.save_val_interval_steps,
                save_top_k=-1,
                save_weights_only=True if opt.training.strategy == 'fsdp' else False,
                verbose=False,
            )
        ],
        num_nodes=opt.num_nodes,
    )
    #
    # trainer.fit(system,
    #     train_dataloaders=train_dataloader,
    #     val_dataloaders=val_dataloader,
    #     # ckpt_path='resume_path'
    # )
    ckpt_path = opt.resume_path if opt.resume_path and os.path.exists(opt.resume_path) else None
    if ckpt_path:
        print(f"Resuming from checkpoint: {ckpt_path}")
    trainer.fit(system,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path  # 传入处理后的路径
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wan2-2_lora_three2one.yaml", help="path to the yaml config file")
    # additional
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_path", type=str, default="", help="path to a .ckpt to resume from")
    # ----------------------------------------------------------------------
    args, extras = parser.parse_known_args() # split paras
    args = vars(args)
    opt = OmegaConf.merge(
        OmegaConf.load(args['config']),
        OmegaConf.from_cli(extras),
        OmegaConf.create(args),
        OmegaConf.create({"num_nodes": int(os.environ.get("NUM_NODES", 1))}),
        OmegaConf.create({"num_gpus": int(torch.cuda.device_count())}),
    )
    # ----------------------------------------------------------------------
    main(opt)
