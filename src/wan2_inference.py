import os

import pytorch_lightning as L
import argparse
from omegaconf import OmegaConf
import numpy as np
from PIL import Image

from datasets.custom_dataset import CustomTestDataset
from torch.utils.data import DataLoader
from models.wan2.custom_pipeline import CustomWanPipeline
from src.wan2_trainer import WorldWanderTrainSystem

import torch
from diffusers.utils import export_to_video
from diffusers.utils import numpy_to_pil
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler


class WorldWanderInferenceSystem(WorldWanderTrainSystem):
    # custom load ckpt
    def load_state_dict(self, state_dict, strict: bool = True):
        # only load the lora
        self.transformer.load_state_dict(state_dict['lora'], strict=False)

    def on_predict_epoch_start(self):
        self.pred_pipeline = CustomWanPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            scheduler=UniPCMultistepScheduler.from_config(
                FlowMatchEulerDiscreteScheduler.from_pretrained(self.hparams.model_id, subfolder="scheduler").config,
                flow_shift=5,
            ),
        )
        self.pred_path = self.hparams.pred_path
        os.makedirs(self.pred_path, exist_ok=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # data process
        input_pixel_values = batch["input_pixel_values"] # B, C, F, H, W
        ref_pixel_values = batch['ref_pixel_values'].unsqueeze(2) if self.hparams.dataset.is_one2three else None # # B, C, 1, H, W
        prompts = batch["prompts"]
        path = batch["path"]
        # save input
        meta_video = input_pixel_values.squeeze(0).permute(1, 2, 3, 0)
        meta_video = ((meta_video + 1) * 0.5).clamp(0, 1).cpu().numpy()

        if self.hparams.dataset.is_one2three:
            meta_ref = ref_pixel_values.squeeze(0).repeat(1, self.hparams.dataset.sample_n_frames, 1, 1).permute(1, 2, 3, 0)
            meta_ref = ((meta_ref + 1) * 0.5).clamp(0, 1).cpu().numpy()
        # ---------------------------------------------------------------------------------
        input_pixel_values = self.vae.encode(input_pixel_values).latent_dist.sample() # [B, C, F, H, W]
        input_pixel_values = (input_pixel_values - self.latents_mean) / self.latents_std # scaling
        # 
        if self.hparams.dataset.is_one2three:
            ref_pixel_values = self.vae.encode(ref_pixel_values).latent_dist.sample()
            ref_pixel_values = (ref_pixel_values - self.latents_mean) / self.latents_std
        else:
            ref_pixel_values = None
        attention_kwargs = {
            'encoder_condition_states': input_pixel_values,
            'encoder_ref_states': ref_pixel_values,
            'use_collaborative_position_encoding': self.hparams.use_collaborative_position_encoding,
        }
        #
        video_generate = self.pred_pipeline(
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
            concatenated_video = np.concatenate([meta_ref, meta_video, video_generate], axis=1)
        else:
            concatenated_video = np.concatenate([meta_video, video_generate], axis=1)
        pred_video_path = os.path.join(self.pred_path, f"{path[0]}.mp4")
        export_to_video(concatenated_video, output_video_path=pred_video_path, fps=self.hparams.dataset.fps)

        return

def main(opt):
    L.seed_everything(opt.seed)
    test_dataset = CustomTestDataset(
        original_video_root=opt.original_video_root,
        ref_image_root=opt.ref_image_root,
        #
        height=opt.dataset.height,
        width=opt.dataset.width,
        sample_n_frames=opt.dataset.sample_n_frames,
        stride=1,
        is_one2three=opt.dataset.is_one2three,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=opt.dataset.num_workers,
        drop_last=opt.dataset.drop_last,
        pin_memory=opt.dataset.pin_memory,
        shuffle=False,
    )
    system = WorldWanderInferenceSystem.load_from_checkpoint(opt.ckpt_path, opt=opt)
    # 
    trainer = L.Trainer(
        logger=False,
        precision=opt.training.precision,
        log_every_n_steps=1,
        accelerator=opt.training.accelerator,
        strategy=opt.training.strategy,
        benchmark=opt.training.benchmark,
        num_nodes=opt.num_nodes,
    )
    trainer.predict(system, dataloaders=test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wan2-2_lora_three2one.yaml", help="path to the yaml config file")
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--json_path", type=str, help="json index file for test")
    parser.add_argument("--video_root", type=str, help="dataset root")
    # parser.add_argument("--original_video_root", type=str, help="original video root")
    # parser.add_argument("--ref_image_root", type=str, help="reference image root")
    parser.add_argument("--pred_path", type=str, default="", help="save path for inference")
    parser.add_argument("--seed", type=int, default=42)
    # ----------------------------------------------------------------------
    args, extras = parser.parse_known_args()
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
