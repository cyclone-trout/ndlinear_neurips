import argparse
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time

import numpy as np
import torch
from PIL import Image
from accelerate import Accelerator
from diffusers.models import AutoencoderKL
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import Dataset, DataLoader
from download import find_model

from diffusion import create_diffusion
from models import DiT_models, NdTimestepEmbedder, TimestepEmbedder

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device

    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8

    if hasattr(args, 'ckpt') and args.ckpt and any(
            substr in args.ckpt for substr in ["DiT-XL-2-256x256.pt", "DiT-XL-2-512x512.pt"]):
        state_dict = find_model(args.ckpt)
        try:
            args.model = "DiT-XL/2"
            model = DiT_models[args.model](
                input_size=latent_size,
                num_classes=args.num_classes,
                use_ndtse=False,
                use_ndmlp=False,
                use_variant=args.mlp_variant,
                use_num_transforms=args.use_num_transforms,
                tse_scale_factor=args.tse_scale_factor)

            model.load_state_dict(state_dict)
            for param in model.parameters():
                param.requires_grad = False

            if args.use_ndtse:
                new_t_embedder = NdTimestepEmbedder(hidden_size=model.hidden_size,
                                                    frequency_embedding_size=256,
                                                    use_num_transforms=args.use_num_transforms,
                                                    tse_scale_factor=1,
                                                    knowledge_transfer=args.knowledge_transfer,
                                                    src_layers=model.t_embedder)
            else:
                new_t_embedder = TimestepEmbedder(hidden_size=model.hidden_size)
            model.t_embedder = new_t_embedder
            for param in model.t_embedder.parameters():
                param.requires_grad = True

            if args.knowledge_transfer:
                original_filename = os.path.basename(args.ckpt)
                torch.save(model.state_dict(), f'ndkt_svd_{original_filename}')
                total_params = sum(p.numel() for p in model.parameters())
                print(f"Total parameters: {total_params}")
                print(model)
                exit(0)
        except RuntimeError as e:
            print(e)
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('_orig_mod.', '')
                new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict)
    else:
        model = DiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
            use_ndtse=args.use_ndtse,
            use_ndmlp=args.use_ndmlp,
            use_variant=args.mlp_variant,
            use_num_transforms=args.use_num_transforms,
            tse_scale_factor=args.tse_scale_factor
        )

    model = model.to(device)

    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    features_dir = f"{args.feature_path}/imagenet256_features"
    labels_dir = f"{args.feature_path}/imagenet256_labels"
    dataset = CustomDataset(features_dir, labels_dir)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.feature_path})")

    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()
    model, opt, loader = accelerator.prepare(model, opt, loader)

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for step, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            x = x.squeeze(dim=1)
            y = y.squeeze(dim=1)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)

            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean() / args.accumulation_steps

            accelerator.backward(loss)

            if (step + 1) % args.accumulation_steps == 0:
                opt.step()
                opt.zero_grad()
                update_ema(ema, model)

            running_loss += loss.item() * args.accumulation_steps
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes
                if accelerator.is_main_process:
                    logger.info(
                        f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                    )

                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    try:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "args": args
                        }
                    except AttributeError as ae:
                        print(f"AttributeError when attempting to save model checkpoints: {str(ae)}")
                        checkpoint = {
                            "model": accelerator.unwrap_model(model).state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "args": args
                        }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
    model.eval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XS/8")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--use-ndmlp", action='store_true')
    parser.add_argument("--use-ndtse", action='store_true')
    parser.add_argument("--mlp-variant", type=int, default=4)
    parser.add_argument("--tse-scale-factor", type=float, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--knowledge-transfer', action='store_true')
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--use-num-transforms", type=int, choices=[2, 3, 4, 21], default=2)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    main(args)