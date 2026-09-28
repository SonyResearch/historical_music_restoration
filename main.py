import argparse
import os
import random

import torch
import torch.distributed as dist
import yaml

from restor.experiment import Experiment
from restor.trainer import Trainer


def _parse_overrides(overrides, cfg):
    for item in overrides:
        key, val = item.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = yaml.safe_load(val)


def train(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    _parse_overrides(args.override, cfg)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=torch.device("cuda", local_rank),
        )
        rank = dist.get_rank()
        affinities = os.environ.get(
            "QUEUE_CPU_AFFINITIES",
            "18-23,66-71;6-11,54-59;42-47,90-95;30-35,78-83",
        ).split(";")
        if local_rank < len(affinities) and hasattr(os, "sched_setaffinity"):
            cpus = set()
            for part in affinities[local_rank].split(","):
                bounds = [int(value) for value in part.split("-", 1)]
                cpus.update(range(bounds[0], bounds[-1] + 1))
            os.sched_setaffinity(0, cpus)
    else:
        rank = 0

    seed = int(cfg["training"]["seed"]) + rank
    torch.manual_seed(seed)
    random.seed(seed)

    try:
        if rank == 0:
            exp = Experiment(
                name=args.name, cfg=cfg,
                root=args.exp_root, resume=args.resume,
            )
        if distributed:
            dist.barrier()
        if rank != 0:
            exp = Experiment(
                name=args.name, cfg=cfg,
                root=args.exp_root, resume=True,
            )
        trainer = Trainer(exp)
        trainer.train()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def infer(args):
    from restor.inference import Inferencer

    inferencer = Inferencer(
        args.checkpoint, config_path=args.config, device=args.device,
    )
    if args.cfm_steps is not None:
        if args.cfm_steps < 1:
            raise ValueError("--cfm-steps must be at least 1")
        denoiser_type = inferencer.cfg["denoiser"]["type"]
        if not denoiser_type.startswith("cfm"):
            raise ValueError("--cfm-steps is valid only for a CFM checkpoint")
        inferencer.cfg["denoiser"][denoiser_type][
            "inference_steps_default"
        ] = args.cfm_steps

    # Seed after model construction/loading so initialization cannot consume
    # values intended for the reproducible CFM Gaussian source latent.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    inferencer.denoise_file(
        args.input, args.output,
        overlap=args.overlap, chunk_sec=args.chunk_sec,
    )
    print(f"Saved denoised audio to {args.output}")


def ls(args):
    Experiment.list_experiments(root=args.exp_root)


def precompute(args):
    """Build the normalized latent-pair cache consumed by Trainer."""
    from restor.data_precompute import data_precompute

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"]["root"] = args.source_dir
    cfg["dataset"]["segment_duration"] = args.segment_duration
    cfg["training"]["batch_size"] = args.batch_size
    cfg["training"]["seed"] = args.seed
    pcfg = cfg.setdefault("precompute", {})
    pcfg.update({
        "multi_song_dir": args.source_dir,
        "multi_song_output_name": os.path.basename(os.path.abspath(args.source_dir)),
        "output_root": args.output_root,
        "degradations_per_clean": args.degradations_per_clean,
        "hop_fraction": args.hop_fraction,
        "val_ratio": args.val_ratio,
        "target_lufs": args.target_lufs,
        "source_loudness_scope": "whole_song_pre_normalized",
        "latent_stats_scope": "train_windows",
        "simple_output_layout": True,
    })
    if args.manifest:
        pcfg["multi_song_manifest"] = args.manifest
        pcfg["related_view_split"] = "aligned_recording_views"
    else:
        pcfg["related_view_split"] = "independent_sources"
    if args.noise_dir:
        cfg["corruption"]["gramophone_noise"]["noise_dir"] = args.noise_dir

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    data_precompute(cfg).build_data_mult_song()


def main():
    parser = argparse.ArgumentParser(description="restor – latent-space audio denoiser")
    parser.add_argument("--exp-root", default="experiments",
                        help="Root directory for all experiments")
    sub = parser.add_subparsers(dest="command")

    # -- train --
    p_train = sub.add_parser("train", help="Train a denoiser")
    p_train.add_argument("--name", required=True, help="Experiment name")
    p_train.add_argument("--config", default="config/samecfm40_fms.yaml")
    p_train.add_argument("--resume", action="store_true",
                         help="Resume from latest checkpoint (uses saved config)")
    p_train.add_argument("--override", nargs="*", default=[],
                         help="dot.key=value config overrides (ignored on --resume)")

    # -- infer --
    p_infer = sub.add_parser("infer", help="Denoise an audio file")
    p_infer.add_argument("--checkpoint", required=True)
    p_infer.add_argument("--input", required=True, help="Input audio file")
    p_infer.add_argument("--output", required=True, help="Output audio file")
    p_infer.add_argument("--config", default=None,
                         help="Config (only if not embedded in checkpoint)")
    p_infer.add_argument("--device", default=None)
    p_infer.add_argument("--overlap", type=float, default=0.5)
    p_infer.add_argument("--chunk-sec", type=float, default=30.0)
    p_infer.add_argument(
        "--cfm-steps", type=int, default=None,
        help="Override CFM Euler steps (paper evaluation used 10)",
    )
    p_infer.add_argument(
        "--seed", type=int, default=42,
        help="Seed for the CFM Gaussian source latent (default: 42)",
    )

    # -- precompute --
    p_precompute = sub.add_parser(
        "precompute",
        help="Create leakage-safe five-stage SAME-L latent training pairs",
    )
    p_precompute.add_argument("--source-dir", required=True,
                              help="Flat folder of clean, pre-normalized WAV files")
    p_precompute.add_argument("--output-root", default="data/fos_precomputed")
    p_precompute.add_argument("--manifest", default=None,
                              help="TSV grouping aligned full-mix/section views")
    p_precompute.add_argument("--noise-dir", default=None,
                              help="Gramophone Record Noise Dataset WAV folder")
    p_precompute.add_argument("--config", default="config/samecfm40_fos.yaml")
    p_precompute.add_argument("--batch-size", type=int, default=4)
    p_precompute.add_argument("--degradations-per-clean", type=int, default=20)
    p_precompute.add_argument("--segment-duration", type=float, default=5.0)
    p_precompute.add_argument("--hop-fraction", type=float, default=0.1)
    p_precompute.add_argument("--val-ratio", type=float, default=0.1)
    p_precompute.add_argument("--target-lufs", type=float, default=-23.0)
    p_precompute.add_argument("--seed", type=int, default=42)

    # -- ls --
    sub.add_parser("ls", help="List all experiments")

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)
    elif args.command == "precompute":
        precompute(args)
    elif args.command == "ls":
        ls(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
