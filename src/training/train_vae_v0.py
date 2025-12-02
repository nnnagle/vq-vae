# src/training/train_vae_v0.py

from src.training.trainer import Trainer, TrainConfig

def main():
    cfg = TrainConfig(
        zarr_path="/data/VA/zarr/va_cube.zarr",
        patch_size=256,
        num_epochs=5,
        batch_size=2,
        debug_window=True,  # flip to False for full domain
    )
    trainer = Trainer(cfg)
    trainer.run()

if __name__ == "__main__":
    main()
