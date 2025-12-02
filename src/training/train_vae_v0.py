# src/training/train_vae_v0.py

from src.training.trainer import Trainer, TrainConfig

def main():
    cfg = TrainConfig.from_yaml("configs/vae_v0.yaml")
    trainer = Trainer(cfg)
    trainer.run()

if __name__ == "__main__":
    main()
