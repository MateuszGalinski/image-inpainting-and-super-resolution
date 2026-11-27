from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from restore_pipeline.representation.train import load_checkpoint


class ClusterLabeler:
    """
    Images -> cluster labels, using a trained representation model and a saved clustering.

    Args:
        checkpoint_path: Model checkpoint made by `save_checkpoint` (e.g. simclr_resnet_full_hard_model.pt).
        clustering_path: Clustering .joblib whose "clusterer" maps embeddings to labels (preprocessing + K-means).
        device: Device the model runs on.
        img_size: Resolution the model was trained on. Images are resized to it, as RESIZE_128 does.
    """

    def __init__(self, checkpoint_path: str | Path, clustering_path: str | Path,
                 device: str = "cuda", img_size: int = 128) -> None:
        self.device = device
        self.img_size = img_size
        self.model = load_checkpoint(checkpoint_path, device)
        self.clusterer = joblib.load(clustering_path)["clusterer"]

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) float images in [0, 1] -> (B,) int64 cluster labels, on the same device as `images`."""
        x = images.to(self.device)
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", antialias=True, align_corners=False)
        embedding = self.model.encode(x).float().cpu().numpy()
        return torch.from_numpy(self.clusterer.predict(embedding)).long().to(images.device)


if __name__ == "__main__":
    # Run from src/:  python -m restore_pipeline.clustering.labeler
    from restore_pipeline.dataprocessing.dataloading import BASIC_TRANSFORM, DataConfig, get_splits

    package = Path(__file__).resolve().parents[1]              # restore_pipeline/
    training, data = package / "training", package / "data"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    labeler = ClusterLabeler(
        training / "checkpoints_simclr_resnet_full_hard" / "simclr_resnet_full_hard_model.pt",
        training / "clusterings" / "simclr_simclr_resnet_full_hard_kmeans12.joblib",
        device=device,
    )

    # A few paintings at their original size: the labeler resizes them itself
    train, _, _ = get_splits(DataConfig(processed_path=data / "wikiart_train_processed", cache_dir=str(data)))
    for i in range(4):
        row = train[i]
        image = BASIC_TRANSFORM(row["image"]).unsqueeze(0)   # (1, 3, H, W) in [0, 1]
        label = labeler(image)[0]
        print(f"{tuple(image.shape[-2:])} px | style {row['style']} | genre {row['genre']} -> cluster {label}")

    # Whole dataset (loader with shuffle=False, so labels[i] belongs to row i):
    # train_labels = torch.cat([labeler(batch["image"]) for batch in train_loader])

    # Inside an inpainting training loop (shuffle=True is fine: labels match the batch they come from):
    # for batch in train_loader:
    #     target = batch["target"].to(device)       # label the clean image: the clustering was fitted on clean paintings
    #     labels = labeler(target)                  # (B,) int64, already on `device`
    #     out = inpainter(batch["image"].to(device), labels)