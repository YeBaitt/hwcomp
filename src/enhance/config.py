"""配置加载：从 config.yaml 读取并包装为 dataclass。"""
from dataclasses import dataclass
from pathlib import Path

import yaml

@dataclass
class Config:
    """运行时配置：把 config.yaml 的字段扁平化为单个 dataclass 对象，后续模块统一从这里取值。"""
    data_root: Path
    image_pairs_train_dir: Path
    val_dir: Path
    test_dir: Path
    out_root: Path
    ckpt_root: Path
    submit_dir: Path
    zip_path: Path
    patch_size: int
    batch_size: int
    num_workers: int
    seed: int
    kind_2k_weight: float
    stage1_epochs: int
    tile_size: int
    overlap: int
    steps: int
    scheduler: str
    batch_tiles: int
    use_tta: bool
    stage1_pretrained: str
    lam: float
    n: float
    w: float
    alpha: float
    beta: float
    sigma: float
    warmup_epochs: int = 3
    early_stop_patience: int = 8
    val_holdout_n: int = 10
    cache_root: Path = Path("dataset/ImagePairs/cache")
    cache_pairs: int = 128
    length_factor: int = 1
    stage1_lr: float = 1e-4
    grad_accum: int = 1

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """从 yaml 文件加载配置并构造 Config 对象。"""
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        out = d["output"]
        return cls(
            data_root=Path(d["data"]["root"]),
            image_pairs_train_dir=Path(d["data"]["image_pairs_train_dir"]),
            val_dir=Path(d["data"]["val_dir"]),
            test_dir=Path(d["data"]["test_dir"]),
            out_root=Path(out["out_root"]),
            ckpt_root=Path(out["ckpt_root"]),
            submit_dir=Path(out["submit_dir"]),
            zip_path=Path(out["zip_path"]),
            patch_size=d["training"]["patch_size"],
            batch_size=d["training"]["batch_size"],
            grad_accum=int(d["training"].get("grad_accum", 1)),
            num_workers=d["training"]["num_workers"],
            seed=d["training"]["seed"],
            kind_2k_weight=d["training"]["kind_2k_weight"],
            stage1_epochs=d["training"]["stage1_epochs"],
            tile_size=d["inference"]["tile_size"],
            overlap=d["inference"]["overlap"],
            steps=d["inference"]["steps"],
            scheduler=d["inference"]["scheduler"],
            batch_tiles=d["inference"]["batch_tiles"],
            use_tta=d["inference"]["use_tta"],
            stage1_pretrained=d["model"]["stage1_pretrained"],
            lam=d["knobs"]["lam"],
            n=d["knobs"]["n"],
            w=d["knobs"]["w"],
            alpha=d["knobs"]["alpha"],
            beta=float(d["knobs"].get("beta", 1.0)),
            sigma=float(d["knobs"].get("sigma", 8.0)),
            warmup_epochs=int(d["training"].get("warmup_epochs", 3)),
            early_stop_patience=int(d["training"].get("early_stop_patience", 8)),
            val_holdout_n=int(d["training"].get("val_holdout_n", 10)),
            cache_root=Path(d["training"].get("cache_root", "dataset/ImagePairs/cache")),
            cache_pairs=int(d["training"].get("cache_pairs", 128)),
            length_factor=int(d["training"].get("length_factor", 1)),
            stage1_lr=float(d["training"].get("stage1_lr", 1e-4)),
        )

if __name__ == "__main__":
    # 使用示例：从项目根目录的 config.yaml 加载配置
    cfg = Config.from_yaml(Path(__file__).resolve().parents[2] / "config.yaml")
    print(cfg)
