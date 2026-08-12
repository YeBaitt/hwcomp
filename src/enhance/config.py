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
        )


if __name__ == "__main__":
    # 使用示例：从项目根目录的 config.yaml 加载配置
    cfg = Config.from_yaml(Path(__file__).resolve().parents[2] / "config.yaml")
    print(cfg)
