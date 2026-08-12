from pathlib import Path

from enhance.config import Config

def test_from_yaml(tmp_path):
    yaml_text = """
data:
  root: dataset
  image_pairs_train_dir: dataset/ImagePairs/train
  val_dir: dataset/huawei/val
  test_dir: dataset/huawei/test
output:
  out_root: output
  ckpt_root: checkpoints
  submit_dir: output_dir
  zip_path: my_work.zip
training:
  patch_size: 256
  batch_size: 8
  num_workers: 4
  seed: 42
  kind_2k_weight: 0.7
  stage1_epochs: 150
inference:
  tile_size: 512
  overlap: 128
  steps: 20
  scheduler: dpm
  batch_tiles: 8
  use_tta: false
model:
  stage1_pretrained: vendor/NAFNet/weights/sidd.pth
knobs:
  lam: 0.3
  n: 0.15
  w: 2.0
  alpha: 0.4
"""
    p = tmp_path / "config.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    cfg = Config.from_yaml(p)
    assert cfg.patch_size == 256
    assert cfg.kind_2k_weight == 0.7
    assert cfg.stage1_epochs == 150
    assert cfg.tile_size == 512
    assert cfg.lam == 0.3
    assert cfg.w == 2.0
    assert cfg.zip_path == Path("my_work.zip")
