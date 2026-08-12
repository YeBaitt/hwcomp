"""提交命名校验与打包为 zip（内含 output_dir/）。"""
import zipfile
from pathlib import Path


def validate_names(images: list[Path], expected: list[str]) -> list[str]:
    errors = []
    names = {p.name for p in images}
    for e in expected:
        if e not in names:
            errors.append(f"缺少 {e}")
    for n in sorted(names - set(expected)):
        errors.append(f"多余 {n}")
    return errors


def package(submit_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for img in sorted(submit_dir.glob("*.jpg")):
            z.write(img, f"output_dir/{img.name}")
