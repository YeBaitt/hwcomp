"""提交命名校验与打包为 zip（内含 output_dir/）。"""
import tempfile
import zipfile
from pathlib import Path

def validate_names(images: list[Path], expected: list[str]) -> list[str]:
    """校验提交图片文件名与期望列表，返回缺少/多余的差异。"""
    errors = []
    names = {p.name for p in images}
    for e in expected:
        if e not in names:
            errors.append(f"缺少 {e}")
    for n in sorted(names - set(expected)):
        errors.append(f"多余 {n}")
    return errors

def package(submit_dir: Path, zip_path: Path) -> None:
    """将 submit_dir 下的 jpg 打包为含 output_dir/ 的 zip。"""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for img in sorted(submit_dir.glob("*.jpg")):
            z.write(img, f"output_dir/{img.name}")

if __name__ == "__main__":
    # 使用示例：临时目录写两个 jpg，校验命名差异并打包为 zip
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "output_dir"
        d.mkdir()
        (d / "case1.jpg").write_bytes(b"\xff\xd8\xff\xe0")
        (d / "case2.jpg").write_bytes(b"\xff\xd8\xff\xe0")
        print(validate_names(list(d.glob("*.jpg")), ["case1.jpg", "case2.jpg"]))
        z = Path(td) / "my_work.zip"
        package(d, z)
        print(f"已打包: {z}")
