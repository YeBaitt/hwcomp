import zipfile

from enhance.submit.package import package, validate_names


def _make_dir(tmp_path):
    d = tmp_path / "output_dir"
    d.mkdir()
    for i in range(1, 4):
        (d / f"case{i}.jpg").write_bytes(b"\xff\xd8\xff\xe0")
    (d / "case5.jpg").write_bytes(b"\xff\xd8\xff\xe0")  # 多余文件
    (d / "bad_name.png").write_bytes(b"\xff\xd8\xff\xe0")  # 非 jpg
    return d


def test_validate_names(tmp_path):
    d = _make_dir(tmp_path)
    errs = validate_names(list(d.glob("*.jpg")), ["case1.jpg", "case2.jpg", "case3.jpg"])
    # case5 多余、case1-3 齐全 → 报出 case5 多余；bad_name.png 不在 jpg 扫描内
    assert any("case5.jpg" in e for e in errs)
    assert not any("case1.jpg" in e for e in errs)


def test_package_zip_structure(tmp_path):
    d = _make_dir(tmp_path)
    z = tmp_path / "my_work.zip"
    package(d, z)
    assert z.exists()
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
    assert "output_dir/case1.jpg" in names
    assert "output_dir/case2.jpg" in names
