# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import sys
import types
from pathlib import Path


def _install_fake_backend(monkeypatch, site_packages, mod_name):
    package_dir = site_packages / mod_name
    package_dir.mkdir()
    backend = types.ModuleType(mod_name)
    backend.__file__ = str(package_dir / "__init__.py")
    backend.__path__ = [str(package_dir)]

    monkeypatch.setitem(sys.modules, mod_name, backend)
    for sub_name in ("_api", "_bindings", "_utils", "logging"):
        monkeypatch.setitem(
            sys.modules,
            f"{mod_name}.{sub_name}",
            types.ModuleType(f"{mod_name}.{sub_name}"),
        )


def _install_fake_torch(monkeypatch, cuda_version):
    torch = types.ModuleType("torch")
    torch_version = types.ModuleType("torch.version")
    torch_version.cuda = cuda_version
    torch.version = torch_version

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.version", torch_version)


def _import_meta_package(monkeypatch):
    module_path = (
        Path(__file__).parents[2]
        / "src"
        / "bindings"
        / "python"
        / "nixl-meta"
        / "nixl"
        / "__init__.py"
    )
    spec = importlib.util.spec_from_file_location(
        "nixl", module_path, submodule_search_locations=[str(module_path.parent)]
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "nixl", module)
    assert spec.loader is not None
    spec.loader.exec_module(module)


def test_import_sets_bundled_ucx_module_dir(monkeypatch, tmp_path):
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    ucx_dir = site_packages / "nixl_cu13.libs" / "ucx"
    ucx_dir.mkdir(parents=True)

    monkeypatch.delenv("UCX_MODULE_DIR", raising=False)
    _install_fake_torch(monkeypatch, "13.0")
    _install_fake_backend(monkeypatch, site_packages, "nixl_cu13")

    _import_meta_package(monkeypatch)

    assert os.environ["UCX_MODULE_DIR"] == str(ucx_dir)


def test_import_keeps_existing_ucx_module_dir(monkeypatch, tmp_path):
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    existing_dir = tmp_path / "custom-ucx"
    existing_dir.mkdir()
    (site_packages / "nixl_cu13.libs" / "ucx").mkdir(parents=True)

    monkeypatch.setenv("UCX_MODULE_DIR", str(existing_dir))
    _install_fake_torch(monkeypatch, "13.0")
    _install_fake_backend(monkeypatch, site_packages, "nixl_cu13")

    _import_meta_package(monkeypatch)

    assert os.environ["UCX_MODULE_DIR"] == str(existing_dir)
