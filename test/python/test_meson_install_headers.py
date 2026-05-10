# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def test_header_install_dirs_use_meson_includedir():
    meson_build = Path(__file__).resolve().parents[2] / "meson.build"
    install_header_lines = [
        line.strip()
        for line in meson_build.read_text().splitlines()
        if line.strip().startswith("install_headers(")
    ]

    assert install_header_lines
    assert all("install_dir: include_dir" in line for line in install_header_lines)
