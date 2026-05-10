# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib
import subprocess
import tempfile
import textwrap
import unittest


class PairwiseSgDescTest(unittest.TestCase):
    def test_pairwise_sg_uses_one_descriptor_for_rank_device(self):
        nixlbench_root = pathlib.Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            src = tmp_path / "pairwise_sg_desc_test.cpp"
            exe = tmp_path / "pairwise_sg_desc_test"
            src.write_text(
                textwrap.dedent(
                    """
                    #include "src/worker/nixl/pairwise_sg_desc.h"
                    #include <cassert>

                    int main() {
                        const auto init_rank = getPairwiseSgDescConfig(true, true, 2, 4, 4);
                        assert(init_rank.configured_num_devices == 4);
                        assert(init_rank.desc_count == 1);
                        assert(init_rank.first_mem_dev_id == 2);

                        const auto target_rank = getPairwiseSgDescConfig(true, false, 6, 4, 4);
                        assert(target_rank.configured_num_devices == 4);
                        assert(target_rank.desc_count == 1);
                        assert(target_rank.first_mem_dev_id == 2);

                        const auto non_sg = getPairwiseSgDescConfig(false, false, 0, 4, 3);
                        assert(non_sg.configured_num_devices == 3);
                        assert(non_sg.desc_count == 3);
                        assert(non_sg.first_mem_dev_id == 0);

                        return 0;
                    }
                    """
                )
            )

            subprocess.run(
                ["g++", "-std=c++17", "-I" + str(nixlbench_root), str(src), "-o", str(exe)],
                check=True,
            )
            subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    unittest.main()
