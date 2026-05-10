/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_WORKER_NIXL_PAIRWISE_SG_DESC_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_WORKER_NIXL_PAIRWISE_SG_DESC_H

#include <cstddef>

struct xferBenchDescConfig {
    size_t configured_num_devices;
    size_t desc_count;
    size_t first_mem_dev_id;
};

static inline xferBenchDescConfig
getPairwiseSgDescConfig(bool is_pairwise_sg,
                        bool is_initiator,
                        int rank,
                        int num_initiator_dev,
                        int num_target_dev) {
    size_t configured_num_devices = is_initiator ? num_initiator_dev : num_target_dev;

    if (!is_pairwise_sg) {
        return {configured_num_devices, configured_num_devices, 0};
    }

    size_t mem_dev_id = is_initiator ? rank : rank - num_initiator_dev;
    return {configured_num_devices, 1, mem_dev_id};
}

#endif // NIXL_BENCHMARK_NIXLBENCH_SRC_WORKER_NIXL_PAIRWISE_SG_DESC_H
