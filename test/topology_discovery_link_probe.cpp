/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cucascade/memory/topology_discovery.hpp>

int main()
{
  cucascade::memory::topology_discovery discovery;
  return discovery.discover() ? 0 : 1;
}
