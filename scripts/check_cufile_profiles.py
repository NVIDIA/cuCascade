# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Validate cuFile examples after stripping their full-line documentation comments."""

import json
import pathlib
import sys


def main():
    failed = False
    for name in sys.argv[1:]:
        try:
            source = pathlib.Path(name).read_text()
            json.loads(
                "\n".join(
                    line
                    for line in source.splitlines()
                    if not line.lstrip().startswith("//")
                )
            )
        except (OSError, ValueError) as error:
            print(f"{name}: {error}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
