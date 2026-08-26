import argparse
from pathlib import Path

import numpy as np
from openpilot.tools.lib.logreader import LogReader
from safetensors.numpy import load_file, save_file

from .localizer import localize
from .logs import FRAME_INFO_METADATA, read_log
from .math import interpolate, nlerp_quaternions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEGMENT = "2333e255f1de1fe83ea5975b24a3cc67"


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", choices=("infer", "ref"), default="infer")
  parser.add_argument("--segment", default=SEGMENT)
  args = parser.parse_args()
  data_dir = PROJECT_ROOT / "data" / args.segment

  localization_input, output_frame_info = read_log(
    LogReader(str(data_dir / "rlog.zst"), sort_by_time=True, only_union_types=True), data_dir
  )
  output_localization = localize(localization_input)

  save_file(output_localization, data_dir / "localizer.safetensors", metadata={"schema_version": "1"})
  save_file(output_frame_info, data_dir / "frame_info.safetensors", metadata=FRAME_INFO_METADATA)

  print(f"segment={args.segment}")
  print(f"states={output_localization['states'].shape} frame_states={output_localization['frame_states'].shape}")
  print(f"outputs={data_dir}")


if __name__ == "__main__":
  main()
