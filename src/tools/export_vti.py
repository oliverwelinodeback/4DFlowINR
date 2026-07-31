"""
Export 4DFlowINR HDF5 predictions to VTI for visualization in ParaView.

HDF5 files are the numerical output of 4DFlowINR
This script provides optional VTI export for visualization

--------
Export one frame:

    python src/tools/export_vti.py \
        models/example/SR_it008000.h5 \
        --index 12

Export all frames:

    python src/tools/export_vti.py \
        models/example/SR_it008000.h5 \
        --all
"""

import argparse
from pathlib import Path
import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export 4DFlowINR HDF5 predictions to VTI."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input HDF5 prediction file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output basename without .vti extension. "
            "Defaults to the input filename."
        ),
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--index",
        type=int,
        default=None,
        help="Cardiac-frame index to export.",
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Export all cardiac frames.",
    )

    return parser.parse_args()

def _frame(dataset, index):
    """Return one spatial frame from a 3D or time-resolved 4D dataset."""

    array = np.asarray(dataset)

    if array.ndim == 4:
        return array[index]

    if array.ndim == 3:
        return array

    raise ValueError(
        f"Expected a 3D or 4D array, got shape {array.shape}."
    )

def export_frame(h5_filename, output_basename, index):
    """Export one cardiac frame from an HDF5 prediction file."""

    # Optional dependency: core training does not require pyevtk.
    try:
        from pyevtk.hl import imageToVTK
    except ImportError as exc:
        raise ImportError(
            "VTI export requires pyevtk. "
            "Install it separately to use this visualization tool."
        ) from exc

    with h5py.File(h5_filename, "r") as hf:

        required = ("u", "v", "w")
        missing = [
            key
            for key in required
            if key not in hf
        ]

        if missing:
            raise KeyError(
                "Missing required HDF5 datasets: "
                + ", ".join(missing)
            )

        u = _frame(hf["u"], index)
        v = _frame(hf["v"], index)
        w = _frame(hf["w"], index)

        if not (u.shape == v.shape == w.shape):
            raise ValueError(
                "Velocity-component shapes do not match: "
                f"u={u.shape}, v={v.shape}, w={w.shape}"
            )

        if "spacing" not in hf.attrs:
            raise KeyError(
                "HDF5 file does not contain the 'spacing' attribute."
            )

        spacing = tuple(
            float(value)
            for value in hf.attrs["spacing"]
        )

        origin = tuple(
            float(value)
            for value in hf.attrs.get(
                "origin",
                (0.0, 0.0, 0.0),
            )
        )

        cell_data = {
            "velocity": (
                np.ascontiguousarray(u),
                np.ascontiguousarray(v),
                np.ascontiguousarray(w),
            )
        }

        if "mask" in hf:
            mask = _frame(hf["mask"], index)
            cell_data["mask"] = np.ascontiguousarray(
                (mask != 0).astype(np.uint8)
            )

        if "p" in hf:
            pressure = _frame(hf["p"], index)
            cell_data["pressure"] = np.ascontiguousarray(
                pressure
            )

        gradient_keys = ("p_x", "p_y", "p_z")

        if all(key in hf for key in gradient_keys):
            p_x = _frame(hf["p_x"], index)
            p_y = _frame(hf["p_y"], index)
            p_z = _frame(hf["p_z"], index)

            cell_data["pressure_gradient"] = (
                np.ascontiguousarray(p_x),
                np.ascontiguousarray(p_y),
                np.ascontiguousarray(p_z),
            )

    imageToVTK(
        str(output_basename),
        origin=origin,
        spacing=spacing,
        pointData={},
        cellData=cell_data,
    )

def main():
    args = parse_args()

    input_path = args.input.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    if args.output is None:
        output_base = input_path.with_suffix("")
    else:
        output_base = args.output.expanduser()

    with h5py.File(input_path, "r") as hf:
        if "u" not in hf:
            raise KeyError(
                "Input file does not contain dataset 'u'"
            )

        u_shape = hf["u"].shape

        if len(u_shape) == 4:
            n_frames = u_shape[0]
        elif len(u_shape) == 3:
            n_frames = 1
        else:
            raise ValueError(
                f"Unexpected velocity shape: {u_shape}"
            )

    if args.all:

        for index in range(n_frames):
            frame_output = Path(
                f"{output_base}_t{index:02d}"
            )

            export_frame(
                input_path,
                frame_output,
                index,
            )

            print(
                f"Exported frame {index}: "
                f"{frame_output}.vti"
            )

    else:

        index = (
            args.index
            if args.index is not None
            else 0
        )

        if not 0 <= index < n_frames:
            raise IndexError(
                f"Frame index {index} is outside "
                f"the valid range 0-{n_frames - 1}."
            )

        export_frame(
            input_path,
            output_base,
            index,
        )

        print(
            f"Exported frame {index}: "
            f"{output_base}.vti"
        )

if __name__ == "__main__":
    main()