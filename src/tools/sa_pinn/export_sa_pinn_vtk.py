"""
Export SA-PINN collocation fields for ParaView.

Examples
--------
Six time bins:

    python src/tools/sa_pinn/export_sa_pinn_vtk.py \
        --checkpoint models/.../sa_pinn_it030000.pth \
        --data-file data/.../case_LR.h5 \
        --dx 0.001 --dt 0.04 \
        --time-bins 6

One input phase:

    python src/tools/sa_pinn/export_sa_pinn_vtk.py \
        --checkpoint models/.../sa_pinn_it030000.pth \
        --data-file data/.../case_LR.h5 \
        --dx 0.001 --dt 0.04 \
        --phase-index 12

# > python src/tools/sa_pinn/export_sa_pinn_vtk.py    
# --checkpoint ../../x_sebjo/SRFlow/models/260505_SAPINN_factorial/SA_PINN_HV01_HRLR2_SA+LBFGS/checkpoints/260505_Vincent_it030000.pth     
# --data-file ../../x_sebjo/SRFlow/data/Vincent/healthy/HV01_05mm3_20ms_HRLR_sv17_tSNR2.h5     
# --dx 0.0005    
#  --dt 0.02    
#   --time-bins 6

"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

SCRIPT_PATH = Path(__file__).resolve()
SRC_DIR = SCRIPT_PATH.parents[2]
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import networks
from configs.extensions.sa_pinn.sa_adam_lbfgs import get_config
from utils.data_io import load_data
from utils.loss_utils import navier_stokes_loss
from utils.prepare_data import prepare_data, sample_collocation_points
from utils.reproducibility import set_seed

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export SA weights, PDE residuals, and predicted velocity as ParaView point clouds."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--dx", type=float, required=True, help="Input spacing [m].")
    parser.add_argument("--dt", type=float, required=True, help="Input timestep [s].")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sa_pinn_vtk"),
    )
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--collocation-file",
        type=Path,
        default=None,
        help=(
            "Optional exact training coordinates as .npy or .npz "
            "(key: xyz_collocation)."
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=5_000)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument(
        "--coordinate-units",
        choices=("m", "mm", "normalized"),
        default="m",
    )

    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument(
        "--time-bins",
        type=int,
        default=None,
        help="Number of equal-width time bins; default 6.",
    )
    time_group.add_argument(
        "--phase-index",
        type=int,
        default=None,
        help="Export one zero-based input cardiac phase.",
    )
    return parser.parse_args()


def resolve_path(path):
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()

    for candidate in (
        Path.cwd() / path,
        REPO_ROOT / path,
        SRC_DIR / path,
    ):
        if candidate.exists():
            return candidate.resolve()

    return (Path.cwd() / path).resolve()


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_saved_collocation(path):
    if path.suffix.lower() == ".npy":
        xyz = np.load(path)
    elif path.suffix.lower() == ".npz":
        with np.load(path) as archive:
            xyz = archive["xyz_collocation"]
    else:
        raise ValueError("--collocation-file must be .npy or .npz.")

    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 4:
        raise ValueError(
            f"Expected collocation shape (N, 4), got {xyz.shape}."
        )
    return xyz


def get_seed(checkpoint, config, requested_seed):
    if requested_seed is not None:
        return int(requested_seed)

    saved_config = checkpoint.get("config")
    if isinstance(saved_config, dict):
        if saved_config.get("random_seed") is not None:
            return int(saved_config["random_seed"])

    return int(getattr(config, "random_seed", 1234))


def get_collocation_points(
    checkpoint,
    config,
    xyz_data,
    mask_flat,
    collocation_file,
):
    if checkpoint.get("xyz_collocation") is not None:
        xyz = np.asarray(
            checkpoint["xyz_collocation"],
            dtype=np.float32,
        )
        source = "checkpoint"
    elif collocation_file is not None:
        path = resolve_path(collocation_file)
        if not path.is_file():
            raise FileNotFoundError(f"Collocation file not found: {path}")
        xyz = load_saved_collocation(path)
        source = str(path)
    else:
        c_weights = checkpoint.get("c_weights")
        if c_weights is not None:
            config.collocation_points = len(c_weights)

        xyz = sample_collocation_points(
            config,
            xyz_data,
            mask_flat.astype(np.uint8),
        ).astype(np.float32)
        source = "regenerated"

        print(
            "\n[WARNING] Original collocation coordinates were not stored.\n"
            "          The cloud was regenerated from the seed. Weight locations\n"
            "          are exact only if code, data, seed, and RNG order match\n"
            "          the original training run.\n"
        )

    if xyz.ndim != 2 or xyz.shape[1] != 4:
        raise ValueError(
            f"Expected collocation shape (N, 4), got {xyz.shape}."
        )
    return xyz, source


def denormalize_coordinates(xyz, factors, config):
    if config.coords_normalization != "standardize":
        raise ValueError(
            "This exporter currently supports standardized coordinates only."
        )

    mean_t, std_t, mean_x, std_x, mean_y, std_y, mean_z, std_z = factors

    time_s = xyz[:, 0] * std_t + mean_t
    spatial_m = np.column_stack(
        (
            xyz[:, 1] * std_x + mean_x,
            xyz[:, 2] * std_y + mean_y,
            xyz[:, 3] * std_z + mean_z,
        )
    )

    if config.coords_characteristic:
        time_s *= config.constants.T
        spatial_m *= config.constants.L

    return time_s.astype(np.float32), spatial_m.astype(np.float32)


def denormalize_velocity(prediction, checkpoint, config, input_u_max):
    velocity = prediction[:, :3].astype(np.float32, copy=True)

    if config.vel_normalization == "characteristic":
        velocity *= float(config.constants.U)
    elif config.vel_normalization == "max_velocity":
        velocity *= float(checkpoint.get("U_max") or input_u_max)
    else:
        raise ValueError(
            f"Unknown velocity normalization: {config.vel_normalization}"
        )
    return velocity


def evaluate_fields(config, model, xyz, factors, device, chunk):
    predictions = []
    residuals = []
    model.eval()

    for start in range(0, len(xyz), chunk):
        end = min(start + chunk, len(xyz))
        coords = (
            torch.from_numpy(xyz[start:end])
            .float()
            .to(device)
            .requires_grad_(True)
        )

        with torch.enable_grad():
            prediction = model(coords)
            residual, _, _ = navier_stokes_loss(
                prediction,
                coords,
                factors,
                config,
                return_per_point=True,
                build_graph=True,
            )

        predictions.append(prediction.detach().cpu().numpy())
        residuals.append(residual.detach().cpu().numpy())

        del coords, prediction, residual
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return (
        np.concatenate(predictions).astype(np.float32),
        np.concatenate(residuals).astype(np.float32),
    )


def make_time_selections(args, time_s, n_phases):
    dt = float(args.dt)

    if args.phase_index is not None:
        phase = args.phase_index
        if not 0 <= phase < n_phases:
            raise IndexError(
                f"Phase {phase} is outside 0-{n_phases - 1}."
            )

        center = (phase + 1) * dt
        lower = center - dt / 2
        upper = center + dt / 2
        mask = (time_s >= lower) & (time_s < upper)
        if phase == n_phases - 1:
            mask = (time_s >= lower) & (time_s <= upper)

        return [
            (
                f"phase{phase:03d}",
                np.flatnonzero(mask),
                center,
            )
        ]

    n_bins = 6 if args.time_bins is None else args.time_bins
    if n_bins < 1:
        raise ValueError("--time-bins must be >= 1.")

    # Collocation points are jittered around phase centres by about half a dt.
    edges = np.linspace(
        0.5 * dt,
        (n_phases + 0.5) * dt,
        n_bins + 1,
    )

    selections = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (time_s >= lower) & (time_s < upper)
        if index == n_bins - 1:
            mask = (time_s >= lower) & (time_s <= upper)

        ids = np.flatnonzero(mask)
        if ids.size:
            selections.append(
                (
                    f"bin{index + 1:02d}",
                    ids,
                    float((lower + upper) / 2),
                )
            )
    return selections


def subsample(indices, maximum, seed):
    if maximum is None or len(indices) <= maximum:
        return indices

    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(indices, size=maximum, replace=False)
    )


def write_pvd(path, entries):
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for time_s, vtk_path in entries:
        lines.append(
            '    <DataSet '
            f'timestep="{time_s:.9g}" group="" part="0" '
            f'file="{vtk_path.name}"/>'
        )
    lines += ["  </Collection>", "</VTKFile>"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()

    checkpoint_path = resolve_path(args.checkpoint)
    data_path = resolve_path(args.data_file)
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if args.dx <= 0 or args.dt <= 0:
        raise ValueError("--dx and --dt must be positive.")

    try:
        from pyevtk.hl import pointsToVTK
    except ImportError as exc:
        raise ImportError(
            "This tool requires pyevtk. Install it separately."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch_load(checkpoint_path, device)

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_state_dict.")

    config = get_config()
    config.data_file = str(data_path)
    config.resolution.from_file = False
    config.resolution.dx = args.dx
    config.resolution.dy = args.dx
    config.resolution.dz = args.dx
    config.resolution.dt = args.dt

    seed = get_seed(checkpoint, config, args.seed)
    set_seed(seed)

    u, v, w, p, px, py, pz, mask, config = load_data(config)
    _, xyz_data, mask_flat, _, factors, input_u_max = prepare_data(
        config, u, v, w, p, px, py, pz, mask
    )

    xyz, source = get_collocation_points(
        checkpoint,
        config,
        xyz_data,
        mask_flat,
        args.collocation_file,
    )
    print(f"Device:              {device}")
    print(f"Collocation source:  {source}")
    print(f"Collocation points:  {len(xyz):,}")

    weights = checkpoint.get("c_weights")
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        if len(weights) != len(xyz):
            raise ValueError(
                f"c_weights has {len(weights)} entries, "
                f"but xyz has {len(xyz)} points."
            )
    else:
        print("[INFO] No c_weights in checkpoint; exporting residuals only.")

    model = networks.build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Evaluating velocity and PDE residuals...")
    prediction, residual = evaluate_fields(
        config,
        model,
        xyz,
        factors,
        device,
        args.chunk,
    )

    time_s, spatial_m = denormalize_coordinates(xyz, factors, config)
    velocity = denormalize_velocity(
        prediction,
        checkpoint,
        config,
        input_u_max,
    )
    speed = np.linalg.norm(velocity, axis=1).astype(np.float32)

    finite = (
        np.isfinite(residual)
        & np.all(np.isfinite(velocity), axis=1)
        & np.all(np.isfinite(xyz), axis=1)
    )
    if weights is not None:
        finite &= np.isfinite(weights)

    xyz = xyz[finite]
    spatial_m = spatial_m[finite]
    time_s = time_s[finite]
    velocity = velocity[finite]
    speed = speed[finite]
    residual = residual[finite]
    if weights is not None:
        weights = weights[finite]

    if args.coordinate_units == "normalized":
        spatial = xyz[:, 1:4]
    elif args.coordinate_units == "mm":
        spatial = spatial_m * 1000
    else:
        spatial = spatial_m

    selections = make_time_selections(args, time_s, u.shape[0])
    if not selections:
        raise RuntimeError("No points matched the requested time selection.")

    basename = args.name or checkpoint_path.stem
    pvd_entries = []

    for selection_number, (label, ids, pvd_time) in enumerate(selections):
        ids = subsample(
            ids,
            args.max_points,
            seed + selection_number,
        )
        selected_xyz = spatial[ids]
        selected_velocity = velocity[ids]
        selected_residual = residual[ids]

        point_data = {
            "velocity": (
                np.ascontiguousarray(selected_velocity[:, 0]),
                np.ascontiguousarray(selected_velocity[:, 1]),
                np.ascontiguousarray(selected_velocity[:, 2]),
            ),
            "speed": np.ascontiguousarray(speed[ids]),
            "physics_residual": np.ascontiguousarray(selected_residual),
            "time_seconds": np.ascontiguousarray(time_s[ids]),
            "time_normalized": np.ascontiguousarray(xyz[ids, 0]),
        }

        if weights is not None:
            selected_weights = weights[ids]
            point_data["sa_weight"] = np.ascontiguousarray(selected_weights)
            point_data["weighted_residual_squared"] = np.ascontiguousarray(
                selected_weights * selected_residual**2
            )

        output_base = output_dir / f"{basename}_{label}"
        pointsToVTK(
            str(output_base),
            np.ascontiguousarray(selected_xyz[:, 0]),
            np.ascontiguousarray(selected_xyz[:, 1]),
            np.ascontiguousarray(selected_xyz[:, 2]),
            data=point_data,
        )

        vtk_path = output_base.with_suffix(".vtu")
        pvd_entries.append((pvd_time, vtk_path))
        print(f"Exported {label}: {len(ids):,} points -> {vtk_path}")

    if len(pvd_entries) > 1:
        pvd_path = output_dir / f"{basename}.pvd"
        write_pvd(pvd_path, pvd_entries)
        print(f"Exported ParaView time series: {pvd_path}")


if __name__ == "__main__":
    main()
