"""Use a given Puffer dataframe and Veritas model for inference.

Put this file and the frame in the Veritas directory and run it.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# =============================================================================
# Script arguments and main function.
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--frame", default="./puffer_data.csv", help="Frame to turn into a dataset."
)
parser.add_argument(
    "--datadir",
    default="./custom_dataset",
    help="Directory to store the dataset in.",
)
parser.add_argument(
    "--model",
    default="./logs/fit/20230725115631:Controlled-GT-Cubic-BBA-LMH-gaussian.asym-v10",
    help="Path to the model to use.",
)
parser.add_argument(
    "--transform",
    default="./logs/transform/",
    help="Directory where Veritas stores outputs.",
)
parser.add_argument("--time-stepsize", default=5.0, help="Time stepsize.")
parser.add_argument(
    "--num-random-samples", default=1, help="Number of random samples."
)
parser.add_argument(
    "--num-sample-seconds", default=300, help="Number of sample seconds."
)
parser.add_argument("--device", default=device, help="Device to use.")
parser.add_argument("--jit", default=True, help="Use JIT.")
parser.add_argument("--seed", default=42, help="Random seed.")


def main():
    """Parse arguments, prepare the dataset, and run inference."""
    args = parser.parse_args()
    datadir = Path(args.datadir)
    output_dir = Path(args.transform)
    test_frame = pd.read_csv(args.frame)
    assert test_frame["day"].nunique() == 1, "Frame must be from one day."
    # Use day with true random suffix (even if random seeds are fixed) to avoid
    # collisions when running multiple instances of this script.
    suffix = f"{test_frame['day'].iloc[0]}_{os.urandom(8).hex()}"

    print(f"Preparing {datadir}.")
    shutil.rmtree(datadir, ignore_errors=True)
    datadir.mkdir(parents=True, exist_ok=True)
    session_map = prepare_dataset(
        frame=test_frame,
        datadir=datadir,
        trained_model=args.model,
        time_stepsize=int(args.time_stepsize),
        max_duration=int(args.num_sample_seconds),
    )

    # input("Press any key to run inference command.")
    print("Running inference command.")
    run_command(
        suffix=suffix,
        datadir=datadir,
        num_random_samples=int(args.num_random_samples),
        num_sample_seconds=int(args.num_sample_seconds),
        seed=args.seed,
        device=args.device,
        jit=args.jit,
    )

    print("Fetching results.")
    results = load_results(
        output_dir=output_dir,
        suffix=suffix,
        session_map=session_map,
        time_stepsize=int(args.time_stepsize),
        cleanup=False,
    )

    # TODO: combine with metadata and create simulator!


# =============================================================================
# Constants for frame parsing and formatting.
# =============================================================================

# Veritas frame metadata.
veritas_columns = [
    "start_time",
    "end_time",
    "size",
    "trans_time",
    "cwnd",
    "rtt",
    "rto",
    "ssthresh",
    "last_snd",
    "min_rtt",
    "delivery_rate",
    "bytes_sent",
    "bytes_retrans",
    "client_trans_time",
]
veritas_scale = {
    "rtt": 1000,  # s -> ms,
    "min_rtt": 1000,  # s -> ms,
    "trans_time": 1000,  # s -> ms,
    "rto": 1000,  # s -> ms,
    "size": 1.0 / 1000,  # B -> kB,
}
veritas_output_scale = 1e6 / 8  # Mbit/s -> B/s
veritas_ground_truth_columns = ["start_time", "bandwidth"]
veritas_time_columns = ["start_time", "end_time"]
veritas_formats = {
    # Veritas requires datetimes to be in this explicit format.
    **{col: "datetime64[ns]" for col in veritas_time_columns},
    # Veritas cannot handle ints, convert them to floats.
    **{
        col: float for col in veritas_columns if col not in veritas_time_columns
    },
}

# Puffer frame metadata.
puffer_session_column = "session"
puffer_column_map = {  # Rename to fit Veritas' column names.
    "time (ns GMT)": "start_time",
    "time (ns GMT) acked": "end_time",
}
utzfmt = "%Y-%m-%dT%H:%M:%S.%fZ"


# =============================================================================
# Prepare inputs.
# =============================================================================


def prepare_dataset(
    *,
    frame,
    datadir,
    trained_model,
    time_stepsize,
    max_duration,
    truncate=False,
):  # TODO false as default.
    """Create a dataset from `session_frame` in datadir and run inference."""
    # We need to bin the frame into time bins of max_duration
    frame = frame.assign(
        time_bin=(frame["time_acked"] // max_duration).astype(int)
    )
    if truncate:  # Only keep the first `max_duration` seconds.
        frame = frame[frame["time_bin"] == 0]

    # Set up subdirectories.
    directory = Path(datadir).absolute()
    for subdir in ("video_session_streams", "ground_truth_capacity"):
        (directory / subdir).mkdir(parents=True, exist_ok=True)

    # For every session in the frame, create the required files.
    file_list = []
    hashes = {
        "video_session_streams": {},
        "ground_truth_capacity": {},
    }
    # Also track which sessions the files belong to so we can reassemble them.
    session_map = {}
    group = [puffer_session_column, "time_bin"]
    for (session, bin_idx), session_frame in frame.groupby(group):
        processed, fake_ground_truth = process_session_frame(
            session_frame, time_stepsize
        )
        if len(processed) < 2:
            continue  # Veritas requires at least two observations.
        assert len(fake_ground_truth) > 0

        # Create a unique fileid and add it to the session map along with the
        # session duration.
        filename = uuid.uuid4().hex
        duration = (
            processed["end_time"].max() - processed["start_time"].min()
        ).total_seconds()
        session_map.setdefault(session, {})[bin_idx] = (filename, duration)

        # Save files.
        s_file = (directory / f"video_session_streams/{filename}").absolute()
        gt_file = (directory / f"ground_truth_capacity/{filename}").absolute()
        csv_kwargs = dict(index=False, date_format=utzfmt)
        processed.to_csv(s_file, **csv_kwargs)
        fake_ground_truth.to_csv(gt_file, **csv_kwargs)

        # Update file list and hashes.
        file_list.append(s_file.name)
        hashes["video_session_streams"][s_file.name] = get_fhash(s_file)
        hashes["ground_truth_capacity"][gt_file.name] = get_fhash(gt_file)

    listfile = directory / "full.json"
    with open(listfile, "w") as file:
        json.dump(file_list, file)
    with open(directory / "fhash.json", "w") as file:
        json.dump(hashes, file)

    # Prepare the model (we need to adjust the arguments.json file).
    trained_model = copy_model(trained_model, directory)

    return session_map


def process_session_frame(session_frame, time_stepsize):
    """Generate filename, updated session frame, and ground truth.

    Missing values are filled with zeros, as done in the Veritas datasets.

    While we don't need the ground truth, it seems to be required for Veritas
    to run, so we create a fake dataframe for it.
    """
    assert session_frame[puffer_session_column].nunique() == 1
    start_time, end_time = veritas_time_columns
    time_stepsize = pd.Timedelta(seconds=time_stepsize)

    # Prepare the session frame:
    # rename columns, fill missing values, sort, set dtypes.
    session_frame = (
        session_frame.rename(columns=puffer_column_map)
        .reindex(columns=veritas_columns, fill_value=0.0)
        .sort_values(start_time)
        .astype(veritas_formats)
    )

    # Create the ground truth frame using the correct time stepsize.
    first_step = session_frame[start_time].min()
    # At least two steps: start and end. Veritas cuts off everything after,
    # but refuses to run if there is not enough ground truth.
    gt_time_steps = 2 + max(
        1, ceil((session_frame[end_time].max() - first_step) / time_stepsize)
    )
    gt_frame = pd.DataFrame(
        {
            start_time: pd.date_range(
                start=first_step, periods=gt_time_steps, freq=time_stepsize
            ),
        }
    ).reindex(columns=veritas_ground_truth_columns, fill_value=0.0)

    # Finally, scale columns (after time has been computed in seconds).
    scale = {k: v for k, v in veritas_scale.items() if k in session_frame}
    scale_cols = list(scale)
    session_frame[scale_cols] = session_frame[scale_cols] * pd.Series(scale)

    return session_frame, gt_frame


def copy_model(model_path: Path, datadir):
    """Copy an existing model and update arguments.json.

    `arguments.json` contains the lists of train/valid/test files to use;
    so if we want to do inference on a different dataset, we need to update
    that file; we copy the model into datadir and safely modify it there.
    """
    target_dir = Path(datadir) / "model"
    argfile = Path(target_dir) / "arguments.json"
    listfile = Path(datadir).absolute() / "full.json"
    assert listfile.exists(), f"`{str(listfile)} does not exist!"

    shutil.copytree(model_path, target_dir)

    with open(argfile, "r") as file:
        args = json.load(file)
    args["train"] = args["valid"] = args["test"] = str(listfile)
    with open(argfile, "w") as file:
        json.dump(args, file)

    return target_dir


def get_fhash(path: str) -> str:
    """Veritas hash function for files."""
    if not os.path.isfile(path):
        return ""
    else:
        with open(path, "rb") as file:
            hunit = hashlib.sha256()
            while chunk := file.read(1024):
                hunit.update(chunk)
        return hunit.hexdigest()


# =============================================================================
# Running veritas.
# =============================================================================


def run_command(
    *,
    suffix,
    datadir,
    num_random_samples,
    num_sample_seconds,
    seed,
    device,
    jit,
):
    cmd = (
        f"{sys.executable} transform.py "
        f"--suffix {suffix} "
        f"--dataset {datadir} "
        f"--transform {datadir}/full.json "
        f"--seed {seed} --device {device} "
        f"{'--jit' if jit else ''} "
        f"--resume {datadir}/model "
        f"--num-random-samples {num_random_samples} "
        f"--num-sample-seconds {num_sample_seconds}"
    )

    print(cmd)
    subprocess.run(cmd, shell=True, check=True)


# =============================================================================
# Loading results.
# =============================================================================


def load_results(output_dir, suffix, session_map, time_stepsize, cleanup=False):
    """Find and load the results of the inference run."""
    # Veritas prepends a timestamp to the suffix, so we need to search through
    # output_dir to find a match.
    potential_dirs = [
        d for d in Path(output_dir).iterdir() if d.name.endswith(suffix)
    ]
    assert len(potential_dirs) == 1, "Found multiple matching directories."
    output_dir = potential_dirs[0] / "sample"

    # Again, veritas prepends something to the filename, this time an index.
    # To avoid search through the filenames multiple times, make a map.
    dir_map = {}
    for subdir in output_dir.iterdir():
        fileid = ".".join(subdir.name.split(".")[1:])
        dir_map[fileid] = subdir

    session_traces = {
        session: load_and_merge_samples(bin_map, dir_map, time_stepsize)
        for session, bin_map in session_map.items()
    }

    if cleanup:
        shutil.rmtree(output_dir, ignore_errors=True)

    return session_traces


def load_and_merge_samples(bin_map, dir_map, time_stepsize):
    """We can only predict a session up to a fixed time, so fetch all parts.

    Return a (seconds, bytes) trace
    """
    bins = sorted(bin_map)
    all_rates = [[0]]  # We start at 0.
    for bin in bins:
        # Veritas adds a header columns simply named '0'.
        file_id, duration = bin_map[bin]
        filename = dir_map[file_id] / "sample_full.csv"
        max_steps = np.ceil(duration / time_stepsize).astype(int)
        rates = pd.read_csv(filename, nrows=max_steps)["0"].to_numpy()
        rates *= veritas_output_scale
        all_rates.append(rates)
    all_rates = np.concatenate(all_rates)

    # Turn sequence of rates into a trace of cumulative bytes.
    # In the first `time_stepsize` we transmit `all_rates[1]` bytes.
    # Thus, because we prepended a 0, we can just use cumsum.
    trace_bytes = np.cumsum(all_rates * time_stepsize)
    trace_seconds = np.arange(len(trace_bytes)) * time_stepsize
    return (trace_seconds, trace_bytes)


# =============================================================================
# Run the script
# =============================================================================

if __name__ == "__main__":
    main()
