# Waymo Format Conversion Guide

This guide explains how to use ScenarioMax to convert various autonomous driving datasets back to the native Waymo Open Dataset format.

## Overview

ScenarioMax can now convert the following datasets to Waymo Open Dataset format:

1. Waymo Open Motion Dataset (round-trip conversion)
2. nuPlan (cross-dataset conversion)
3. nuScenes (cross-dataset conversion)
4. Argoverse2 (cross-dataset conversion)

The conversion process follows these steps:
1. Convert the raw dataset to the unified pickle format
2. Convert the unified pickle format to Waymo Open Dataset format

## Conversion Methods

### One-Step Conversion

For a direct conversion from raw data to Waymo format, use:

```bash
python scenariomax/convert_dataset.py \
  --waymo_src /path/to/waymo/data \
  --dst /path/to/output/directory \
  --target_format waymo_format \
  --num_workers 8
```

Similarly for other datasets:

```bash
python scenariomax/convert_dataset.py \
  --nuplan_src /path/to/nuplan/data \
  --dst /path/to/output/directory \
  --target_format waymo_format \
  --num_workers 8
```

### Two-Step Conversion

This approach gives you more flexibility and allows for better error handling:

#### Step 1: Convert to Unified Pickle Format

```bash
python scenariomax/convert_dataset.py \
  --waymo_src /path/to/waymo/data \
  --dst /path/to/intermediate/directory \
  --target_format pickle \
  --num_workers 8
```

#### Step 2: Convert Pickle to Waymo Format

```bash
python scenariomax/scripts/convert_pickle_to_waymo.py \
  -d waymo \
  --input_dir /path/to/intermediate/directory \
  --output_dir /path/to/final/directory \
  --num_workers 8 \
  --merged_filename waymo_dataset.tfrecord
```

## Script Arguments

The `convert_pickle_to_waymo.py` script supports the following arguments:

| Argument | Description |
|----------|-------------|
| `-d`, `--dataset` | Dataset to process (waymo, nuplan, nuscenes, argoverse2) |
| `--input_dir` | Directory containing pickle files |
| `--output_dir` | Directory to write TFRecord files |
| `--num_workers` | Number of worker processes (default: 8) |
| `--merged_filename` | If provided, merge all TFRecord files into a single file |
| `--log_level` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `--log_file` | Optional log file path |

## Example: Converting nuPlan to Waymo Format

```bash
# Step 1: Convert nuPlan to pickle
python scenariomax/convert_dataset.py \
  --nuplan_src /path/to/nuplan/data \
  --dst /path/to/intermediate \
  --target_format pickle \
  --num_workers 8

# Step 2: Convert pickle to Waymo format
python scenariomax/scripts/convert_pickle_to_waymo.py \
  -d nuplan \
  --input_dir /path/to/intermediate \
  --output_dir /path/to/output \
  --num_workers 8 \
  --merged_filename nuplan_as_waymo.tfrecord
```

## Validating the Output

To validate that the converted data is in the correct Waymo format, you can use Waymo's own tools:

```python
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2

# Load a converted scenario
dataset = tf.data.TFRecordDataset('/path/to/output/waymo_dataset.tfrecord')

# Parse the first scenario
for record in dataset.take(1):
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(record.numpy())
    
    print(f"Scenario ID: {scenario.scenario_id}")
    print(f"Number of tracks: {len(scenario.tracks)}")
    print(f"Number of timestamps: {len(scenario.timestamps_seconds)}")
    print(f"Current time index: {scenario.current_time_index}")
```

## Limitations

- When converting from non-Waymo datasets, some dataset-specific information may be lost or approximated
- Traffic light states may have different semantics between datasets
- Some road types or lane markings may not have direct equivalents in the Waymo format

## Advanced Usage: Programmatic API

You can also use the conversion functionality programmatically in your own Python code:

```python
from scenariomax.unified_to_waymo.build_waymo_example import pickle_to_waymo
from scenariomax.unified_to_waymo.process_waymo import process_unified_to_waymo

# Convert a single pickle file
pickle_to_waymo(
    input_path="/path/to/scenario.pkl",
    output_path="/path/to/scenario.tfrecord"
)

# Convert a directory of pickle files
process_unified_to_waymo(
    input_dir="/path/to/pickle/directory",
    output_dir="/path/to/output/directory",
    num_workers=8,
    merged_filename="merged_dataset.tfrecord"
)
``` 