import glob
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional

import tensorflow as tf
from tqdm import tqdm

from scenariomax.logger_utils import get_logger
from scenariomax.unified_to_waymo.build_waymo_example import build_waymo_example, write_to_tfrecord


logger = get_logger(__name__)


def process_pickle_file(pickle_file: str, output_dir: str) -> str:
    """
    Process a single pickle file and convert it to Waymo format.
    
    Args:
        pickle_file: Path to the pickle file
        output_dir: Directory to write the TFRecord file
        
    Returns:
        Path to the written TFRecord file
    """
    try:
        # Load the pickle file
        with open(pickle_file, 'rb') as f:
            scenario_data = pickle.load(f)
        
        # Convert to Waymo format
        waymo_scenario = build_waymo_example(scenario_data)
        
        # Create output file path
        base_name = os.path.basename(pickle_file)
        file_name = os.path.splitext(base_name)[0]
        output_file = os.path.join(output_dir, f"{file_name}.tfrecord")
        
        # Write to TFRecord file
        write_to_tfrecord(waymo_scenario, output_file)
        
        return output_file
    except Exception as e:
        logger.error(f"Error processing {pickle_file}: {e}")
        return ""


def process_directory(input_dir: str, output_dir: str, num_workers: int = 8, file_pattern: str = "*.pkl") -> List[str]:
    """
    Process all pickle files in a directory and convert them to Waymo format.
    
    Args:
        input_dir: Directory containing pickle files
        output_dir: Directory to write TFRecord files
        num_workers: Number of worker processes to use
        file_pattern: File pattern to match pickle files
        
    Returns:
        List of paths to written TFRecord files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all pickle files
    pickle_files = glob.glob(os.path.join(input_dir, file_pattern))
    logger.info(f"Found {len(pickle_files)} pickle files in {input_dir}")
    
    output_files = []
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_pickle_file, pickle_file, output_dir): pickle_file
            for pickle_file in pickle_files
        }
        
        # Process as they complete
        with tqdm(total=len(pickle_files), desc="Converting to Waymo format") as pbar:
            for future in as_completed(future_to_file):
                pickle_file = future_to_file[future]
                try:
                    output_file = future.result()
                    if output_file:
                        output_files.append(output_file)
                except Exception as e:
                    logger.error(f"Failed to process {pickle_file}: {e}")
                
                pbar.update(1)
    
    return output_files


def merge_tfrecord_files(input_dir: str, output_file: str) -> None:
    """
    Merge multiple TFRecord files into a single file.
    
    Args:
        input_dir: Directory containing TFRecord files
        output_file: Path to output merged TFRecord file
    """
    # Find all TFRecord files
    tfrecord_files = glob.glob(os.path.join(input_dir, "*.tfrecord"))
    logger.info(f"Found {len(tfrecord_files)} TFRecord files in {input_dir}")
    
    # Merge files
    with tf.io.TFRecordWriter(output_file) as writer:
        for tfrecord_file in tqdm(tfrecord_files, desc="Merging TFRecord files"):
            try:
                for record in tf.data.TFRecordDataset([tfrecord_file]):
                    writer.write(record.numpy())
            except Exception as e:
                logger.error(f"Error reading {tfrecord_file}: {e}")
    
    logger.info(f"Merged {len(tfrecord_files)} TFRecord files into {output_file}")


def postprocess_waymo(
    output_path: str,
    worker_index: int,
    scenarios: List,
    convert_func,
    dataset_version: str,
    dataset_name: str,
    pbar,
    **kwargs,
) -> None:
    """
    Post-process function for the Waymo converter.
    This is called by the write_to_directory function in the main conversion pipeline.
    
    Args:
        output_path: Path to output directory
        worker_index: Worker index
        scenarios: List of scenarios to process
        convert_func: Function to convert scenarios
        dataset_version: Dataset version
        dataset_name: Dataset name
        pbar: Progress bar
        **kwargs: Additional arguments
    """
    # Create output directory for TFRecord files
    tfrecord_dir = os.path.join(output_path, "tfrecord")
    os.makedirs(tfrecord_dir, exist_ok=True)
    
    # Process each scenario
    for scenario in scenarios:
        try:
            # Convert scenario to unified format
            scenario_data = convert_func(scenario, dataset_version, **kwargs)
            
            # Get scenario ID
            scenario_id = scenario_data[SD.METADATA][SD.ID]
            
            # Convert to Waymo format
            waymo_scenario = build_waymo_example(scenario_data)
            
            # Write to TFRecord file
            output_file = os.path.join(tfrecord_dir, f"{scenario_id}.tfrecord")
            write_to_tfrecord(waymo_scenario, output_file)
            
            pbar.update(1)
        except Exception as e:
            logger.error(f"Error processing scenario {scenario}: {e}")
            pbar.update(1)


def process_unified_to_waymo(input_dir: str, output_dir: str, num_workers: int = 8, merged_filename: Optional[str] = None) -> None:
    """
    Convert unified pickle format to Waymo TFRecord format.
    
    Args:
        input_dir: Directory containing pickle files
        output_dir: Directory to write TFRecord files
        num_workers: Number of worker processes to use
        merged_filename: If provided, merge all TFRecord files into a single file with this name
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary directory for individual TFRecord files
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process all pickle files
    output_files = process_directory(input_dir, temp_dir, num_workers)
    logger.info(f"Converted {len(output_files)} pickle files to TFRecord format")
    
    # Merge files if requested
    if merged_filename:
        merged_file = os.path.join(output_dir, merged_filename)
        merge_tfrecord_files(temp_dir, merged_file)
        logger.info(f"Merged all TFRecord files into {merged_file}")
    
    logger.info(f"Conversion complete. Output files are in {output_dir}")


# This can be used as a standalone script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert unified pickle format to Waymo TFRecord format")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing pickle files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write TFRecord files")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes to use")
    parser.add_argument("--merged_filename", type=str, default=None, help="If provided, merge all TFRecord files into a single file with this name")
    
    args = parser.parse_args()
    
    process_unified_to_waymo(args.input_dir, args.output_dir, args.num_workers, args.merged_filename) 