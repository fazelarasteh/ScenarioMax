#!/usr/bin/env python3
import argparse
import logging
import os

from scenariomax.logger_utils import get_logger, setup_logger
from scenariomax.unified_to_waymo.process_waymo import process_unified_to_waymo


logger = get_logger(__name__)


def main():
    """
    Main entry point for converting pickle files to Waymo format.
    """
    parser = argparse.ArgumentParser(description="Convert unified pickle format to Waymo TFRecord format")
    parser.add_argument(
        "-d", "--dataset", 
        type=str, 
        required=True, 
        choices=["waymo", "nuplan", "nuscenes", "argoverse2"],
        help="Dataset name to process"
    )
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True, 
        help="Directory containing pickle files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to write TFRecord files"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=8, 
        help="Number of worker processes to use"
    )
    parser.add_argument(
        "--merged_filename", 
        type=str, 
        default=None, 
        help="If provided, merge all TFRecord files into a single file with this name"
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional log file path",
    )
    
    args = parser.parse_args()
    
    # Setup logger
    setup_logger(log_level=getattr(logging, args.log_level), log_file=args.log_file)
    
    logger.info(f"Converting {args.dataset} pickle files to Waymo format")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Construct the input path based on dataset
    input_path = os.path.join(args.input_dir, args.dataset)
    if not os.path.exists(input_path):
        logger.warning(f"Dataset directory {input_path} does not exist, trying direct input path")
        input_path = args.input_dir
    
    if not os.path.exists(input_path):
        logger.error(f"Input directory {input_path} does not exist")
        return
    
    # Call the main conversion function
    process_unified_to_waymo(
        input_dir=input_path,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        merged_filename=args.merged_filename
    )
    
    logger.info("Conversion complete!")


if __name__ == "__main__":
    main() 