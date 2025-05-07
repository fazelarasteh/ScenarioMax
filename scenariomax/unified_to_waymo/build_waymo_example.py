import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from scenariomax.logger_utils import get_logger
from scenariomax.raw_to_unified.converter.waymo.waymo_protos import scenario_pb2
from scenariomax.raw_to_unified.description import ScenarioDescription as SD
from scenariomax.raw_to_unified.type import ScenarioType


logger = get_logger(__name__)

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.FATAL)


def build_waymo_example(scenario_data: Dict[str, Any]) -> scenario_pb2.Scenario:
    """
    Converts a unified pickle format scenario to a Waymo Open Dataset scenario protobuf.
    
    Args:
        scenario_data: Scenario data in unified format
        
    Returns:
        A Waymo Open Dataset scenario protobuf
    """
    # Create a new Waymo scenario protobuf
    scenario = scenario_pb2.Scenario()
    
    # Set basic metadata
    scenario.scenario_id = scenario_data[SD.METADATA][SD.ID]
    
    # Set timestamps
    timestamps_seconds = scenario_data[SD.METADATA][SD.TIMESTEP]
    for ts in timestamps_seconds:
        scenario.timestamps_seconds.append(float(ts))
    
    # Set current time index (default to middle of scenario if not specified)
    if "current_time_index" in scenario_data[SD.METADATA]:
        scenario.current_time_index = scenario_data[SD.METADATA]["current_time_index"]
    else:
        scenario.current_time_index = len(timestamps_seconds) // 2
    
    # Convert tracks
    tracks = scenario_data[SD.TRACKS]
    sdc_id = scenario_data[SD.METADATA][SD.SDC_ID]
    sdc_track_index = None
    
    for i, (track_id, track_data) in enumerate(tracks.items()):
        waymo_track = convert_track(track_id, track_data)
        scenario.tracks.append(waymo_track)
        
        # Keep track of SDC index for later setting
        if track_id == sdc_id:
            sdc_track_index = i
    
    # Set SDC track index
    if sdc_track_index is not None:
        scenario.sdc_track_index = sdc_track_index
    else:
        logger.warning(f"SDC track index not found for scenario {scenario.scenario_id}")
    
    # Convert dynamic map states
    if SD.DYNAMIC_MAP_STATES in scenario_data:
        for ts_idx in range(len(timestamps_seconds)):
            dynamic_map_state = convert_dynamic_map_state(scenario_data[SD.DYNAMIC_MAP_STATES], ts_idx)
            scenario.dynamic_map_states.append(dynamic_map_state)
    
    # Convert map features
    if SD.MAP_FEATURES in scenario_data:
        for feature_id, feature_data in scenario_data[SD.MAP_FEATURES].items():
            map_feature = convert_map_feature(feature_id, feature_data)
            if map_feature is not None:
                scenario.map_features.append(map_feature)
    
    # Convert objects of interest
    if "objects_of_interest" in scenario_data[SD.METADATA]:
        for obj_id in scenario_data[SD.METADATA]["objects_of_interest"]:
            if obj_id in tracks:
                # Find the track index for this object ID
                for i, (track_id, _) in enumerate(tracks.items()):
                    if track_id == obj_id:
                        scenario.objects_of_interest.append(i)
                        break
    
    # Convert tracks to predict
    if "tracks_to_predict" in scenario_data[SD.METADATA]:
        for track_id, track_info in scenario_data[SD.METADATA]["tracks_to_predict"].items():
            if track_id in tracks:
                prediction = scenario_pb2.RequiredPrediction()
                
                # Find track index
                for i, (t_id, _) in enumerate(tracks.items()):
                    if t_id == track_id:
                        prediction.track_index = i
                        break
                
                # Set difficulty level
                difficulty = track_info.get("difficulty", 0)
                if difficulty == 1:
                    prediction.difficulty = scenario_pb2.RequiredPrediction.LEVEL_1
                elif difficulty == 2:
                    prediction.difficulty = scenario_pb2.RequiredPrediction.LEVEL_2
                else:
                    prediction.difficulty = scenario_pb2.RequiredPrediction.NONE
                
                scenario.tracks_to_predict.append(prediction)
    
    return scenario


def convert_track(track_id: str, track_data: Dict[str, Any]) -> scenario_pb2.Track:
    """Convert a unified format track to a Waymo track."""
    track = scenario_pb2.Track()
    
    # Set track ID (converting string to int if possible)
    try:
        track.id = int(track_id)
    except ValueError:
        logger.warning(f"Could not convert track ID {track_id} to integer, hashing instead")
        # Use a hash if the ID is not a simple integer
        track.id = hash(track_id) % (2**31 - 1)  # Ensure it fits in a 32-bit signed integer
    
    # Set object type
    track.object_type = convert_object_type(track_data["type"])
    
    # Convert states
    state_data = track_data["state"]
    track_length = track_data["metadata"]["track_length"]
    
    for i in range(track_length):
        if not state_data["valid"][i]:
            # Skip invalid states
            state = scenario_pb2.ObjectState()
            state.valid = False
            track.states.append(state)
            continue
        
        state = scenario_pb2.ObjectState()
        
        # Set position
        state.center_x = float(state_data["position"][i][0])
        state.center_y = float(state_data["position"][i][1])
        state.center_z = float(state_data["position"][i][2])
        
        # Set dimensions
        state.length = float(state_data["length"][i])
        state.width = float(state_data["width"][i])
        state.height = float(state_data["height"][i])
        
        # Set heading
        state.heading = float(state_data["heading"][i])
        
        # Set velocity
        state.velocity_x = float(state_data["velocity"][i][0])
        state.velocity_y = float(state_data["velocity"][i][1])
        
        # Set valid flag
        state.valid = True
        
        track.states.append(state)
    
    return track


def convert_object_type(type_str: str) -> scenario_pb2.Track.ObjectType:
    """Convert ScenarioType to Waymo ObjectType."""
    if type_str == ScenarioType.VEHICLE:
        return scenario_pb2.Track.TYPE_VEHICLE
    elif type_str == ScenarioType.PEDESTRIAN:
        return scenario_pb2.Track.TYPE_PEDESTRIAN
    elif type_str == ScenarioType.CYCLIST:
        return scenario_pb2.Track.TYPE_CYCLIST
    else:
        return scenario_pb2.Track.TYPE_OTHER


def convert_dynamic_map_state(dynamic_map_states: Dict[str, Any], time_index: int) -> scenario_pb2.DynamicMapState:
    """Convert unified format dynamic map state to Waymo dynamic map state."""
    dynamic_map_state = scenario_pb2.DynamicMapState()
    
    if "traffic_light_states" in dynamic_map_states:
        for light_id, light_states in dynamic_map_states["traffic_light_states"].items():
            if time_index < len(light_states["state"]):
                lane_state = scenario_pb2.TrafficSignalLaneState()
                
                try:
                    lane_state.lane = int(light_id)
                except ValueError:
                    logger.warning(f"Could not convert lane ID {light_id} to integer, hashing instead")
                    lane_state.lane = hash(light_id) % (2**31 - 1)
                
                # Convert traffic light state
                state_val = light_states["state"][time_index]
                lane_state.state = convert_traffic_light_state(state_val)
                
                dynamic_map_state.lane_states.append(lane_state)
    
    return dynamic_map_state


def convert_traffic_light_state(state_val: Any) -> scenario_pb2.TrafficSignalLaneState.State:
    """Convert unified format traffic light state to Waymo traffic light state."""
    # This mapping should be adjusted based on your specific state values
    if state_val == "UNKNOWN":
        return scenario_pb2.TrafficSignalLaneState.UNKNOWN
    elif state_val == "ARROW_STOP":
        return scenario_pb2.TrafficSignalLaneState.ARROW_STOP
    elif state_val == "ARROW_CAUTION":
        return scenario_pb2.TrafficSignalLaneState.ARROW_CAUTION
    elif state_val == "ARROW_GO":
        return scenario_pb2.TrafficSignalLaneState.ARROW_GO
    elif state_val == "STOP":
        return scenario_pb2.TrafficSignalLaneState.STOP
    elif state_val == "CAUTION":
        return scenario_pb2.TrafficSignalLaneState.CAUTION
    elif state_val == "GO":
        return scenario_pb2.TrafficSignalLaneState.GO
    elif state_val == "FLASHING_STOP":
        return scenario_pb2.TrafficSignalLaneState.FLASHING_STOP
    elif state_val == "FLASHING_CAUTION":
        return scenario_pb2.TrafficSignalLaneState.FLASHING_CAUTION
    else:
        return scenario_pb2.TrafficSignalLaneState.UNKNOWN


def convert_map_feature(feature_id: str, feature_data: Dict[str, Any]) -> Optional[scenario_pb2.MapFeature]:
    """Convert unified format map feature to Waymo map feature."""
    map_feature = scenario_pb2.MapFeature()
    
    try:
        map_feature.id = int(feature_id)
    except ValueError:
        logger.warning(f"Could not convert map feature ID {feature_id} to integer, hashing instead")
        map_feature.id = hash(feature_id) % (2**31 - 1)
    
    feature_type = feature_data.get("type", "")
    
    # Convert lane
    if "speed_limit_mph" in feature_data:
        lane = scenario_pb2.Lane()
        
        # Set speed limit
        lane.speed_limit_mph = float(feature_data.get("speed_limit_mph", 0))
        
        # Set lane type
        lane.type = convert_lane_type(feature_data.get("type", ""))
        
        # Set polyline
        if "polyline" in feature_data:
            for point in feature_data["polyline"]:
                map_point = scenario_pb2.MapPoint()
                map_point.x = float(point[0])
                map_point.y = float(point[1])
                if len(point) > 2:
                    map_point.z = float(point[2])
                lane.polyline.append(map_point)
        
        # Set interpolating flag
        lane.interpolating = feature_data.get("interpolating", False)
        
        # Set entry and exit lanes
        for entry_lane in feature_data.get("entry_lanes", []):
            lane.entry_lanes.append(str(entry_lane))
        
        for exit_lane in feature_data.get("exit_lanes", []):
            lane.exit_lanes.append(str(exit_lane))
        
        # Set boundaries
        for boundary in feature_data.get("left_boundaries", []):
            lane_boundary = convert_lane_boundary(boundary)
            lane.left_boundaries.append(lane_boundary)
        
        for boundary in feature_data.get("right_boundaries", []):
            lane_boundary = convert_lane_boundary(boundary)
            lane.right_boundaries.append(lane_boundary)
        
        # Set neighbors
        for neighbor in feature_data.get("left_neighbor", []):
            lane_neighbor = convert_lane_neighbor(neighbor)
            lane.left_neighbors.append(lane_neighbor)
        
        for neighbor in feature_data.get("right_neighbor", []):
            lane_neighbor = convert_lane_neighbor(neighbor)
            lane.right_neighbors.append(lane_neighbor)
        
        map_feature.lane.CopyFrom(lane)
        return map_feature
    
    # Convert road line
    elif feature_type in ["BROKEN_SINGLE_WHITE", "SOLID_SINGLE_WHITE", "SOLID_DOUBLE_WHITE", 
                         "BROKEN_SINGLE_YELLOW", "BROKEN_DOUBLE_YELLOW", "SOLID_SINGLE_YELLOW",
                         "SOLID_DOUBLE_YELLOW", "PASSING_DOUBLE_YELLOW"]:
        road_line = scenario_pb2.RoadLine()
        
        # Set road line type
        road_line.type = convert_road_line_type(feature_type)
        
        # Set polyline
        if "polyline" in feature_data:
            for point in feature_data["polyline"]:
                map_point = scenario_pb2.MapPoint()
                map_point.x = float(point[0])
                map_point.y = float(point[1])
                if len(point) > 2:
                    map_point.z = float(point[2])
                road_line.polyline.append(map_point)
        
        map_feature.road_line.CopyFrom(road_line)
        return map_feature
    
    # Convert road edge
    elif feature_type in ["BOUNDARY_UNKNOWN", "ROAD_EDGE_BOUNDARY", "ROAD_EDGE_MEDIAN"]:
        road_edge = scenario_pb2.RoadEdge()
        
        # Set road edge type
        road_edge.type = convert_road_edge_type(feature_type)
        
        # Set polyline
        if "polyline" in feature_data:
            for point in feature_data["polyline"]:
                map_point = scenario_pb2.MapPoint()
                map_point.x = float(point[0])
                map_point.y = float(point[1])
                if len(point) > 2:
                    map_point.z = float(point[2])
                road_edge.polyline.append(map_point)
        
        map_feature.road_edge.CopyFrom(road_edge)
        return map_feature
    
    # Convert stop sign
    elif feature_type == ScenarioType.STOP_SIGN:
        stop_sign = scenario_pb2.StopSign()
        
        # Set position
        if "position" in feature_data:
            position = scenario_pb2.MapPoint()
            position.x = float(feature_data["position"][0])
            position.y = float(feature_data["position"][1])
            position.z = float(feature_data["position"][2])
            stop_sign.position.CopyFrom(position)
        
        # Set lanes
        for lane_id in feature_data.get("lane", []):
            stop_sign.lane.append(str(lane_id))
        
        map_feature.stop_sign.CopyFrom(stop_sign)
        return map_feature
    
    # Convert crosswalk
    elif feature_type == ScenarioType.CROSSWALK:
        crosswalk = scenario_pb2.Crosswalk()
        
        # Set polyline for crosswalk boundary
        if "polygon" in feature_data:
            for point in feature_data["polygon"]:
                map_point = scenario_pb2.MapPoint()
                map_point.x = float(point[0])
                map_point.y = float(point[1])
                if len(point) > 2:
                    map_point.z = float(point[2])
                crosswalk.polygon.append(map_point)
        
        map_feature.crosswalk.CopyFrom(crosswalk)
        return map_feature
    
    # Skip unsupported feature types
    else:
        logger.debug(f"Skipping unsupported map feature type: {feature_type}")
        return None


def convert_lane_type(lane_type: str) -> scenario_pb2.Lane.LaneType:
    """Convert unified format lane type to Waymo lane type."""
    # Map your ScenarioType lane types to Waymo lane types
    if lane_type == "UNKNOWN":
        return scenario_pb2.Lane.LANE_TYPE_UNDEFINED
    elif lane_type == "FREEWAY":
        return scenario_pb2.Lane.FREEWAY
    elif lane_type == "SURFACE_STREET":
        return scenario_pb2.Lane.SURFACE_STREET
    elif lane_type == "BIKE_LANE":
        return scenario_pb2.Lane.BIKE_LANE
    else:
        return scenario_pb2.Lane.LANE_TYPE_UNDEFINED


def convert_road_line_type(line_type: str) -> scenario_pb2.RoadLine.RoadLineType:
    """Convert unified format road line type to Waymo road line type."""
    if line_type == "BROKEN_SINGLE_WHITE":
        return scenario_pb2.RoadLine.BROKEN_SINGLE_WHITE
    elif line_type == "SOLID_SINGLE_WHITE":
        return scenario_pb2.RoadLine.SOLID_SINGLE_WHITE
    elif line_type == "SOLID_DOUBLE_WHITE":
        return scenario_pb2.RoadLine.SOLID_DOUBLE_WHITE
    elif line_type == "BROKEN_SINGLE_YELLOW":
        return scenario_pb2.RoadLine.BROKEN_SINGLE_YELLOW
    elif line_type == "BROKEN_DOUBLE_YELLOW":
        return scenario_pb2.RoadLine.BROKEN_DOUBLE_YELLOW
    elif line_type == "SOLID_SINGLE_YELLOW":
        return scenario_pb2.RoadLine.SOLID_SINGLE_YELLOW
    elif line_type == "SOLID_DOUBLE_YELLOW":
        return scenario_pb2.RoadLine.SOLID_DOUBLE_YELLOW
    elif line_type == "PASSING_DOUBLE_YELLOW":
        return scenario_pb2.RoadLine.PASSING_DOUBLE_YELLOW
    else:
        return scenario_pb2.RoadLine.TYPE_UNDEFINED


def convert_road_edge_type(edge_type: str) -> scenario_pb2.RoadEdge.RoadEdgeType:
    """Convert unified format road edge type to Waymo road edge type."""
    if edge_type == "BOUNDARY_UNKNOWN":
        return scenario_pb2.RoadEdge.BOUNDARY_UNKNOWN
    elif edge_type == "ROAD_EDGE_BOUNDARY":
        return scenario_pb2.RoadEdge.ROAD_EDGE_BOUNDARY
    elif edge_type == "ROAD_EDGE_MEDIAN":
        return scenario_pb2.RoadEdge.ROAD_EDGE_MEDIAN
    else:
        return scenario_pb2.RoadEdge.TYPE_UNDEFINED


def convert_lane_boundary(boundary: Dict[str, Any]) -> scenario_pb2.LaneBoundary:
    """Convert unified format lane boundary to Waymo lane boundary."""
    lane_boundary = scenario_pb2.LaneBoundary()
    
    # Set boundary ID if available
    if "boundary_feature_id" in boundary:
        lane_boundary.boundary_feature_id = str(boundary["boundary_feature_id"])
    
    # Set boundary type
    if "boundary_type" in boundary:
        lane_boundary.boundary_type = convert_road_line_type(boundary["boundary_type"])
    
    return lane_boundary


def convert_lane_neighbor(neighbor: Dict[str, Any]) -> scenario_pb2.LaneNeighbor:
    """Convert unified format lane neighbor to Waymo lane neighbor."""
    lane_neighbor = scenario_pb2.LaneNeighbor()
    
    # Set neighbor lane ID
    if "feature_id" in neighbor:
        lane_neighbor.feature_id = str(neighbor["feature_id"])
    
    # Set self-to-neighbor edge
    if "self_start_index" in neighbor and "self_end_index" in neighbor:
        lane_neighbor.self_start_index = int(neighbor["self_start_index"])
        lane_neighbor.self_end_index = int(neighbor["self_end_index"])
    
    # Set neighbor-to-self edge
    if "neighbor_start_index" in neighbor and "neighbor_end_index" in neighbor:
        lane_neighbor.neighbor_start_index = int(neighbor["neighbor_start_index"])
        lane_neighbor.neighbor_end_index = int(neighbor["neighbor_end_index"])
    
    return lane_neighbor


def write_to_tfrecord(scenario: scenario_pb2.Scenario, output_path: str) -> None:
    """
    Write a Waymo scenario protobuf to a TFRecord file.
    
    Args:
        scenario: Waymo scenario protobuf
        output_path: Path to output TFRecord file
    """
    with tf.io.TFRecordWriter(output_path) as writer:
        writer.write(scenario.SerializeToString())


def pickle_to_waymo(input_path: str, output_path: str) -> None:
    """
    Convert a pickle file with unified format to Waymo TFRecord format.
    
    Args:
        input_path: Path to input pickle file
        output_path: Path to output TFRecord file
    """
    # Load pickle file
    with open(input_path, "rb") as f:
        scenario_data = pickle.load(f)
    
    # Convert to Waymo format
    waymo_scenario = build_waymo_example(scenario_data)
    
    # Write to TFRecord file
    write_to_tfrecord(waymo_scenario, output_path)
    
    logger.info(f"Converted {input_path} to {output_path}") 