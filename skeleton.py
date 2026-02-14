"""
Real-time Human Pose Detection and Floor Position Mapping

This module captures video from a camera or file, detects human poses using YOLOv8,
and maps the detected person's floor position to real-world coordinates using 
homography transformation. It sends position data via OSC for integration with
interactive systems.

Usage examples:
    # Live camera feed (camera index 2)
    python3 skeleton.py --cam 2 --model yolov8s-pose.pt
    
    # Video file input
    python3 skeleton.py --video path/to/video.mp4 --model yolov8s-pose.pt

Requirements:
    - floor_homography.npy: Pre-computed homography matrix for pixel-to-floor mapping
    - YOLOv8 pose model file (e.g., yolov8s-pose.pt)
"""

# ================================
# CONFIGURATION VARIABLES
# ================================

# Camera configuration - Change this value to use different camera
DEFAULT_CAMERA_INDEX = 0  # Set to your preferred camera index (0, 1, 2, etc.)

# ================================
# Other configuration options (modify as needed)
# ================================

# Model configuration
DEFAULT_MODEL_PATH = "yolov8s-pose.pt"
DEFAULT_CONFIDENCE = 0.3
DEFAULT_KEYPOINT_CONFIDENCE = 0.5
DEFAULT_INFERENCE_SIZE = 960

# OSC configuration  
DEFAULT_OSC_HOST = "127.0.0.1"
DEFAULT_OSC_PORT = 8000

# Tracking configuration
DEFAULT_TRACKING_DISTANCE = 2.0
DEFAULT_TRACKING_TIMEOUT = 30

# Visualization configuration
DEFAULT_FLIP_CAMERA = False  # Set to True to flip camera horizontally
DEFAULT_SHOW_BOUNDARY = True

import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional, Tuple, List, NamedTuple, Protocol
from pythonosc.udp_client import SimpleUDPClient
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
import math
from collections import defaultdict
# Add recording imports
import os
import json
from datetime import datetime



# ================================
# POSE MODEL CONFIGURATION
# ================================

# COCO keypoint indices and names for YOLOv8 pose estimation (17 keypoints total)
# Optimized for body tracking - face keypoints removed from processing
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",  # Still present in model output
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Keypoint indices for easier reference - optimized for body tracking
class KeypointIndices:
    """Semantic mapping of body part names to their COCO keypoint indices"""
    # Face keypoints (not used for tracking but still present in model output)
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    
    # Body keypoints (used for tracking and visualization)
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15  # Primary keypoints for floor position
    RIGHT_ANKLE = 16  # Primary keypoints for floor position

# Skeleton connections for visualization - optimized for body-only tracking
# Removed face connections for cleaner visualization
SKELETON_BONE_CONNECTIONS = [
    # Arm connections
    (KeypointIndices.LEFT_SHOULDER, KeypointIndices.RIGHT_SHOULDER),
    (KeypointIndices.LEFT_SHOULDER, KeypointIndices.LEFT_ELBOW),
    (KeypointIndices.LEFT_ELBOW, KeypointIndices.LEFT_WRIST),
    (KeypointIndices.RIGHT_SHOULDER, KeypointIndices.RIGHT_ELBOW),
    (KeypointIndices.RIGHT_ELBOW, KeypointIndices.RIGHT_WRIST),
    
    # Torso connections
    (KeypointIndices.LEFT_SHOULDER, KeypointIndices.LEFT_HIP),
    (KeypointIndices.RIGHT_SHOULDER, KeypointIndices.RIGHT_HIP),
    (KeypointIndices.LEFT_HIP, KeypointIndices.RIGHT_HIP),
    
    # Leg connections
    (KeypointIndices.LEFT_HIP, KeypointIndices.LEFT_KNEE),
    (KeypointIndices.LEFT_KNEE, KeypointIndices.LEFT_ANKLE),
    (KeypointIndices.RIGHT_HIP, KeypointIndices.RIGHT_KNEE),
    (KeypointIndices.RIGHT_KNEE, KeypointIndices.RIGHT_ANKLE)
]

# ================================
# VISUALIZATION COLORS (BGR format for OpenCV)
# ================================
class Colors:
    """Color constants for visualization elements"""
    BOUNDING_BOX = (0, 255, 0)      # Green
    SKELETON_BONES = (0, 255, 255)   # Yellow
    KEYPOINTS = (255, 0, 0)          # Blue
    FLOOR_POSITION = (0, 0, 255)     # Red
    TEXT = (0, 0, 255)               # Red


# ================================
# DATA STRUCTURES
# ================================

@dataclass(frozen=True)
class FloorPosition:
    """Represents a person's floor position with coordinate transformation"""
    pixel_x: float
    pixel_y: float
    floor_x: float
    floor_y: float
    person_id: int
    confidence: float


@dataclass(frozen=True)
class SkeletonBone:
    """Represents a skeleton bone connection between two keypoints"""
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    confidence: float


@dataclass(frozen=True)
class Keypoint:
    """Represents a single detected keypoint"""
    x: int
    y: int
    confidence: float
    keypoint_type: int


@dataclass(frozen=True)
class BoundingBox:
    """Represents a person's bounding box"""
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class PersonDetection:
    """Represents a complete person detection with all associated data"""
    person_id: int
    keypoints: Tuple[Keypoint, ...]
    skeleton_bones: Tuple[SkeletonBone, ...]
    bounding_box: Optional[BoundingBox]
    floor_position: Optional[FloorPosition]
    hands_raised: bool  # New field for hands tracking


@dataclass(frozen=True)
class FrameAnalysis:
    """Represents complete frame analysis results"""
    frame_number: int
    detected_people: Tuple[PersonDetection, ...]
    processing_time_ms: float


@dataclass
class TrackedPerson:
    """Mutable tracking state for a person across frames"""
    person_id: int
    last_floor_position: Optional[FloorPosition]
    frames_since_detection: int
    total_detections: int
    first_seen_frame: int
    last_seen_frame: int
    
    def update_detection(self, floor_position: Optional[FloorPosition], frame_number: int) -> None:
        """Update tracking info when person is detected in a frame"""
        self.last_floor_position = floor_position
        self.frames_since_detection = 0
        self.total_detections += 1
        self.last_seen_frame = frame_number
    
    def increment_missed_frames(self) -> None:
        """Increment counter when person is not detected in a frame"""
        self.frames_since_detection += 1


# ================================
# I/O PROTOCOLS
# ================================

class OutputHandler(Protocol):
    """Protocol for handling output operations"""
    
    def log_detection_info(self, frame_analysis: FrameAnalysis) -> None:
        """Log detection information to console or other output"""
        ...
    
    def send_positions_frame(self, frame_analysis: FrameAnalysis) -> None:
        """Send all positions for the current frame"""
        ...


class Visualizer(Protocol):
    """Protocol for handling visualization operations"""
    
    def create_visualization_frame(
        self, 
        original_frame: np.ndarray, 
        frame_analysis: FrameAnalysis
    ) -> np.ndarray:
        """Create visualization frame without modifying the original"""
        ...

def transform_pixel_to_floor_coordinates(
    pixel_x: float, 
    pixel_y: float, 
    homography_matrix: np.ndarray
) -> Tuple[float, float]:
    """
    Transform a pixel coordinate to real-world floor coordinates using homography.
    
    This function applies a perspective transformation (homography) to convert
    image pixel coordinates to real-world floor coordinates in meters.
    
    Args:
        pixel_x: X coordinate in the image (pixels)
        pixel_y: Y coordinate in the image (pixels)
        homography_matrix: 3x3 homography transformation matrix
        
    Returns:
        Tuple of (floor_x, floor_y) coordinates in meters
        
    Note:
        The homography matrix must be pre-computed using corresponding points
        between the image plane and the real-world floor plane.
    """
    # OpenCV requires points in specific format: [[[x, y]]] for perspectiveTransform
    pixel_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    
    # Apply perspective transformation to get real-world coordinates
    floor_coordinates = cv2.perspectiveTransform(pixel_point, homography_matrix)[0, 0]
    
    return float(floor_coordinates[0]), float(floor_coordinates[1])


def is_point_in_tracked_area(
    floor_x: float, 
    floor_y: float, 
    boundary_margin: float = 0.1
) -> bool:
    """
    Check if a floor position is within the tracked area boundaries.
    
    Args:
        floor_x: X coordinate in meters
        floor_y: Y coordinate in meters  
        boundary_margin: Margin in meters to allow slight overlap
        
    Returns:
        True if the point is inside the tracked area, False otherwise
    """
    # Based on compute_homography.py: 8 columns × 7 rows, 50cm spacing
    # X = 0-3.5m, Y = 0-3.0m with small margin for tolerance
    return ((-boundary_margin <= floor_x <= 3.5 + boundary_margin) and 
            (-boundary_margin <= floor_y <= 3.0 + boundary_margin))


def create_keypoints_from_detection(
    person_keypoints: np.ndarray, 
    keypoint_confidence_threshold: float
) -> Tuple[Keypoint, ...]:
    """
    Create keypoint objects from raw detection data - optimized for body tracking.
    
    Args:
        person_keypoints: Raw keypoint array (17, 3) from YOLO
        keypoint_confidence_threshold: Minimum confidence for valid keypoints
        
    Returns:
        Tuple of Keypoint objects (excluding low-confidence face keypoints)
    """
    keypoints = []
    for keypoint_idx, (x, y, confidence) in enumerate(person_keypoints):
        # Skip face keypoints if they have low confidence (optimization)
        if keypoint_idx <= KeypointIndices.RIGHT_EAR and confidence < keypoint_confidence_threshold * 1.5:
            continue
            
        if confidence > keypoint_confidence_threshold:
            keypoints.append(Keypoint(
                x=int(x),
                y=int(y),
                confidence=float(confidence),
                keypoint_type=keypoint_idx
            ))
    return tuple(keypoints)


def create_skeleton_bones_from_keypoints(
    person_keypoints: np.ndarray,
    keypoint_confidence_threshold: float
) -> Tuple[SkeletonBone, ...]:
    """
    Create skeleton bone objects from keypoint data.
    
    Args:
        person_keypoints: Raw keypoint array (17, 3) from YOLO
        keypoint_confidence_threshold: Minimum confidence for valid bones
        
    Returns:
        Tuple of SkeletonBone objects
    """
    bones = []
    for start_idx, end_idx in SKELETON_BONE_CONNECTIONS:
        start_point = person_keypoints[start_idx]
        end_point = person_keypoints[end_idx]
        
        if (start_point[2] > keypoint_confidence_threshold and 
            end_point[2] > keypoint_confidence_threshold):
            
            bones.append(SkeletonBone(
                start_x=int(start_point[0]),
                start_y=int(start_point[1]),
                end_x=int(end_point[0]),
                end_y=int(end_point[1]),
                confidence=min(float(start_point[2]), float(end_point[2]))
            ))
    return tuple(bones)


def create_floor_position_from_keypoints(
    person_keypoints: np.ndarray,
    person_id: int,
    keypoint_confidence_threshold: float,
    homography_matrix: np.ndarray
) -> Optional[FloorPosition]:
    """
    Create floor position from keypoint data - only for people in tracked area.
    
    Args:
        person_keypoints: Raw keypoint array for one person
        person_id: Unique identifier for this person
        keypoint_confidence_threshold: Minimum confidence threshold
        homography_matrix: Homography matrix for coordinate transformation
        
    Returns:
        FloorPosition object or None if no reliable position found or outside tracked area
    """
    pixel_x, pixel_y = determine_person_floor_position(
        person_keypoints, keypoint_confidence_threshold
    )
    
    if pixel_x is not None and pixel_y is not None:
        floor_x, floor_y = transform_pixel_to_floor_coordinates(
            pixel_x, pixel_y, homography_matrix
        )
        
        # Check if the person is within the tracked area boundaries
        if not is_point_in_tracked_area(floor_x, floor_y):
            return None  # Person is outside tracked area, don't create floor position
        
        # Calculate confidence as average of ankle confidences
        left_ankle = person_keypoints[KeypointIndices.LEFT_ANKLE]
        right_ankle = person_keypoints[KeypointIndices.RIGHT_ANKLE]
        
        confidence = max(left_ankle[2], right_ankle[2])
        if left_ankle[2] > keypoint_confidence_threshold and right_ankle[2] > keypoint_confidence_threshold:
            confidence = (left_ankle[2] + right_ankle[2]) / 2.0
        
        return FloorPosition(
            pixel_x=float(pixel_x),
            pixel_y=float(pixel_y),
            floor_x=float(floor_x),
            floor_y=float(floor_y),
            person_id=person_id,
            confidence=float(confidence)
        )
    
    return None


# ================================
# I/O IMPLEMENTATIONS
# ================================

class ConsoleOutputHandler:
    """Handles console output for detection results"""
    
    def __init__(self, show_tracking_info: bool = False):
        self.show_tracking_info = show_tracking_info
    
    def log_detection_info(self, frame_analysis: FrameAnalysis) -> None:
        """Log detection information to console - only for tracked people"""
        # Count tracked vs total people
        tracked_people = [p for p in frame_analysis.detected_people if p.floor_position is not None]
        total_people = len(frame_analysis.detected_people)
        
        print(f"\n--- Frame {frame_analysis.frame_number}: {len(tracked_people)} tracked / {total_people} total person(s) ---")
        print(f"Processing time: {frame_analysis.processing_time_ms:.1f}ms")
        
        # Only log tracked people (those inside the boundary)
        for person in tracked_people:
            print(f"\nPerson ID {person.person_id} (TRACKED):")
            if person.floor_position:
                fp = person.floor_position
                print(f"  FLOOR POSITION (meters): X={fp.floor_x:.2f}, Y={fp.floor_y:.2f} (confidence: {fp.confidence:.2f})")
                print(f"  HANDS RAISED: {person.hands_raised}")
        
        # Briefly mention untracked people
        untracked_count = total_people - len(tracked_people)
        if untracked_count > 0:
            print(f"\n{untracked_count} person(s) visible but outside tracked area")
    
    def send_positions_frame(self, frame_analysis: FrameAnalysis) -> None:
        """Console handler doesn't send position frames"""
        pass


class OSCOutputHandler:
    """Handles OSC communication for position data"""
    
    def __init__(self, osc_host: str = "127.0.0.1", osc_port: int = 8000):  # Changed from 9000 to 8000
        self.osc_client = SimpleUDPClient(osc_host, osc_port)
        self.last_positions: dict[int, Tuple[float, float]] = {}
        print(f"OSC client initialized - sending to {osc_host}:{osc_port}")
    
    def log_detection_info(self, frame_analysis: FrameAnalysis) -> None:
        """OSC handler doesn't log to console"""
        pass
    
    def send_positions_frame(self, frame_analysis: FrameAnalysis) -> None:
        """Send all current positions with hands data if there are changes from last frame"""
        current_positions = {}
        
        # Extract current positions and hands data
        for person in frame_analysis.detected_people:
            if person.floor_position:
                current_positions[person.person_id] = (
                    person.floor_position.floor_x,
                    person.floor_position.floor_y,
                    person.hands_raised  # Add hands data
                )
        
        # Always send if we have people (for debugging)
        if current_positions:
            # Send OSC message with all current positions and hands data
            message_data = []
            for person_id, (x, y, hands_raised) in current_positions.items():
                message_data.extend([person_id, x, y, int(hands_raised)])  # Convert bool to int
            
            try:
                self.osc_client.send_message("/people/positions", message_data)
                hands_summary = [f"ID{pid}:{'🙌' if hr else '👇'}" for pid, (x, y, hr) in current_positions.items()]
                print(f"🔊 Sent OSC /people/positions with {len(current_positions)} people: {message_data} [{', '.join(hands_summary)}]")
            except Exception as e:
                print(f"❌ Failed to send OSC message: {e}")
            
            self.last_positions = current_positions.copy()
        elif self.last_positions:
            # Send empty message when no people detected (to clear previous positions)
            try:
                self.osc_client.send_message("/people/positions", [])
                print(f"🔊 Sent empty OSC /people/positions (no people detected)")
            except Exception as e:
                print(f"❌ Failed to send empty OSC message: {e}")
            
            self.last_positions = {}

class CombinedOutputHandler:
    """Combines multiple output handlers for comprehensive I/O"""
    
    def __init__(self, handlers: List[OutputHandler]):
        self.handlers = handlers
    
    def log_detection_info(self, frame_analysis: FrameAnalysis) -> None:
        for handler in self.handlers:
            handler.log_detection_info(frame_analysis)
    
    def send_positions_frame(self, frame_analysis: FrameAnalysis) -> None:
        for handler in self.handlers:
            handler.send_positions_frame(frame_analysis)


class OpenCVVisualizer:
    """Handles OpenCV-based visualization rendering - optimized for body tracking"""
    
    def create_visualization_frame(
        self, 
        original_frame: np.ndarray, 
        frame_analysis: FrameAnalysis
    ) -> np.ndarray:
        """Create visualization frame without modifying the original"""
        # Create a copy for visualization
        visualization_frame = original_frame.copy()
        
        # Draw floor tracking boundary first (so it appears behind people)
        self._draw_floor_boundary(visualization_frame)
        
        # Draw all detected people (both tracked and untracked)
        for person in frame_analysis.detected_people:
            self._draw_person_on_frame(visualization_frame, person)
        
        return visualization_frame
    
    def _draw_floor_boundary(self, visualization_frame: np.ndarray) -> None:
        """Draw the floor tracking boundary based on homography calibration area"""
        # Define floor boundary in world coordinates (meters)
        # Based on your compute_homography.py: 8 columns × 7 rows, 50cm spacing
        # X = 0-3.5m (7 * 0.5), Y = 0-3.0m (6 * 0.5) 
        floor_boundary_world = np.array([
            [0.0, 0.0],    # Top-left corner
            [3.5, 0.0],    # Top-right corner  
            [3.5, 3.0],    # Bottom-right corner
            [0.0, 3.0],    # Bottom-left corner
            [0.0, 0.0]     # Close the boundary
        ], dtype=np.float32)
        
        try:
            # Transform world coordinates back to pixel coordinates
            # We need the inverse homography for this
            homography_matrix = self._get_homography_matrix()
            if homography_matrix is not None:
                # Calculate inverse homography (world -> pixel)
                inv_homography = np.linalg.inv(homography_matrix)
                
                # Transform boundary points to pixel coordinates
                boundary_pixels = []
                for world_point in floor_boundary_world:
                    # Apply inverse perspective transformation
                    world_point_homogeneous = np.array([[[world_point[0], world_point[1]]]], dtype=np.float32)
                    pixel_point = cv2.perspectiveTransform(world_point_homogeneous, inv_homography)[0, 0]
                    boundary_pixels.append([int(pixel_point[0]), int(pixel_point[1])])
                
                boundary_pixels = np.array(boundary_pixels, dtype=np.int32)
                
                # Draw the boundary as a polygon outline
                cv2.polylines(
                    visualization_frame, 
                    [boundary_pixels], 
                    isClosed=False,  # We already closed it in the array
                    color=(0, 255, 0),  # Green color
                    thickness=3,
                    lineType=cv2.LINE_AA
                )
                
                # Draw corner markers
                for i, corner in enumerate(boundary_pixels[:-1]):  # Skip the last point (duplicate)
                    cv2.circle(visualization_frame, tuple(corner), 8, (0, 255, 0), -1)
                    # Label corners with world coordinates
                    world_coord = floor_boundary_world[i]
                    label = f"({world_coord[0]:.1f},{world_coord[1]:.1f})"
                    cv2.putText(
                        visualization_frame,
                        label,
                        (corner[0] + 10, corner[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA
                    )
                
                
        except Exception as e:
            # If homography visualization fails, just continue without it
            cv2.putText(
                visualization_frame,
                "Floor boundary visualization unavailable",
                (10, visualization_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )
    
    def _get_homography_matrix(self) -> Optional[np.ndarray]:
        """Get homography matrix for boundary visualization"""
        try:
            return np.load("floor_homography.npy")
        except Exception:
            return None
    
    def _draw_person_on_frame(
        self, 
        visualization_frame: np.ndarray, 
        person: PersonDetection
    ) -> None:
        """Draw a single person's skeleton and information on the frame - body-focused"""
        frame_height, frame_width = visualization_frame.shape[:2]
        
        # Determine if person is being tracked (has floor position) or just visible
        is_tracked = person.floor_position is not None
        
        # Use different colors for tracked vs untracked people
        if is_tracked:
            skeleton_color = Colors.SKELETON_BONES  # Yellow for tracked
            keypoint_color = Colors.KEYPOINTS       # Blue for tracked
            id_bg_color = (0, 0, 0)                # Black background for tracked
            id_text_color = (255, 255, 255)        # White text for tracked
        else:
            skeleton_color = (128, 128, 128)        # Gray for untracked
            keypoint_color = (100, 100, 100)       # Dark gray for untracked
            id_bg_color = (50, 50, 50)             # Dark gray background for untracked
            id_text_color = (150, 150, 150)        # Light gray text for untracked
        
        # Calculate ID text position (prioritize shoulders over face for body tracking)
        id_x, id_y = self._calculate_id_position(person, frame_width, frame_height)
        
        # Draw person ID with background for visibility
        if is_tracked:
            id_text = f"ID {person.person_id}"
        else:
            id_text = f"-- {person.person_id}"  # Different format for untracked
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(id_text, font, font_scale, thickness)
        
        # Draw background rectangle for text
        bg_x1 = max(0, id_x - 5)
        bg_y1 = max(0, id_y - text_height - 10)
        bg_x2 = min(frame_width, id_x + text_width + 5)
        bg_y2 = id_y + 5
        
        cv2.rectangle(visualization_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), id_bg_color, -1)
        
        # Draw ID text
        cv2.putText(
            visualization_frame, 
            id_text, 
            (id_x, id_y), 
            font, 
            font_scale, 
            id_text_color,
            thickness
        )
        
        # Only draw hands status for tracked people
        if is_tracked:
            hands_status = "1" if person.hands_raised else "0"
            hands_color = (0, 255, 0) if person.hands_raised else (0, 0, 255)  # Green for raised, Red for down
            hands_y = id_y + 30
            
            # Background for hands status
            hands_text_size = cv2.getTextSize(f"Hands: {hands_status}", font, 0.7, 2)[0]
            cv2.rectangle(
                visualization_frame, 
                (id_x - 5, hands_y - hands_text_size[1] - 5),
                (id_x + hands_text_size[0] + 5, hands_y + 5),
                (0, 0, 0), -1
            )
            
            cv2.putText(
                visualization_frame,
                f"Hands: {hands_status}",
                (id_x, hands_y),
                font,
                0.7,
                hands_color,
                2
            )
        else:
            # Show "OUTSIDE" status for untracked people
            status_y = id_y + 30
            cv2.putText(
                visualization_frame,
                "OUTSIDE",
                (id_x, status_y),
                font,
                0.6,
                (0, 100, 200),  # Orange color
                2
            )
        
        # Draw bounding box if available (same for all people)
        if person.bounding_box:
            bbox = person.bounding_box
            bbox_color = Colors.BOUNDING_BOX if is_tracked else (80, 80, 80)  # Dimmer for untracked
            cv2.rectangle(
                visualization_frame,
                (bbox.x1, bbox.y1),
                (bbox.x2, bbox.y2),
                bbox_color,
                2
            )
        
        # Draw skeleton bones with appropriate color
        for bone in person.skeleton_bones:
            cv2.line(
                visualization_frame,
                (bone.start_x, bone.start_y),
                (bone.end_x, bone.end_y),
                skeleton_color,
                2
            )
        
        # Draw keypoints with different sizes based on importance and tracking status
        for keypoint in person.keypoints:
            # Emphasize ankle keypoints (used for floor position) only for tracked people
            if keypoint.keypoint_type in [KeypointIndices.LEFT_ANKLE, KeypointIndices.RIGHT_ANKLE]:
                if is_tracked:
                    radius = 6
                    color = Colors.FLOOR_POSITION  # Red for ankle keypoints
                else:
                    radius = 4
                    color = (100, 100, 100)  # Gray for untracked ankles
            # De-emphasize face keypoints
            elif keypoint.keypoint_type <= KeypointIndices.RIGHT_EAR:
                radius = 2
                color = (128, 128, 128) if is_tracked else (80, 80, 80)
            else:
                radius = 4
                color = keypoint_color
                
            cv2.circle(
                visualization_frame,
                (keypoint.x, keypoint.y),
                radius,
                color,
                -1
            )
        
        # Only draw floor position indicator for tracked people
        if person.floor_position:
            fp = person.floor_position
            cv2.circle(
                visualization_frame,
                (int(fp.pixel_x), int(fp.pixel_y)),
                10,  # Larger circle for floor position
                Colors.FLOOR_POSITION,
                -1
            )
            
            # Add text label with coordinates
            position_text = f"{fp.floor_x:.2f},{fp.floor_y:.2f}m"
            cv2.putText(
                visualization_frame,
                position_text,
                (int(fp.pixel_x) + 15, int(fp.pixel_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                Colors.TEXT,
                2
            )

    def _calculate_id_position(
        self, 
        person: PersonDetection, 
        frame_width: int, 
        frame_height: int
    ) -> Tuple[int, int]:
        """Calculate the best position for drawing person ID - body-focused positioning"""
        
        # Priority 1: Use bounding box top-left if available
        if person.bounding_box:
            id_x = max(10, min(frame_width - 100, person.bounding_box.x1))
            id_y = max(30, person.bounding_box.y1 + 20)
            return id_x, id_y
        
        # Priority 2: Use shoulder keypoints (more reliable than face for body tracking)
        shoulder_keypoints = [kp for kp in person.keypoints 
                            if kp.keypoint_type in [KeypointIndices.LEFT_SHOULDER, KeypointIndices.RIGHT_SHOULDER]]
        
        if shoulder_keypoints:
            avg_shoulder_x = sum(kp.x for kp in shoulder_keypoints) / len(shoulder_keypoints)
            min_shoulder_y = min(kp.y for kp in shoulder_keypoints)
            
            id_x = max(10, min(frame_width - 100, int(avg_shoulder_x)))
            id_y = max(30, min_shoulder_y - 10)
            return id_x, id_y
        
        # Priority 3: Use any available keypoint
        if person.keypoints:
            min_y = min(kp.y for kp in person.keypoints)
            avg_x = sum(kp.x for kp in person.keypoints) / len(person.keypoints)
            
            id_x = max(10, min(frame_width - 100, int(avg_x)))
            id_y = max(30, min_y - 10)
            return id_x, id_y
        
        # Priority 4: Use floor position if available
        if person.floor_position:
            id_x = max(10, min(frame_width - 100, int(person.floor_position.pixel_x)))
            id_y = max(30, int(person.floor_position.pixel_y) - 50)
            return id_x, id_y
        
        # Fallback: Use person ID to distribute across top of frame
        id_x = (person.person_id * 120) % (frame_width - 100) + 10
        id_y = 30
        return id_x, id_y


class PersonTracker:
    """
    Manages stable person IDs across frames using floor position tracking.
    
    This class maintains a registry of people and matches new detections to
    existing tracked people based on floor position proximity, ensuring
    stable IDs for tracking applications.
    """
    
    def __init__(
        self,
        max_distance_threshold: float = 1.0,  # Reduced from 2.0 for better accuracy
        max_frames_missing: int = 60,  # Increased from 30 to be more patient
        min_detections_for_stability: int = 5,  # Increased from 3 for more stability
        fps: float = 30.0  # Expected frames per second for timing calculations
    ):
        """
        Initialize person tracker with configurable parameters.
        
        Args:
            max_distance_threshold: Maximum distance (meters) to consider a match
            max_frames_missing: Frames without detection before removing a person
            min_detections_for_stability: Minimum detections for a stable track
            fps: Expected frames per second for timing calculations
        """
        self.max_distance_threshold = max_distance_threshold
        self.max_frames_missing = max_frames_missing
        self.min_detections_for_stability = min_detections_for_stability
        self.fps = fps
        
        self.tracked_people: dict[int, TrackedPerson] = {}
        self.next_person_id = 0
        self.current_frame = 0
        
        # Track when we last had any people detected
        self.frames_with_no_people = 0
        self.id_reset_threshold = int(1.0 * fps)  # 1 second worth of frames
    
    def update_frame(self, detected_people: List[PersonDetection], frame_number: int) -> Tuple[List[PersonDetection], List[int]]:
        """
        Update tracking for a new frame and return people with stable IDs.
        
        Args:
            detected_people: List of detections from current frame
            frame_number: Current frame number
            
        Returns:
            Tuple of:
            - List of PersonDetection objects with stable person_ids
            - List of person IDs that left the frame
        """
        self.current_frame = frame_number
        
        # Extract detections with floor positions for tracking
        detections_with_positions = [
            (i, person) for i, person in enumerate(detected_people)
            if person.floor_position is not None
        ]
        
        # Check if we should reset IDs based on people INSIDE the boundary only
        people_in_boundary = len(detections_with_positions)
        if people_in_boundary == 0:
            self.frames_with_no_people += 1
            if self.frames_with_no_people >= self.id_reset_threshold:
                # Reset ID system when no people IN BOUNDARY for more than 1 second
                if self.tracked_people:  # Only print if we had tracked people
                    print(f"\n🔄 No people in boundary for {self.frames_with_no_people} frames (>{self.id_reset_threshold}). Resetting ID counter.")
                self._reset_id_system()
        else:
            self.frames_with_no_people = 0  # Reset counter when people are detected IN BOUNDARY
        
        # Match detections to existing tracked people
        matched_pairs, unmatched_detections, unmatched_tracked = self._match_detections_to_tracks(
            detections_with_positions
        )
        
        # Update existing tracks with matches
        updated_people = list(detected_people)  # Start with original detections
        
        for detection_idx, track_id in matched_pairs:
            original_person = detected_people[detection_idx]
            tracked_person = self.tracked_people[track_id]
            
            # Update tracking state
            tracked_person.update_detection(original_person.floor_position, frame_number)
            
            # Create new PersonDetection with stable ID
            updated_person = self._create_person_with_stable_id(
                original_person, track_id
            )
            updated_people[detection_idx] = updated_person
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            original_person = detected_people[detection_idx]
            new_track_id = self._create_new_track(original_person, frame_number)
            
            # Create PersonDetection with new stable ID
            updated_person = self._create_person_with_stable_id(
                original_person, new_track_id
            )
            updated_people[detection_idx] = updated_person
        
        # Update unmatched tracked people (increment missed frames)
        for track_id in unmatched_tracked:
            self.tracked_people[track_id].increment_missed_frames()
        
        # Remove lost tracks and get their IDs
        removed_person_ids = self._remove_lost_tracks()
        
        return updated_people, removed_person_ids
    
    def _reset_id_system(self) -> None:
        """Reset the ID system when no people have been detected for a while"""
        self.tracked_people.clear()
        self.next_person_id = 0
        self.frames_with_no_people = 0
    
    def _match_detections_to_tracks(
        self, 
        detections_with_positions: List[Tuple[int, PersonDetection]]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match current detections to existing tracks using floor position distance.
        Enhanced with prediction for better matching.
        """
        if not self.tracked_people or not detections_with_positions:
            unmatched_detections = [idx for idx, _ in detections_with_positions]
            unmatched_tracked = list(self.tracked_people.keys())
            return [], unmatched_detections, unmatched_tracked
        
        # Calculate distance matrix with predicted positions
        distance_matrix = []
        track_ids = list(self.tracked_people.keys())
        
        for detection_idx, person in detections_with_positions:
            detection_distances = []
            for track_id in track_ids:
                tracked_person = self.tracked_people[track_id]
                if (tracked_person.last_floor_position and person.floor_position):
                    distance = self._calculate_floor_distance(
                        person.floor_position, tracked_person.last_floor_position
                    )
                    
                    # Apply stricter matching for established tracks
                    if tracked_person.total_detections >= self.min_detections_for_stability:
                        distance *= 0.8  # Favor established tracks
                    
                    detection_distances.append(distance)
                else:
                    detection_distances.append(float('inf'))
            distance_matrix.append(detection_distances)
        
        # Use Hungarian-like algorithm for better matching
        matched_pairs = []
        unmatched_detections = set(range(len(detections_with_positions)))
        unmatched_tracked = set(range(len(track_ids)))
        
        # Sort potential matches by distance
        all_matches = []
        for det_idx in range(len(detections_with_positions)):
            for track_idx in range(len(track_ids)):
                distance = distance_matrix[det_idx][track_idx]
                if distance <= self.max_distance_threshold:
                    all_matches.append((distance, det_idx, track_idx))
        
        # Sort by distance (best matches first)
        all_matches.sort(key=lambda x: x[0])
        
        # Assign matches greedily from best to worst
        for distance, det_idx, track_idx in all_matches:
            if det_idx in unmatched_detections and track_idx in unmatched_tracked:
                # Record match using original detection index and track ID
                original_detection_idx = detections_with_positions[det_idx][0]
                track_id = track_ids[track_idx]
                matched_pairs.append((original_detection_idx, track_id))
                
                unmatched_detections.remove(det_idx)
                unmatched_tracked.remove(track_idx)
        
        # Convert remaining unmatched indices back to original detection indices
        unmatched_detection_indices = [
            detections_with_positions[i][0] for i in unmatched_detections
        ]
        unmatched_track_ids = [track_ids[i] for i in unmatched_tracked]
        
        return matched_pairs, unmatched_detection_indices, unmatched_track_ids
    
    def _calculate_floor_distance(
        self, 
        pos1: FloorPosition, 
        pos2: FloorPosition
    ) -> float:
        """Calculate Euclidean distance between two floor positions in meters"""
        dx = pos1.floor_x - pos2.floor_x
        dy = pos1.floor_y - pos2.floor_y
        return math.sqrt(dx * dx + dy * dy)
    
    def _create_new_track(self, person: PersonDetection, frame_number: int) -> int:
        """Create a new track for an unmatched detection"""
        new_id = self.next_person_id
        self.next_person_id += 1
        
        self.tracked_people[new_id] = TrackedPerson(
            person_id=new_id,
            last_floor_position=person.floor_position,
            frames_since_detection=0,
            total_detections=1,
            first_seen_frame=frame_number,
            last_seen_frame=frame_number
        )
        
        return new_id
    
    def _create_person_with_stable_id(
        self, 
        original_person: PersonDetection, 
        stable_id: int
    ) -> PersonDetection:
        """Create a new PersonDetection with stable ID, preserving all other data"""
        return PersonDetection(
            person_id=stable_id,
            keypoints=original_person.keypoints,
            skeleton_bones=original_person.skeleton_bones,
            bounding_box=original_person.bounding_box,
            floor_position=original_person.floor_position,
            hands_raised=original_person.hands_raised
        )
    
    def _remove_lost_tracks(self) -> List[int]:
        """Remove tracks that haven't been detected for too many frames"""
        tracks_to_remove = [
            track_id for track_id, tracked_person in self.tracked_people.items()
            if tracked_person.frames_since_detection > self.max_frames_missing
        ]
        
        for track_id in tracks_to_remove:
            del self.tracked_people[track_id]
        
        return tracks_to_remove
    
    def get_active_tracks_info(self) -> dict[int, dict]:
        """Get information about currently active tracks for debugging"""
        return {
            track_id: {
                'frames_since_detection': tracked.frames_since_detection,
                'total_detections': tracked.total_detections,
                'first_seen': tracked.first_seen_frame,
                'last_seen': tracked.last_seen_frame,
                'stable': tracked.total_detections >= self.min_detections_for_stability
            }
            for track_id, tracked in self.tracked_people.items()
        }


def determine_person_floor_position(
    detected_keypoints: np.ndarray,
    minimum_keypoint_confidence: float
) -> Tuple[Optional[float], Optional[float]]:
    """
    Determine the best pixel position representing a person's location on the floor.
    
    This function analyzes ankle keypoints to estimate where the person is standing.
    It prioritizes the midpoint between both ankles when both are confidently detected,
    and falls back to individual ankles when only one is reliable.
    
    Args:
        detected_keypoints: Array of shape (17, 3) containing [x, y, confidence] 
                           for each of the 17 COCO keypoints
        minimum_keypoint_confidence: Threshold for considering a keypoint reliable
        
    Returns:
        Tuple of (pixel_x, pixel_y) coordinates, or (None, None) if no reliable
        ankle positions are found
        
    Note:
        Ankle positions are preferred over other keypoints because they represent
        the person's contact with the ground plane most accurately.
    """
    left_ankle = detected_keypoints[KeypointIndices.LEFT_ANKLE]
    right_ankle = detected_keypoints[KeypointIndices.RIGHT_ANKLE]

    # Check if both ankles are confidently detected
    left_ankle_reliable = left_ankle[2] > minimum_keypoint_confidence
    right_ankle_reliable = right_ankle[2] > minimum_keypoint_confidence

    if left_ankle_reliable and right_ankle_reliable:
        # Use midpoint between both ankles for most accurate floor position
        midpoint_x = (left_ankle[0] + right_ankle[0]) / 2.0
        midpoint_y = (left_ankle[1] + right_ankle[1]) / 2.0
        return float(midpoint_x), float(midpoint_y)

    # Fall back to individual ankles if only one is reliable
    if left_ankle_reliable:
        return float(left_ankle[0]), float(left_ankle[1])

    if right_ankle_reliable:
        return float(right_ankle[0]), float(right_ankle[1])

    # Return None if no ankles are confidently detected
    return None, None


def detect_hands_raised(
    person_keypoints: np.ndarray,
    keypoint_confidence_threshold: float
) -> bool:
    """
    Detect if both hands (wrists) are raised above the shoulders.
    
    Args:
        person_keypoints: Raw keypoint array (17, 3) from YOLO
        keypoint_confidence_threshold: Minimum confidence for valid keypoints
        
    Returns:
        True if both wrists are above their corresponding shoulders, False otherwise
    """
    # Get keypoint data
    left_wrist = person_keypoints[KeypointIndices.LEFT_WRIST]
    right_wrist = person_keypoints[KeypointIndices.RIGHT_WRIST]
    left_shoulder = person_keypoints[KeypointIndices.LEFT_SHOULDER]
    right_shoulder = person_keypoints[KeypointIndices.RIGHT_SHOULDER]
    
    # Check if all required keypoints are confident enough
    required_keypoints = [left_wrist, right_wrist, left_shoulder, right_shoulder]
    if not all(kp[2] > keypoint_confidence_threshold for kp in required_keypoints):
        return False
    
    # Check if both wrists are above their corresponding shoulders
    left_hand_raised = left_wrist[1] < left_shoulder[1]  # y decreases upward
    right_hand_raised = right_wrist[1] < right_shoulder[1]  # y decreases upward
    
    return left_hand_raised and right_hand_raised


class SkeletonTracker:
    """
    Main class for real-time skeleton tracking and floor position mapping.
    
    This class handles pose detection and coordinate transformation with
    stable person ID tracking across frames.
    """
    
    def __init__(
        self,
        pose_model_path: str,
        homography_file_path: str,
        tracking_distance_threshold: float = 2.0,
        tracking_max_frames_missing: int = 30,
        expected_fps: float = 30.0,  # New parameter for FPS estimation
        show_floor_boundary: bool = True  # New parameter to control boundary visualization
    ):
        """
        Initialize the skeleton tracker with required models and tracking parameters.
        
        Args:
            pose_model_path: Path to YOLOv8 pose detection model file
            homography_file_path: Path to pre-computed homography matrix (.npy file)
            tracking_distance_threshold: Max distance (meters) for person matching
            tracking_max_frames_missing: Frames before considering person lost
            expected_fps: Expected frames per second for timing calculations
            show_floor_boundary: Whether to show the floor tracking boundary
        """
        self.pose_model = YOLO(pose_model_path)
        self.show_floor_boundary = show_floor_boundary
        
        try:
            self.floor_homography_matrix = np.load(homography_file_path)
        except Exception as error:
            raise RuntimeError(
                f"Failed to load homography matrix from '{homography_file_path}': {error}"
            )
        
        self._frame_counter = 0
        self.person_tracker = PersonTracker(
            max_distance_threshold=tracking_distance_threshold,
            max_frames_missing=tracking_max_frames_missing,
            fps=expected_fps
        )
        self._last_removed_person_ids: List[int] = []

    def get_last_removed_person_ids(self) -> List[int]:
        """Get person IDs that were removed in the last frame analysis"""
        return self._last_removed_person_ids

    def analyze_frame(
        self, 
        input_frame: np.ndarray, 
        detection_confidence: float,
        keypoint_confidence: float,
        inference_size: int
    ) -> FrameAnalysis:
        """
        Analyze a single frame and return detection results.
        
        Args:
            input_frame: Original camera/video frame
            detection_confidence: Minimum confidence for person detection
            keypoint_confidence: Minimum confidence for individual keypoints
            inference_size: Size for model inference
            
        Returns:
            FrameAnalysis object with all detection results
        """
        start_time = time.time()
        self._frame_counter += 1

        # Run pose detection
        detection_results = self.pose_model.predict(
            source=input_frame,
            imgsz=inference_size,
            conf=detection_confidence,
            verbose=False
        )

        # Process detection results into structured data
        detected_people = []
        primary_result = detection_results[0]

        if primary_result.keypoints is not None and len(primary_result.keypoints) > 0:
            all_keypoints = primary_result.keypoints.data.cpu().numpy()
            bounding_boxes = (primary_result.boxes.xyxy.cpu().numpy() 
                            if primary_result.boxes is not None else None)

            # Process each detected person with temporary indices first
            for person_index, person_keypoints in enumerate(all_keypoints):
                person_detection = self._create_person_detection(
                    person_keypoints, 
                    person_index,  # Temporary index, will be replaced by tracker
                    keypoint_confidence,
                    bounding_boxes[person_index] if bounding_boxes is not None else None
                )
                detected_people.append(person_detection)

        # Apply person tracking to assign stable IDs
        detected_people_with_stable_ids, removed_person_ids = self.person_tracker.update_frame(
            detected_people, self._frame_counter
        )
        
        # Store removed person IDs for later use
        self._last_removed_person_ids = removed_person_ids

        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        return FrameAnalysis(
            frame_number=self._frame_counter,
            detected_people=tuple(detected_people_with_stable_ids),
            processing_time_ms=processing_time
        )
    
    def _create_person_detection(
        self,
        person_keypoints: np.ndarray,
        person_index: int,
        keypoint_confidence: float,
        bounding_box_data: Optional[np.ndarray]
    ) -> PersonDetection:
        """Create PersonDetection from raw detection data"""
        # Create keypoints
        keypoints = create_keypoints_from_detection(
            person_keypoints, keypoint_confidence
        )
        
        # Create skeleton bones
        skeleton_bones = create_skeleton_bones_from_keypoints(
            person_keypoints, keypoint_confidence
        )
        
        # Create bounding box if available
        bounding_box = None
        if bounding_box_data is not None:
            x1, y1, x2, y2 = bounding_box_data
            bounding_box = BoundingBox(
                x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2)
            )
        
        # Create floor position
        floor_position = create_floor_position_from_keypoints(
            person_keypoints, person_index, keypoint_confidence, self.floor_homography_matrix
        )
        
        # Detect hands raised status
        hands_raised = detect_hands_raised(person_keypoints, keypoint_confidence)
        
        return PersonDetection(
            person_id=person_index,
            keypoints=keypoints,
            skeleton_bones=skeleton_bones,
            bounding_box=bounding_box,
            floor_position=floor_position,
            hands_raised=hands_raised
        )


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure command-line argument parser.
    
    Returns:
        Configured ArgumentParser instance with all required options
    """
    parser = argparse.ArgumentParser(
        description="Real-time human pose detection with floor position mapping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input source options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--cam", 
        type=int, 
        default=DEFAULT_CAMERA_INDEX,  # Use the configuration variable
        help="Camera index for live video capture"
    )
    input_group.add_argument(
        "--video", 
        type=str, 
        help="Path to video file for processing"
    )
    
    # Model configuration
    parser.add_argument(
        "--model", 
        default=DEFAULT_MODEL_PATH, 
        help="Path to YOLOv8 pose detection model file"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=DEFAULT_CONFIDENCE, 
        help="Person detection confidence threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--kpt_conf", 
        type=float, 
        default=DEFAULT_KEYPOINT_CONFIDENCE, 
        help="Individual keypoint confidence threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=DEFAULT_INFERENCE_SIZE, 
        help="Model inference size in pixels (larger = more accurate but slower)"
    )
    
    # Coordinate transformation
    parser.add_argument(
        "--homography", 
        default="floor_homography.npy", 
        help="Path to homography matrix file for pixel-to-floor coordinate transformation"
    )
    
    # Visualization options
    parser.add_argument(
        "--flip", 
        action="store_true", 
        default=DEFAULT_FLIP_CAMERA,
        help="Horizontally flip the camera view (useful for mirror-like display)"
    )
    parser.add_argument(
        "--show_boundary", 
        action="store_true", 
        default=DEFAULT_SHOW_BOUNDARY,
        help="Show the floor tracking boundary visualization"
    )
    
    # OSC communication
    parser.add_argument(
        "--osc_host", 
        type=str, 
        default=DEFAULT_OSC_HOST, 
        help="OSC server hostname for sending position data"
    )
    parser.add_argument(
        "--osc_port", 
        type=int, 
        default=DEFAULT_OSC_PORT,
        help="OSC server port for sending position data"
    )
    
    # Person tracking
    parser.add_argument(
        "--tracking_distance", 
        type=float, 
        default=DEFAULT_TRACKING_DISTANCE, 
        help="Maximum distance (meters) for matching people between frames"
    )
    parser.add_argument(
        "--tracking_timeout", 
        type=int, 
        default=DEFAULT_TRACKING_TIMEOUT, 
        help="Frames without detection before considering person lost"
    )
    
    return parser

def save_recording(recording_data, filename):
    """Save recorded tracking data to JSON file."""
    output = {
        'metadata': {
            'recorded_at': datetime.now().isoformat(),
            'total_frames': len(recording_data),
            'duration': recording_data[-1]['timestamp'] if recording_data else 0
        },
        'frames': recording_data
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved {len(recording_data)} frames to {filename}")

def replay_tracking_data(json_file, osc_host, osc_port):
    """Replay recorded tracking data and send via OSC."""
    print(f"\n=== REPLAYING: {json_file} ===")
    
    # Implementation can be added later if needed
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    frames = data['frames']
    osc_client = SimpleUDPClient(osc_host, osc_port)
    
    print(f"Replaying {len(frames)} frames...")
    start_time = time.time()
    
    for frame_data in frames:
        target_time = start_time + frame_data['timestamp']
        current_time = time.time()
        
        if target_time > current_time:
            time.sleep(target_time - current_time)
        
        # Send OSC message for this frame
        visitor_positions = []
        for track in frame_data['tracks']:
            track_id = track['track_id']
            x_normalized = track['X'] / 3.5  # Normalize to room size
            y_normalized = track['Y'] / 2.5
            visitor_positions.extend([track_id, x_normalized, y_normalized, 0])
        
        if visitor_positions:
            osc_client.send_message("/visitors", visitor_positions)
            print(f"Replayed frame: {visitor_positions}")

def main():
    """
    Main entry point for the skeleton tracking application.
    
    Handles argument parsing, video capture setup, and the main processing loop
    for real-time pose detection and floor position mapping.
    """
    # Parse command line arguments
    argument_parser = create_argument_parser()
    args = argument_parser.parse_args()

    # Initialize the skeleton tracker with FPS estimation
    try:
        skeleton_tracker = SkeletonTracker(
            pose_model_path=args.model,
            homography_file_path=args.homography,
            tracking_distance_threshold=args.tracking_distance,
            tracking_max_frames_missing=args.tracking_timeout,
            expected_fps=30.0,  # Reasonable default FPS
            show_floor_boundary=args.show_boundary
        )
    except Exception as error:
        print(f"Failed to initialize skeleton tracker: {error}")
        return

    # Set up output handlers
    console_handler = ConsoleOutputHandler()
    osc_handler = OSCOutputHandler(args.osc_host, args.osc_port)
    output_handler = CombinedOutputHandler([console_handler, osc_handler])
    visualizer = OpenCVVisualizer()

    # Recording state
    is_recording = False
    recording_data = []
    recording_start_time = None
    
    # Create recordings folder if it doesn't exist
    recordings_folder = "recordings"
    os.makedirs(recordings_folder, exist_ok=True)

    # Set up video capture source (camera or file)
    if args.video:
        video_capture = cv2.VideoCapture(args.video)
        print(f"Processing video file: {args.video}")
    elif args.cam is not None:
        video_capture = cv2.VideoCapture(args.cam)
        print(f"Using camera index: {args.cam} (configured default: {DEFAULT_CAMERA_INDEX})")
    else:
        argument_parser.error("Must specify either --cam or --video")

    # Verify that video source opened successfully
    if not video_capture.isOpened():
        if args.video:
            raise RuntimeError(f"Could not open video file: {args.video}")
        else:
            raise RuntimeError(f"Could not open camera index: {args.cam}")

    print("Skeleton tracking with floor mapping is running...")
    print("Press 'q' to quit, or close the window to stop.")
    print("Hold 'r' to record tracking data, release to save.")
    print("Press 's' to stop recording explicitly.")
    print("Green boundary shows the 3.5m x 3.0m tracked floor area.")
    print("Only people INSIDE the boundary are tracked, recorded, and sent via OSC.")
    print("People outside are visible but not processed.")
    print(f"Recordings will be saved to: {os.path.abspath(recordings_folder)}/")

    try:
        # Main processing loop
        while True:
            # Capture frame from video source
            frame_captured_successfully, current_frame = video_capture.read()
            
            # Check if we've reached the end of a video file
            if not frame_captured_successfully:
                if args.video:
                    print("Reached end of video file.")
                else:
                    print("Failed to capture frame from camera.")
                break

            # Apply horizontal flip if requested (useful for mirror-like camera view)
            if args.flip:
                current_frame = cv2.flip(current_frame, 1)

            # Analyze the frame and get detection results
            frame_analysis = skeleton_tracker.analyze_frame(
                input_frame=current_frame,
                detection_confidence=args.conf,
                keypoint_confidence=args.kpt_conf,
                inference_size=args.imgsz
            )

            # Handle logging and communication
            output_handler.log_detection_info(frame_analysis)
            
            # Send all positions for this frame (only if changed)
            output_handler.send_positions_frame(frame_analysis)

            # Record frame data if recording
            if is_recording:
                frame_time = time.time() - recording_start_time
                frame_data = {
                    'timestamp': frame_time,
                    'tracks': []
                }
                for person in frame_analysis.detected_people:
                    if person.floor_position:
                        frame_data['tracks'].append({
                            'track_id': person.person_id,
                            'X': float(person.floor_position.floor_x),
                            'Y': float(person.floor_position.floor_y)
                        })
                recording_data.append(frame_data)

            # Create and display visualization
            visualization_frame = visualizer.create_visualization_frame(
                current_frame, frame_analysis
            )
            
            # Add recording indicator to visualization
            if is_recording:
                cv2.putText(visualization_frame, "RECORDING", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.circle(visualization_frame, (200, 20), 10, (0, 0, 255), -1)
                cv2.putText(visualization_frame, f"Frames: {len(recording_data)}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Real-time Skeleton Tracking with Floor Position Mapping", visualization_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            # Check if 'r' key is currently pressed
            if key == ord('r') and not is_recording:
                # Start recording
                is_recording = True
                recording_data = []
                recording_start_time = time.time()
                print("\n=== RECORDING STARTED ===")
            elif key != ord('r') and key != 255 and is_recording and key != ord('q'):
                # Any other key pressed while recording - stop and save
                is_recording = False
                if len(recording_data) > 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(recordings_folder, f"tracking_recording_{timestamp}.json")
                    save_recording(recording_data, filename)
                    print(f"\n=== RECORDING SAVED: {filename} ===")
                    print(f"Total frames: {len(recording_data)}")
                    print(f"Duration: {recording_data[-1]['timestamp']:.2f}s")
                else:
                    print("\n=== RECORDING CANCELLED (no data) ===")
            elif key == ord('s') and is_recording:
                # Press 's' to stop recording explicitly
                is_recording = False
                if len(recording_data) > 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(recordings_folder, f"tracking_recording_{timestamp}.json")
                    save_recording(recording_data, filename)
                    print(f"\n=== RECORDING SAVED: {filename} ===")
                    print(f"Total frames: {len(recording_data)}")
                    print(f"Duration: {recording_data[-1]['timestamp']:.2f}s")
                else:
                    print("\n=== RECORDING CANCELLED (no data) ===")
            
            # Check for quit command (press 'q' key)
            if key == ord("q"):
                # Save recording if still recording when quitting
                if is_recording and len(recording_data) > 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(recordings_folder, f"tracking_recording_{timestamp}.json")
                    save_recording(recording_data, filename)
                    print(f"\n=== RECORDING AUTO-SAVED: {filename} ===")
                print("Quit command received.")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
        # Save recording if interrupted while recording
        if is_recording and len(recording_data) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(recordings_folder, f"tracking_recording_{timestamp}.json")
            save_recording(recording_data, filename)
            print(f"\n=== RECORDING AUTO-SAVED: {filename} ===")
    except Exception as error:
        print(f"An error occurred during processing: {error}")
        # Save recording if error occurred while recording
        if is_recording and len(recording_data) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(recordings_folder, f"tracking_recording_{timestamp}.json")
            save_recording(recording_data, filename)
            print(f"\n=== RECORDING AUTO-SAVED: {filename} ===")
    finally:
        # Clean up resources
        video_capture.release()
        cv2.destroyAllWindows()
        print("Resources cleaned up. Application terminated.")


if __name__ == "__main__":
    main()