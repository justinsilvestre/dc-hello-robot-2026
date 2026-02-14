"""
Visualization tool for recorded tracking data.

This tool reads JSON recording files from the recordings folder and displays
the movement paths on a 2D grid representing the real-world room dimensions.

Usage:
# Interactive menu to select recording
python3 visualize_recordings.py

# Visualize specific recording
python3 visualize_recordings.py --recording recordings/tracking_recording_20260207_201728.json

# Create animated visualization
python3 visualize_recordings.py --mode animated

# Create heatmap
python3 visualize_recordings.py --mode heatmap

# Compare multiple recordings
python3 visualize_recordings.py --mode compare

# Overlay all recordings
python3 visualize_recordings.py --mode overlay

# Custom room dimensions
python3 visualize_recordings.py --room_width 4.0 --room_height 3.0

# Save visualization
python3 visualize_recordings.py --mode static --save tracking_viz.png


    python3 visualize_recordings.py
    python3 visualize_recordings.py --recording specific_file.json
    python3 visualize_recordings.py --room_width 4.0 --room_height 3.0
"""

import argparse
import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gs
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import seaborn as sns
import threading
import time

# Try to import watchdog, make it optional
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("⚠️  Warning: 'watchdog' module not installed. Live monitoring will be disabled.")
    print("   To enable live monitoring, install with: pip3 install watchdog")

# Room dimensions (meters) - Updated to match your actual coordinate system
DEFAULT_ROOM_WIDTH = 3.5   # X dimension in meters
DEFAULT_ROOM_HEIGHT = 2.5  # Y dimension in meters

# Coordinate bounds - proper room dimensions from (0,0) to (room_width, room_height)
DEFAULT_X_MIN = 0.0   # Start at 0
DEFAULT_X_MAX = 3.5   # Match room width
DEFAULT_Y_MIN = 0.0   # Start at 0 (fixed from 0.5)
DEFAULT_Y_MAX = 2.5   # Match room height (fixed from 2.8)

# Visualization settings - Updated for black/white design
PERSON_GRAYS = [1.0, 0.8, 0.6, 0.4, 0.9, 0.7, 0.5, 0.3, 0.85, 0.65]  # Different grayscale values (0=black, 1=white)
TRAIL_ALPHA = 0.8
CURRENT_POSITION_SIZE = 100
TRAIL_LINE_WIDTH = 2

class RecordingFileHandler:
    """File system event handler for monitoring new JSON recordings"""
    
    def __init__(self, visualizer, callback):
        if not WATCHDOG_AVAILABLE:
            raise ImportError("watchdog module not available")
        
        # Import here to avoid errors when watchdog is not available
        from watchdog.events import FileSystemEventHandler
        
        class Handler(FileSystemEventHandler):
            def __init__(self, vis, cb):
                self.visualizer = vis
                self.callback = cb
                self.last_modified = {}
                
            def on_created(self, event):
                if not event.is_directory and event.src_path.endswith('.json'):
                    print(f"\n📁 New recording detected: {os.path.basename(event.src_path)}")
                    # Wait a moment for file to be completely written
                    time.sleep(0.5)
                    self.callback()
            
            def on_modified(self, event):
                if not event.is_directory and event.src_path.endswith('.json'):
                    # Avoid duplicate notifications for the same file
                    current_time = time.time()
                    if event.src_path not in self.last_modified or \
                       current_time - self.last_modified[event.src_path] > 1.0:
                        self.last_modified[event.src_path] = current_time
                        print(f"\n📝 Recording updated: {os.path.basename(event.src_path)}")
                        time.sleep(0.5)
                        self.callback()
        
        self.handler = Handler(visualizer, callback)
    
    def __getattr__(self, name):
        # Delegate all other calls to the internal handler
        return getattr(self.handler, name)

class TrackingDataVisualizer:
    """Visualizes recorded tracking data on a 2D room grid"""
    
    def __init__(self, room_width: float = DEFAULT_ROOM_WIDTH, room_height: float = DEFAULT_ROOM_HEIGHT):
        self.room_width = room_width
        self.room_height = room_height
        self.x_min = 0.0  # Always start at 0
        self.x_max = room_width
        self.y_min = 0.0  # Always start at 0
        self.y_max = room_height
        self.recordings_folder = "recordings"
        self.live_monitoring = False
        self.observer = None
        self.monitoring_thread = None
        self.current_mode = None
        self.current_figure = None
        
    def start_live_monitoring(self, mode: str, refresh_callback):
        """Start live monitoring for new recordings"""
        if not WATCHDOG_AVAILABLE:
            print("❌ Live monitoring unavailable: watchdog module not installed")
            print("   Install with: pip3 install watchdog")
            return False
            
        if self.live_monitoring:
            return True
            
        self.live_monitoring = True
        self.current_mode = mode
        
        print(f"\n🔴 LIVE MONITORING ACTIVE for {mode.upper()} mode")
        print("   Watching for new recordings... Press Ctrl+C to stop")
        
        try:
            # Set up file system watcher
            event_handler = RecordingFileHandler(self, refresh_callback)
            self.observer = Observer()
            self.observer.schedule(event_handler, self.recordings_folder, recursive=False)
            self.observer.start()
            return True
        except Exception as e:
            print(f"❌ Failed to start live monitoring: {e}")
            self.live_monitoring = False
            return False
        
    def stop_live_monitoring(self):
        """Stop live monitoring"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
        self.live_monitoring = False
        print("\n🔴 Live monitoring stopped")

    def load_recording(self, filename: str) -> Dict:
        """Load a single recording JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            print(f"✅ Loaded recording: {filename}")
            print(f"   Duration: {data['metadata']['duration']:.2f}s")
            print(f"   Frames: {data['metadata']['total_frames']}")
            return data
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return None
    
    def get_available_recordings(self) -> List[str]:
        """Get list of available recording files"""
        pattern = os.path.join(self.recordings_folder, "*.json")
        files = glob.glob(pattern)
        files.sort(key=os.path.getmtime, reverse=True)  # Sort by modification time, newest first
        return files
    
    def extract_paths(self, recording_data: Dict) -> Dict[int, List[Tuple[float, float, float]]]:
        """Extract movement paths for each person, clamping coordinates to room boundaries"""
        paths = {}  # person_id -> [(timestamp, x, y), ...]
        
        for frame in recording_data['frames']:
            timestamp = frame['timestamp']
            for track in frame['tracks']:
                person_id = track['track_id']
                
                # Clamp coordinates to room boundaries (negatives become 0)
                x = max(0.0, min(track['X'], self.room_width))
                y = max(0.0, min(track['Y'], self.room_height))
                
                if person_id not in paths:
                    paths[person_id] = []
                
                paths[person_id].append((timestamp, x, y))
        
        return paths
    
    def _setup_room_axes(self, ax):
        """Set up consistent room axes for all visualization modes"""
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        
        # Add room boundary - full room dimensions
        room_rect = patches.Rectangle(
            (0, 0), self.room_width, self.room_height,
            linewidth=2, edgecolor='white', facecolor='none'
        )
        ax.add_patch(room_rect)
        
    def create_dashboard_visualization(self, recording_files: List[str], live_mode: bool = False) -> None:
        """Create a dashboard with heatmap visualization"""
        if not recording_files:
            print("❌ No recording files provided for dashboard")
            return
        
        # Disable live mode if watchdog is not available
        if live_mode and not WATCHDOG_AVAILABLE:
            print("⚠️  Live monitoring disabled: watchdog not installed")
            live_mode = False
        
        # Set up black background
        plt.style.use('dark_background')
        
        def update_dashboard():
            """Update dashboard with current files"""
            # Get fresh file list
            current_files = self.get_available_recordings() if live_mode else recording_files
            
            if hasattr(self, 'current_figure') and self.current_figure:
                plt.close(self.current_figure)
            
            # Create figure with GridSpec for heatmap and colorbar
            fig = plt.figure(figsize=(12, 8), facecolor='black')
            self.current_figure = fig
            
            # Create gridspec: 1 large column for heatmap, 1 small column for colorbar
            grid = gs.GridSpec(1, 2, figure=fig, width_ratios=[1, 0.05], wspace=0.1)
            
            fig.suptitle(f'Tracking Dashboard - Heatmap', 
                        fontsize=16, color='white', y=0.95)
            
            # Process recordings data for heatmap
            combined_occupancy_grid = np.zeros((50, 50))  # For heatmap
            total_duration = 0
            total_frames = 0
            processed_recordings = 0
            
            # Load and process all recordings
            for recording_file in current_files:
                recording_data = self.load_recording(recording_file)
                if not recording_data:
                    continue
                    
                paths = self.extract_paths(recording_data)
                if not paths:
                    continue
                
                processed_recordings += 1
                total_duration += recording_data['metadata']['duration']
                total_frames += recording_data['metadata']['total_frames']
                
                # Build combined heatmap data
                for person_id, path_data in paths.items():
                    for timestamp, x, y in path_data:
                        grid_x = int(np.clip(x / self.room_width * 50, 0, 49))
                        grid_y = int(np.clip(y / self.room_height * 50, 0, 49))
                        combined_occupancy_grid[grid_y, grid_x] += 1
            
            # Create Heatmap (main area)
            ax = fig.add_subplot(grid[0, 0])
            ax.set_facecolor('black')
            
            if processed_recordings > 0:
                im = ax.imshow(combined_occupancy_grid, extent=[0, self.room_width, 0, self.room_height],
                              origin='lower', cmap='hot', alpha=0.8)
                
                # Add colorbar in the second column
                cax = fig.add_subplot(grid[0, 1])
                cbar = plt.colorbar(im, cax=cax)
                cbar.set_label('Occupancy Count', rotation=270, labelpad=15, color='white')
                cbar.ax.yaxis.set_tick_params(color='white')
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            
            self._setup_room_axes(ax)
            
            # Add subtitle with statistics
            subtitle = f'Records: {processed_recordings} | Duration: {total_duration:.1f}s | Frames: {total_frames}'
            if live_mode:
                subtitle 
            ax.set_title(subtitle, fontsize=12, color='white', pad=15)
            
            plt.subplots_adjust(top=0.85)
            plt.draw()
        
        # Initial visualization
        update_dashboard()
        
        if live_mode:
            # Start live monitoring
            monitoring_started = self.start_live_monitoring("dashboard", update_dashboard)
            
            if monitoring_started:
                try:
                    plt.show()
                except KeyboardInterrupt:
                    pass
                finally:
                    self.stop_live_monitoring()
            else:
                # Fallback to static mode
                plt.show()
        else:
            plt.show()

    def _add_direction_arrows(self, ax, x_coords, y_coords, color):
        """Add direction arrows along the path"""
        if len(x_coords) < 2:
            return
        
        # Add arrows at regular intervals
        arrow_interval = max(1, len(x_coords) // 8)
        
        for i in range(arrow_interval, len(x_coords), arrow_interval):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            
            if abs(dx) > 0.01 or abs(dy) > 0.01:  # Only add arrow if there's significant movement
                ax.arrow(x_coords[i-1], y_coords[i-1], dx*0.3, dy*0.3,
                        head_width=0.03, head_length=0.03, fc=color, ec=color, alpha=0.6)
    
    def _add_coordinate_labels(self, ax):
        """Add coordinate grid labels using room coordinate system"""
        # Add coordinate markers at grid intersections in white
        x_step = self.room_width / 7   # 7 divisions for room width
        y_step = self.room_height / 5  # 5 divisions for room height
        
        for x in np.arange(0, self.room_width + x_step/2, x_step):
            for y in np.arange(0, self.room_height + y_step/2, y_step):
                ax.plot(x, y, '+', color='white', markersize=2, alpha=0.4)

def main():
    parser = argparse.ArgumentParser(
        description="Visualize recorded tracking data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--recording",
        type=str,
        help="Specific recording file to visualize (otherwise shows menu)"
    )
    
    parser.add_argument(
        "--room_width",
        type=float,
        default=DEFAULT_ROOM_WIDTH,
        help="Room width in meters"
    )
    
    parser.add_argument(
        "--room_height",
        type=float,
        default=DEFAULT_ROOM_HEIGHT,
        help="Room height in meters"
    )
    
    parser.add_argument(
        "--mode",
        choices=['static', 'animated', 'heatmap', 'compare', 'overlay', 'dashboard'],
        default='dashboard',
        help="Visualization mode"
    )
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = TrackingDataVisualizer(args.room_width, args.room_height)
    
    # Get available recordings
    available_files = visualizer.get_available_recordings()
    
    if not available_files:
        print("❌ No recording files found in 'recordings' folder")
        return
    
    # Select recording(s)
    live_monitoring_mode = False
    
    if args.recording:
        if os.path.exists(args.recording):
            selected_files = [args.recording]
        else:
            print(f"❌ Recording file not found: {args.recording}")
            return
    else:
        print("\n📁 Available recordings:")
        for i, filename in enumerate(available_files):
            file_stat = os.stat(filename)
            file_size = file_stat.st_size / 1024  # KB
            mod_time = datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  {i+1}: {os.path.basename(filename)} ({file_size:.1f} KB, {mod_time})")
        
        if args.mode in ['compare', 'overlay', 'heatmap', 'dashboard']:
            mode_name = args.mode.title()
            live_suffix = " (live monitoring available)" if WATCHDOG_AVAILABLE else " (live monitoring unavailable)"
            print(f"\n🎯 {mode_name} mode: Select recordings")
            print(f"Enter recording numbers (comma-separated, e.g., 1,2,3) or 'all' for all recordings{live_suffix}:")
            selection = input("Selection: ").strip()
            
            if selection.lower() == 'all':
                selected_files = available_files
                live_monitoring_mode = WATCHDOG_AVAILABLE
                if WATCHDOG_AVAILABLE:
                    print(f"🔴 Selected all {len(selected_files)} recordings for {mode_name.lower()} with LIVE monitoring")
                else:
                    print(f"Selected all {len(selected_files)} recordings for {mode_name.lower()} (static mode)")
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in selection.split(',')]
                    selected_files = [available_files[i] for i in indices if 0 <= i < len(available_files)]
                    print(f"Selected {len(selected_files)} recordings for {mode_name.lower()}")
                except ValueError:
                    print("❌ Invalid selection")
                    return
        else:
            print(f"\nSelect a recording (1-{len(available_files)}):")
            try:
                selection = int(input("Selection: ")) - 1
                if 0 <= selection < len(available_files):
                    selected_files = [available_files[selection]]
                else:
                    print("❌ Invalid selection")
                    return
            except ValueError:
                print("❌ Invalid selection")
                return
    
    # Create visualization
    print(f"\n🎨 Creating {args.mode} visualization...")
    if live_monitoring_mode:
        print("🔴 Live monitoring enabled - new recordings will be automatically included")
    elif WATCHDOG_AVAILABLE is False and args.mode in ['compare', 'overlay', 'heatmap', 'dashboard']:
        print("📊 Static mode - install 'watchdog' for live monitoring: pip3 install watchdog")
    
    try:
        if args.mode == 'dashboard':
            visualizer.create_dashboard_visualization(selected_files, live_monitoring_mode)
        elif args.mode == 'compare':
            visualizer.compare_recordings(selected_files, live_monitoring_mode)
        elif args.mode == 'overlay':
            visualizer.create_overlay_visualization(selected_files, live_monitoring_mode)
        elif args.mode == 'heatmap':
            # For heatmap with multiple recordings, use combined heatmap
            if len(selected_files) == 1 and not live_monitoring_mode:
                recording_data = visualizer.load_recording(selected_files[0])
                if recording_data:
                    visualizer.create_heatmap_visualization(recording_data)
            else:
                visualizer.create_combined_heatmap_visualization(selected_files, live_mode=live_monitoring_mode)
        else:
            recording_data = visualizer.load_recording(selected_files[0])
            if not recording_data:
                return
            
            if args.mode == 'static':
                visualizer.create_static_visualization(recording_data)
            elif args.mode == 'animated':
                visualizer.create_animated_visualization(recording_data)
    except KeyboardInterrupt:
        print("\n🛑 Visualization stopped by user")
    finally:
        if live_monitoring_mode:
            visualizer.stop_live_monitoring()

if __name__ == "__main__":
    main()