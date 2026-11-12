# Utility functions for motion data and MIDI processing
import numpy as np
import more_itertools as mit
import cv2
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from math import atan2

# MIDI note name array (12 notes from A through G#)
note_names = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]


def midi_name(midi_num):
    """
    Convert a MIDI note number to its octave and note name

    Args:
        midi_num (int): MIDI note number (0-127)

    Returns:
        tuple: (octave number, note name)
    """
    octave = str(midi_num // 12 - 1)  # Compute octave
    note = note_names[(midi_num + 3) % 12]  # Compute note name
    return octave, note


def get_close_gap(points, max_gap=15, min_points=3):
    """
    Find and return small gaps of NaN values in the data
    Identifies short missing-value segments in motion capture data to locate regions suitable for interpolation

    Args:
        points (array): Array of data points
        max_gap (int): Maximum gap size (default: 15)
        min_points (int): Minimum number of points (default: 3)

    Returns:
        list: List of small gaps
    """
    # Find positions of NaN values
    nan_pos = np.nonzero(np.isnan(points))[0]
    if len(nan_pos) == 0:
        return []

    # Group consecutive NaN positions together
    groups = [list(group) for group in mit.consecutive_groups(nan_pos)]
    new_groups = []
    current_group = []

    # Inspect each group and keep only those that match the criteria
    for n in range(len(groups)):
        if len(groups[n]) > max_gap:  # Gap is too large
            if len(current_group) != 0:
                new_groups.extend(current_group)
            current_group = []
        else:
            # Skip small gaps near the start or end
            if groups[n][0] < min_points and groups[n][-1] > len(points) - min_points:
                continue
            current_group.append(groups[n])

    if len(current_group) != 0:
        new_groups.extend(current_group)
    return new_groups


def img2mp4(paths, pathOut, fps=120):
    """
    Convert a sequence of image files into an MP4 video

    Args:
        paths (list): List of image file paths
        pathOut (str): Output video file path
        fps (int): Frame rate (default: 120)
    """
    frame_array = []

    # Read every image and store it in the frame array
    for idx, path in tqdm(enumerate(paths)):
        img = cv2.imread(path)
        height, width, layers = img.shape
        size = (width, height)
        frame_array.append(img)

    # Create the video writer and write frames
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*"DIVX"), fps, size)
    for i in tqdm(range(len(frame_array))):
        out.write(frame_array[i])
    out.release()


def unit_vector(vector):
    """
    Convert a vector into a unit vector

    Args:
        vector (array): Input vector

    Returns:
        array: Unit vector (vector with magnitude 1)
    """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """
    Compute the angle between two vectors in radians

    Args:
        v1, v2 (array): Two vectors

    Returns:
        float: Angle between the two vectors (in radians)

    Examples:
        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966  # pi/2 radians (90 degrees)
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0  # 0 radians (0 degrees)
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793  # pi radians (180 degrees)
    """
    # Convert both vectors into unit vectors
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    # Compute the angle via the dot product (clamp to avoid numerical errors)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_between_2d(v1, v2):
    """
    Compute the angle between two points on a 2D plane

    Args:
        v1, v2 (array): Two 2D points

    Returns:
        float: Angle (in radians)
    """
    angle = atan2(v1[1] - v2[1], v1[0] - v2[0])
    # Note: extra logic for direction-aware angle calculation (currently disabled)
    # if(v1_u.lX*b.Y - a.Y*b.X < 0):
    # .        angle=-angle
    return angle


def set_axes_equal(ax):
    """
    Make the axis scales of a 3D plot equal
    Adjusts the axis ranges so that spheres look like spheres and cubes like cubes

    Args:
        ax: matplotlib 3D axes object (e.g., the output of plt.gca())
    """
    # Get the current limits of each axis
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    # Compute the range and center of each axis
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # Make the plot bounding box a sphere in the infinity-norm sense
    # Use half of the largest range as the plot radius
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    # Apply the same range to every axis
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def make_index_list(data):
    """
    Sort the data by value and return the original index order

    Args:
        data (list): Data to sort

    Returns:
        list: List of original indices in sorted order
    """
    # Pair each value with its index
    indexed_data = list(enumerate(data))
    # Sort by value
    sorted_indexed_data = sorted(indexed_data, key=lambda x: x[1])
    # Extract just the indices in the sorted order
    sorted_indices = [index for index, value in sorted_indexed_data]

    return sorted_indices


def lineseg_dist(p, a, b):
    """
    Compute the shortest distance between a point and a line segment in 3D space

    Args:
        p (array): Coordinates of the point
        a, b (array): Coordinates of the two endpoints of the segment

    Returns:
        float: Shortest distance between the point and the segment

    Reference: https://stackoverflow.com/questions/56463412/distance-from-a-point-to-a-line-segment-in-3d-python
    """
    # Compute the normalized direction vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # Signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # Clamped parallel distance (clamped to the segment's range)
    h = np.maximum.reduce([s, t, 0])

    # Perpendicular distance component
    c = np.cross(p - a, d)

    # Compute Euclidean distance (Pythagorean theorem)
    return np.hypot(h, np.linalg.norm(c))


def load_fingertip(file_path):
    """
    Load fingertip motion data
    Extracts 3D coordinates of the wrists and each fingertip from a CSV file

    Args:
        file_path (str): Path to the motion data CSV file

    Returns:
        tuple: (left wrist 1, left wrist 2, right wrist 1, right wrist 2,
                right thumb~little, left thumb~little) coordinate arrays
    """
    # Load motion data
    data = pd.read_csv(file_path)

    # Note: this format does not have a TimeCode column
    # Extract coordinates using the correct column naming convention (_z, _x, _y)

    # Resolve column name variants (Lend1 vs LEnd1)
    def col(name):
        if name in data.columns: return name
        # Try case variants
        for c in data.columns:
            if c.lower() == name.lower(): return c
        raise KeyError(f"Column not found: {name}")

    # Left hand endpoints
    Lend1_z = data[col("LEnd1_z")]
    Lend1_x = data[col("LEnd1_x")]
    Lend1_y = data[col("LEnd1_y")]

    Lend2_z = data[col("LEnd_z")]
    Lend2_x = data[col("LEnd_x")]
    Lend2_y = data[col("LEnd_y")]

    # Right hand endpoints
    Rend1_z = data[col("REnd1_z")]
    Rend1_x = data[col("REnd1_x")]
    Rend1_y = data[col("REnd1_y")]

    Rend2_z = data[col("REnd_z")]
    Rend2_x = data[col("REnd_x")]
    Rend2_y = data[col("REnd_y")]

    # Arrange coordinates in the correct order (x, y, z)
    Lend1 = np.array([Lend1_x, Lend1_y, Lend1_z]).T
    Lend2 = np.array([Lend2_x, Lend2_y, Lend2_z]).T

    Rend1 = np.array([Rend1_x, Rend1_y, Rend1_z]).T
    Rend2 = np.array([Rend2_x, Rend2_y, Rend2_z]).T

    # Fallback: if End1 is entirely NaN, fall back to End
    if np.all(np.isnan(Lend1)):
        Lend1 = Lend2.copy()
    if np.all(np.isnan(Rend1)):
        Rend1 = Rend2.copy()

    # Right hand fingertips (RTF: thumb, RIF: index, RMF: middle, RRF: ring, RLF: little)
    rtf_z_array = data["RTF_z"].values
    rtf_x_array = data["RTF_x"].values
    rtf_y_array = data["RTF_y"].values

    rif_z_array = data["RIF_z"].values
    rif_x_array = data["RIF_x"].values
    rif_y_array = data["RIF_y"].values

    rmf_z_array = data["RMF_z"].values
    rmf_x_array = data["RMF_x"].values
    rmf_y_array = data["RMF_y"].values

    rrf_z_array = data["RRF_z"].values
    rrf_x_array = data["RRF_x"].values
    rrf_y_array = data["RRF_y"].values

    rlf_z_array = data["RLF_z"].values
    rlf_x_array = data["RLF_x"].values
    rlf_y_array = data["RLF_y"].values

    # Left hand fingertips (LTF: thumb, LIF: index, LMF: middle, LRF: ring, LLF: little)
    ltf_z_array = data["LTF_z"].values
    ltf_x_array = data["LTF_x"].values
    ltf_y_array = data["LTF_y"].values

    lif_z_array = data["LIF_z"].values
    lif_x_array = data["LIF_x"].values
    lif_y_array = data["LIF_y"].values

    lmf_z_array = data["LMF_z"].values
    lmf_x_array = data["LMF_x"].values
    lmf_y_array = data["LMF_y"].values

    lrf_z_array = data["LRF_z"].values
    lrf_x_array = data["LRF_x"].values
    lrf_y_array = data["LRF_y"].values

    llf_z_array = data["LLF_z"].values
    llf_x_array = data["LLF_x"].values
    llf_y_array = data["LLF_y"].values

    # Combine x, y, z coordinates in the correct order
    rtf_array = np.array([rtf_x_array, rtf_y_array, rtf_z_array]).T
    rif_array = np.array([rif_x_array, rif_y_array, rif_z_array]).T
    rmf_array = np.array([rmf_x_array, rmf_y_array, rmf_z_array]).T
    rrf_array = np.array([rrf_x_array, rrf_y_array, rrf_z_array]).T
    rlf_array = np.array([rlf_x_array, rlf_y_array, rlf_z_array]).T

    ltf_array = np.array([ltf_x_array, ltf_y_array, ltf_z_array]).T
    lif_array = np.array([lif_x_array, lif_y_array, lif_z_array]).T
    lmf_array = np.array([lmf_x_array, lmf_y_array, lmf_z_array]).T
    lrf_array = np.array([lrf_x_array, lrf_y_array, lrf_z_array]).T
    llf_array = np.array([llf_x_array, llf_y_array, llf_z_array]).T

    return (
        Lend1,
        Lend2,
        Rend1,
        Rend2,
        rtf_array,
        rif_array,
        rmf_array,
        rrf_array,
        rlf_array,
        ltf_array,
        lif_array,
        lmf_array,
        lrf_array,
        llf_array,
    )

def load_all_data(file_path):
    """
    Load all joint data
    Returns every joint position from a CSV file as a dictionary

    Args:
        file_path (str): Path to the motion data CSV file

    Returns:
        dict: Dictionary keyed by joint name, with 3D coordinate arrays as values
    """
    # Load motion data
    data = pd.read_csv(file_path)

    # Note: this format does not have a TimeCode column
    # Extract every available column
    columns = data.columns

    # Filter out unnamed columns
    columns = [col for col in columns if 'Unnamed' not in col]

    # Find unique joint names by stripping the _x, _y, _z suffixes
    joint_names = []
    for col in columns:
        if col.endswith('_x'):
            joint_name = col[:-2]  # Strip the '_x' suffix
            joint_names.append(joint_name)

    # Dictionary to hold each joint's position data
    joint_pos = defaultdict(list)
    for joint in joint_names:
        # Extract x, y, z coordinates using the correct naming convention
        x = data[joint + '_x'].values
        y = data[joint + '_y'].values
        z = data[joint + '_z'].values

        # Combine into an array in the proper coordinate order (x, y, z)
        joint_pos[joint] = np.array([x, y, z]).T

    return joint_pos
