import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from joblib import Parallel, delayed
from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm
import contextlib
import joblib
from scipy.ndimage import uniform_filter
from itertools import combinations
import matplotlib.colors
from scipy.ndimage import center_of_mass
from skimage.filters import difference_of_gaussians


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def replace_nans_with_average(images):
    def average_of_neighbors(images, i, j):
        neighbors = []
        if i > 0:
            neighbors.append(images[i - 1, j])
        if i < images.shape[0] - 1:
            neighbors.append(images[i + 1, j])
        if j > 0:
            neighbors.append(images[i, j - 1])
        if j < images.shape[1] - 1:
            neighbors.append(images[i, j + 1])

        neighbors = np.array(neighbors)
        valid_neighbors = neighbors[~np.isnan(neighbors)]

        if len(valid_neighbors) > 0:
            return np.nanmean(valid_neighbors)
        else:
            return np.nan

    for i in range(images.shape[0]):
        for j in range(images.shape[1]):
            if np.any(np.isnan(images[i, j])):
                print(i, j)
                images[i, j] = average_of_neighbors(images, i, j)

    return images

def apply_spatial_local_averaging(data4d, kernel_size):
    """
    Apply local averaging over neighboring images in the (n, m) grid.

    Parameters:
        images_2d (np.ndarray): 4D array of shape (n, m, height, width).
        kernel_size (int or tuple): Size of the averaging window in the (n, m) grid.

    Returns:
        np.ndarray: Locally averaged images with the same shape as input.
    """
    n, m, h, w = data4d.shape
    kernel_size = (kernel_size, kernel_size, 1, 1)  # Apply kernel over (n, m), not (h, w)
    
    # Apply uniform filter over the first two dimensions (spatial dimensions)
    return uniform_filter(data4d, size=kernel_size, mode='nearest')


def annular_mask(array, center=None, inner_radius=0):
    """
    Create an annular mask for a 2D array.
    
    Parameters:
        array (numpy.ndarray): Input 2D array.
        center (tuple, optional): (x, y) coordinates of the center. Defaults to the center of the array.
        inner_radius (float): Inner radius of the annulus.
    
    Returns:
        numpy.ndarray: Boolean mask of the annular region.
    """
    rows, cols = array.shape[-2:]
    if center is None:
        center = (cols // 2, rows // 2)  # (x, y) convention
    cx, cy = center
    
    y, x = np.ogrid[:rows, :cols]
    distance = np.hypot(x - cx, y - cy)
    max_radius = min(cx, cy, cols - cx, rows - cy)
    return ((distance >= inner_radius) & (distance <= max_radius), (max_radius+inner_radius)//2)


def normalize_azimuthal_scan(input_array, mask, bin_width=1, center=None):
    """
    Normalize the azimuthal scan by partitioning the annular mask into concentric rings
    and normalizing the signal variation in each ring.
    
    Parameters:
        input_array (numpy.ndarray): Input 2D array of the diffraction pattern.
        mask (numpy.ndarray): Boolean annular mask (2D).
        bin_width (int, optional): Width of each annular partition. Defaults to 1.
        center (tuple, optional): (x, y) coordinates of the center. Defaults to the center of the array.
    
    Returns:
        numpy.ndarray: Normalized 2D array with the same shape as input_array.
    """
    # Ensure input is 2D
    if input_array.ndim != 2:
        raise ValueError("Input array must be 2D")
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D")
    if input_array.shape != mask.shape:
        raise ValueError("Input array and mask must have the same shape")
    
    rows, cols = input_array.shape
    if center is None:
        center = (cols // 2, rows // 2)  # (x, y) convention
    cx, cy = center
    
    # Apply mask to input array
    masked_array = input_array * mask
    normalized_array = np.copy(masked_array)
    
    # Compute distances from center
    y, x = np.ogrid[:rows, :cols]
    distances = np.hypot(x - cx, y - cy)
    
    # Iterate over radial bins
    max_radius = int(np.max(distances))
    for radius in range(0, max_radius, bin_width):
        # Create ring mask
        ring_mask = (distances >= radius) & (distances < radius + bin_width) & mask
        
        if np.any(ring_mask):
            # Compute mean and std within the ring
            ring_values = masked_array[ring_mask]
            mean_value = np.mean(ring_values)
            std_value = np.std(ring_values)
            
            # Avoid division by zero
            if std_value == 0:
                std_value = 1
            
            # Normalize values within the ring
            normalized_array[ring_mask] = (masked_array[ring_mask] - mean_value) / std_value
    
    return normalized_array


def extract_azimuthal_profile(normalized_array, mask, azimuthal_bin_width=1, center=None):
    """
    Extract a smooth 1D azimuthal profile using continuous angular bins.
    
    Parameters:
        normalized_array (np.ndarray): Input 2D array of the diffraction pattern.
        mask (np.ndarray): Boolean annular mask (2D).
        azimuthal_bin_width (float): Angular bin width in degrees.
        center (tuple, optional): (x, y) center coordinates. Defaults to the center of the array.
    
    Returns:
        tuple: A tuple containing:
            - azimuthal_profile (np.ndarray): 1D array of shape (360,).
            - binmask (np.ndarray): 3D array of shape (H, W, 360) with continuous weights.
    """
    # Ensure input is 2D
    if normalized_array.ndim != 2:
        raise ValueError("Input array must be 2D")
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D")
    if normalized_array.shape != mask.shape:
        raise ValueError("Input array and mask must have the same shape")
    
    rows, cols = normalized_array.shape
    if center is None:
        center = (cols // 2, rows // 2)  # (x, y) convention
    cx, cy = center

    # Compute angles for all pixels [0, 360)
    y, x = np.ogrid[:rows, :cols]
    angles = np.rad2deg(np.arctan2(y - cy, x - cx)) % 360

    # Initialize binmask with continuous weights
    binmask = np.zeros((rows, cols, 360), dtype=np.float32)
    bin_centers = np.arange(360)
    half_width = azimuthal_bin_width / 2

    # Create continuous angular bins
    for i, center_angle in enumerate(bin_centers):
        # Calculate angular distance from center [-180, 180)
        delta = (angles - center_angle + 180) % 360 - 180
        
        # Weight = 1 within bin width, 0 outside (rectangular window)
        weights = np.where(np.abs(delta) <= half_width, 1.0, 0.0)
        
        # Apply mask and normalize
        binmask[..., i] = weights * mask / azimuthal_bin_width

    # Compute azimuthal profile by summing over angular bins
    azimuthal_profile = np.sum(normalized_array[..., np.newaxis] * binmask, axis=(0, 1))
    
    return azimuthal_profile, binmask


def high_pass_filter_azimuthal(azimuthal_profile, cutoff=1):
    """
    Apply a high-pass filter to a 1D azimuthal profile by removing the lowest-order Fourier coefficients.
    
    Parameters:
        azimuthal_profile (numpy.ndarray): Input 1D array of shape (360,).
        cutoff (int, optional): Number of lowest-frequency coefficients to remove (including DC). Defaults to 1.
    
    Returns:
        numpy.ndarray: High-pass filtered 1D azimuthal profile of shape (360,).
    """
    # Ensure input is 1D
    if azimuthal_profile.ndim != 1:
        raise ValueError("Input array must be 1D")
    if len(azimuthal_profile) != 360:
        raise ValueError("Input array must have exactly 360 elements")
    
    # Compute the real FFT
    fft_data = np.fft.rfft(azimuthal_profile)
    
    # Zero out the first `cutoff` coefficients (DC and low frequencies)
    fft_data[:cutoff] = 0
    
    # Inverse FFT to reconstruct the filtered signal
    filtered_profile = np.fft.irfft(fft_data, n=360)
    
    return filtered_profile


def detect_kikuchi_peaks(azimuthal_profile, min_prominence=0.1, min_distance=5):
    """
    Detect and rank potential Kikuchi band peaks in a 1D circular azimuthal profile.
    
    Parameters:
        azimuthal_profile (np.ndarray): Input 1D array of shape (360,).
        min_prominence (float): Minimum peak prominence for detection.
        min_distance (int): Minimum peak separation in degrees.
        
    Returns:
        list: Sorted list of (position, height) tuples, ranked by descending height.
    """
    # Ensure input is 1D
    if azimuthal_profile.ndim != 1:
        raise ValueError("Input array must be 1D")
    if len(azimuthal_profile) != 360:
        raise ValueError("Input array must have exactly 360 elements")

    # Pad profile to handle circular boundary
    pad_width = min_distance
    padded_profile = np.concatenate([azimuthal_profile[-pad_width:], 
                                    azimuthal_profile, 
                                    azimuthal_profile[:pad_width]])
    
    # Find peaks in padded profile
    peaks_padded, properties = find_peaks(
        padded_profile,
        prominence=min_prominence,
        distance=min_distance
    )
    
    # Convert positions to original indices
    peaks_original = (peaks_padded - pad_width) % len(azimuthal_profile)
    
    # Remove duplicate peaks from padding
    unique_peaks = np.unique(peaks_original)
    peak_heights = azimuthal_profile[unique_peaks]
    
    # Sort by descending height
    sorted_peaks = sorted(zip(unique_peaks, peak_heights),
                        key=lambda x: x[1], reverse=True)
    
    return np.asarray(sorted_peaks)


def plot_peaks_on_image(image, radius, peaks_angles, center=None, scatter_kwargs={'c': 'r', 's': 20}):
    """
    Plot detected peaks as points at specified radius from center in 2D image.
    
    Parameters:
        image (np.ndarray): 2D input array
        center (tuple): (x, y) center coordinates in pixels
        radius (float): Radial distance from center to plot peaks (pixels)
        peaks_angles (array-like): Peak positions in degrees (0-360)
        scatter_kwargs (dict): Keyword arguments for plt.scatter
    """
    # Convert angles to Cartesian coordinates
    rows, cols = image.shape[-2:]
    if center is None:
        center = (cols // 2, rows // 2)  # (x, y) convention
    theta_rad = np.deg2rad(np.asarray(peaks_angles))
    x = center[0] + radius * np.cos(theta_rad)
    y = center[1] + radius * np.sin(theta_rad)
    
    # Create plot
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.scatter(x, y, **scatter_kwargs)
    ax.set_axis_off()
    return fig, ax


def plot_optimal_crossing(image, optimal_crossing, inner_radius=20, scatter_kwargs={'c': 'r', 's': 20}):
    center = optimal_crossing['center']
    radius = optimal_crossing['radius']
    mask, _ = annular_mask(image, center=center, inner_radius=inner_radius)
    # Process data through pipeline
    normalized = normalize_azimuthal_scan(image, mask, center=center)

    fig, ax = plot_peaks_on_image(normalized, radius, optimal_crossing['peaks'][:, 0], center=center, scatter_kwargs=scatter_kwargs)
    angles = np.deg2rad(optimal_crossing['peaks'][:, 0])
    n = len(angles) // 2  # Half the total peaks

    # Convert polar to Cartesian coordinates
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)

    ax.scatter(x, y, color='red', label="Kikuchi Bands")  # Plot peak points
    # Plot the connections
    for idx, (x0, y0) in enumerate(zip(x, y)):
        ax.text(x0, y0, s=f"#{idx}\n{int(optimal_crossing['peaks'][idx, 0])}" + r'$\deg$',color='red')
    for i in range(n):
        ax.plot([x[i], x[i + n]], [y[i], y[i + n]], 'b--', lw=0.5)  # Connect paired peaks
    ax.scatter(optimal_crossing['crossings'][:, 0], optimal_crossing['crossings'][:, 1], s=20, label='Crossing')
    ax.scatter(center[0], center[1], marker='x', c='r', label='Mask center')
    plt.legend()



def find_optimal_crossings(candidate_peaks, n, center, radius, stability_ratio=0.5, max_samples=1000):
    """
    Find optimal Kikuchi band crossings by evaluating multiple peak sets.
    
    Parameters:
        candidate_peaks (list): (angle, height) tuples sorted descending by height
        n (int): Number of Kikuchi bands (each band has 2 peaks)
        center (tuple): (x, y) center coordinates in pixels
        radius (float): Radial distance of peaks from center
        stability_ratio (float): Ratio of strong peaks to keep fixed (0-1)
        max_samples (int): Maximum number of combinations to iterate (fallback to random sampling if exceeded)
        
    Returns:
        dict: Best solution containing crossings, peaks, and metrics
    """
    if len(candidate_peaks) < 2 * n:
        raise ValueError(f"Need at least {2 * n} candidate peaks, got {len(candidate_peaks)}")

    num_fixed = int(2 * n * stability_ratio)
    needed = 2 * n - num_fixed
    fixed = candidate_peaks[:num_fixed]
    remaining = candidate_peaks[num_fixed:]

    num_combinations = np.math.comb(len(remaining), needed) if len(remaining) >= needed else 0

    if num_combinations <= max_samples:
        # Try all combinations if feasible
        sample_sets = list(combinations(remaining, needed))
    else:
        # Sample random sets
        sample_sets = [tuple(np.random.choice(remaining, needed, replace=False)) for _ in range(max_samples)]

    best = {'scatter': np.inf, 'crossings': None, 'peaks': None}
    
    for selected in sample_sets:
        selected = np.vstack([fixed, selected])  # Combine fixed and selected peaks
        angle_sorted = sorted(selected, key=lambda x: x[0])

        try:
            crossings = calculate_line_crossings(angle_sorted, center, radius)
        except (ValueError, np.linalg.LinAlgError):
            continue

        if len(crossings) < 2:
            continue

        hull = ConvexHull(crossings)
        scatter_metric =  hull.volume  # Convex hull area for 2D points or np.sum(np.std(crossings, axis=0))

        if scatter_metric < best['scatter']:
            best = {
                'scatter': scatter_metric,
                'crossings': crossings,
                'peaks': np.asarray(angle_sorted),
                'hull': hull,
                'radius': radius,
                'num_crossings': len(crossings),
                'center': center
            }

    return best


def calculate_line_crossings(sorted_peaks, center, radius):
    """Calculate intersections for a given peak set (modified for stability)"""
    angles = [p[0] for p in sorted_peaks]
    if len(angles) % 2 != 0:
        raise ValueError("Peak list must contain even number of elements")
    
    n = len(angles) // 2
    theta = np.deg2rad(angles)
    X = np.column_stack([
        center[0] + radius * np.cos(theta),
        center[1] + radius * np.sin(theta)
    ])
    
    crossings = []
    for i in range(n):
        for j in range(i+1, n):
            # Get line pairs (i, i+n) and (j, j+n)
            Xi, Xip = X[i], X[i+n]
            Xj, Xjp = X[j], X[j+n]
            
            # Solve intersection
            A = np.column_stack([Xip - Xi, -(Xjp - Xj)])
            B = Xj - Xi
            try:
                t = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                continue
                
            if t[0] >= 0 and t[1] >= 0:  # Check intersection within segments
                crossings.append(Xi + t[0]*(Xip - Xi))
    
    return np.array(crossings)


def point_in_hull(point, hull):
    """
    Check point containment in convex hull and return centroid if outside.
    
    Parameters:
        point (np.ndarray): 2D point coordinates (x, y)
        hull (scipy.spatial.ConvexHull): Convex hull object
        
    Returns:
        Union[bool, np.ndarray]: 
            - True if point is inside or on the hull boundary
            - Hull centroid coordinates if point is outside
    """
    
    # Get hull vertices coordinates
    hull_points = hull.points[hull.vertices]
    
    # Calculate hull centroid
    centroid = np.mean(hull_points, axis=0)
    
    # Check point containment
    delaunay = Delaunay(hull_points)
    if delaunay.find_simplex(point) >= 0:
        return (True, centroid)
    else:
        return (False, centroid)
    

def analyze_kikuchi_pattern(input_array, initial_center=None, inner_radius=40, 
                           azimuthal_bin_width=5, n_bands=6, hp_cutoff=5,
                           min_prominence=0.5, min_distance=10, stability_ratio=0.8,
                           max_samples=10000,
                           max_iterations=10, tolerance=1.0):
    """
    Iterative Kikuchi pattern analysis with automatic center adjustment.
    
    Parameters:
        input_array (np.ndarray): 2D diffraction pattern
        initial_center (tuple): (x, y) initial center guess
        inner_radius (int): Inner radius for annular mask
        azimuthal_bin_width (int): Angular bin width for profile extraction
        n_bands (int): Expected number of Kikuchi bands
        hp_cutoff (int): High-pass filter cutoff frequency
        min_prominence (float): Peak detection prominence threshold
        min_distance (int): Minimum peak separation
        stability_ratio (float): Ratio of strong peaks to keep fixed
        max_iterations (int): Maximum center adjustment iterations
        tolerance (float): Convergence tolerance in pixels
        
    Returns:
        dict: Analysis results containing:
            - final_center: (x, y) coordinates
            - converged: Boolean indicating if solution stabilized
            - crossings: Optimal crossing points
            - hull: Convex hull of crossings
            - iterations: List of iteration data
    """
    # Initialize tracking variables
    current_center = initial_center if initial_center else \
        (input_array.shape[1]//2, input_array.shape[0]//2)
    iteration_data = []
    converged = False
    
    try:
        for iteration in range(max_iterations):
            # Create annular mask with current center
            mask, radius = annular_mask(input_array, center=current_center, inner_radius=inner_radius)
            # Process data through pipeline
            normalized = normalize_azimuthal_scan(input_array, mask, center=current_center)
            profile, _ = extract_azimuthal_profile(normalized, mask, 
                                                azimuthal_bin_width=azimuthal_bin_width,
                                                center=current_center)
            filtered = high_pass_filter_azimuthal(profile, cutoff=hp_cutoff)
            peaks = detect_kikuchi_peaks(filtered, min_prominence=min_prominence,
                                        min_distance=min_distance)
            
            trials = 0
            while len(peaks) < 2 * n_bands and trials<10:
                min_prominence = min_prominence*0.8
                peaks = detect_kikuchi_peaks(filtered, min_prominence=min_prominence,
                                        min_distance=min_distance)
                trials += 1

            # Find optimal crossings
            res = find_optimal_crossings(
                peaks, n=n_bands, 
                center=current_center, radius=radius,
                stability_ratio=stability_ratio, max_samples=max_samples
            )
                    
            # Store iteration data
            iteration_data.append(res)
            
            # Check center containment
            converged, containment = point_in_hull(current_center, res['hull'])
            new_center = containment
            if converged:
                break
                            
            # Check convergence
            if np.linalg.norm(np.array(new_center) - np.array(current_center)) < tolerance:
                converged = True
                break
                
            current_center = new_center
            
        return {
            'final_center': tuple(new_center),
            'converged': converged,
            'crossings': res['crossings'],
            'hull': res['hull'],
            'radius': res['radius'],
            'peaks': res['peaks'],
            'iterations': iteration_data,
            'n_iterations': iteration + 1,
        }
    except:
        if 'new_center' in locals():
            center = new_center
        elif 'current_center' in locals():
            center = current_center
        else:
            center = [np.nan, np.nan]
        return {
            'final_center': tuple(center),
            'converged': False,
            'iterations': iteration_data,
            'n_iterations': iteration + 1,
        }


def analyze_4d_kikuchi_patterns(input_4d, n_jobs=-1, **kwargs):
    """
    Process a 4D array of diffraction patterns in parallel with a progress bar.
    
    Parameters:
        input_4d (np.ndarray): 4D array of shape (..., height, width)
        n_jobs (int): Number of parallel jobs (-1 for all cores)
        **kwargs: Arguments to pass to analyze_kikuchi_pattern
        
    Returns:
        np.ndarray: Array of results with shape input_4d.shape[:-2]
    """
    # Validate input dimensions
    if input_4d.ndim != 4:
        raise ValueError("Input must be a 4D array (n, m, height, width)")
    
    # Reshape to 2D array of patterns
    original_shape = input_4d.shape
    patterns = input_4d.reshape(-1, *original_shape[-2:])
    
    # Create parallel processing function
    def process_pattern(pattern):
        return analyze_kikuchi_pattern(pattern, **kwargs)
    
    if n_jobs == 0:
        results = []
        for pattern in tqdm(patterns, desc="Processing Patterns", unit="pattern"):
            results.append(process_pattern(pattern))
    else:
        # Use tqdm to monitor joblib's Parallel execution
        with tqdm_joblib(tqdm(desc="Processing Patterns", total=len(patterns), unit="pattern")):
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_pattern)(pattern) for pattern in patterns
            )
    
    # Reshape results to match original 4D structure
    return np.array(results, dtype=object).reshape(original_shape[:-2])



import numpy as np

def create_peak_masks_and_data(results_4d, data_4d, peak_ids, 
                              inner_radius, outer_radius, angular_range):
    """
    Create masks and masked data for specified peaks in 4D data.
    
    Parameters:
        results_4d (np.ndarray): Results from analyze_4d_kikuchi_patterns
        data_4d (np.ndarray): Original 4D diffraction data (n, m, h, w)
        peak_ids (list): List of peak indices to create masks for
        inner_radius (float): Inner radius for masks in pixels
        outer_radius (float): Outer radius for masks in pixels
        angular_range (float): Angular range (+/- degrees) around each peak
        
    Returns:
        tuple: (masks, masked_data) where:
            - masks: 5D array of shape (n, m, num_peaks, h, w)
            - masked_data: 5D array of same shape with masked data values
    """
    # Validate input shapes
    if results_4d.shape[:2] != data_4d.shape[:2]:
        raise ValueError("First two dimensions of results_4d and data_4d must match")
    
    n, m = data_4d.shape[:2]
    h, w = data_4d.shape[2:]
    num_peaks = len(peak_ids)
    
    # Initialize output arrays
    masks = np.zeros((n, m, num_peaks, h, w), dtype=np.float32)
    masked_data = np.zeros_like(masks)
    
    # Create reusable grid
    y, x = np.ogrid[:h, :w]
    
    for i in range(n):
        for j in range(m):
            res = results_4d[i, j]
            
            # Skip unconverged results
            if not res.get('converged', False):
                continue

            center = res['final_center']
            peaks = res['peaks']
            
            # Convert center to array coordinates
            cx, cy = center[0], center[1]
            
            # Calculate radial distances (reused for all peaks)
            dx = x - cx
            dy = y - cy
            distances = np.hypot(dx, dy)
            radial_mask = (distances >= inner_radius) & (distances <= outer_radius)
            
            # Calculate angles (reused for all peaks)
            angles = np.degrees(np.arctan2(dy, dx)) % 360
            
            for k, pid in enumerate(peak_ids):
                # Verify valid peak ID
                if pid >= len(peaks):
                    continue
                
                # Get peak angle and calculate angular mask
                peak_angle = peaks[pid, 0]
                angle_diff = np.abs(angles - peak_angle)
                angle_diff = np.minimum(angle_diff, 360 - angle_diff)
                angular_mask = angle_diff <= angular_range
                
                # Combine masks
                full_mask = radial_mask & angular_mask
                
                # Store results
                masks[i, j, k] = full_mask
                masked_data[i, j, k] = data_4d[i, j] * full_mask
                
    return masks, masked_data


# Assuming 'images' is your (X, Y, 128, 128) array
def compute_centers_of_mass(images):
    X_dim, Y_dim, _, _ = images.shape
    centers = np.zeros((X_dim, Y_dim, 2))  # To store (row, col) for each image

    for i in range(X_dim):
        for j in range(Y_dim):
            centers[i, j] = center_of_mass(images[i, j])
    
    return np.rollaxis(centers, -1)  # Shape (2, X, Y,), where last dimension is (row, col)


def save_dict_array(dict_array, discard_keys=None, output_file='fittedCenter'):
    """
    Processes a 2D array of dictionaries by discarding specified keys from each dictionary.

    Parameters:
        dict_array (list of list of dict): The 2D array of dictionaries.
        discard_keys (list, optional): A list of keys to discard. Defaults to None.
        output_file (str, optional): File path to save the processed array as JSON. Defaults to None.
    
    Returns:
        list of list of dict: The processed 2D array of dictionaries.
    """

    if discard_keys is not None:
        # Process the 2D array of dictionaries
        processed_array = []
        for row in dict_array:
            new_row = []
            for d in row:
                # Create a new dictionary with only the keys not in discard_keys
                new_dict = {k: v for k, v in d.items() if k not in discard_keys}
                new_row.append(new_dict)
            processed_array.append(new_row)
    else:
        processed_array = dict_array
    np.save(f'{output_file}.npy', arr=processed_array, allow_pickle=True)


def rotate_uv(U, V, X):
    """
    Rotate U, V components back to the original coordinate system.
    
    Parameters:
    U, V : array-like
        Components in the rotated coordinate system.
    X : float
        Rotation angle in degrees (positive = clockwise, negative = counterclockwise).
    
    Returns:
    U', V' : array-like
        Components in the original coordinate system.
    """
    X_rad = np.radians(X)  # Convert degrees to radians
    
    # Rotation matrix (clockwise for positive X, counterclockwise for negative X)
    cos_x, sin_x = np.cos(X_rad), np.sin(X_rad)
    
    U_prime = cos_x * U + sin_x * V
    V_prime = -sin_x * U + cos_x * V
    
    return U_prime, V_prime



def vector_to_rgb(angle, absolute, max_abs):
    """
    Convert an angle and magnitude into an RGB color using HSV color space.

    Parameters:
        angle (float): Angle in radians.
        absolute (float): Magnitude of the vector.
        max_abs (float): Maximum absolute value for normalization.
    
    Returns:
        tuple: (R, G, B) color values.
    """
    angle = angle % (2 * np.pi)
    if angle < 0:
        angle += 2 * np.pi
    return matplotlib.colors.hsv_to_rgb((angle / (2 * np.pi), 
                                         absolute / max_abs, 
                                       absolute / max_abs))


def rotate_uv(U, V, angle):
    """
    Rotate the vector field (U, V) by a given angle in degrees.
    
    Parameters:
        U (np.ndarray): X-component of the vector field.
        V (np.ndarray): Y-component of the vector field.
        angle (float): Rotation angle in degrees.
    
    Returns:
        tuple: Rotated (U, V) components.
    """
    theta = np.radians(angle)
    U_rot = U * np.cos(theta) - V * np.sin(theta)
    V_rot = U * np.sin(theta) + V * np.cos(theta)
    return U_rot, V_rot


def plot_vector_field(center, rotation_angle=None, max_abs=1, pixel_size=1, arrow_scale=100):
    """
    Plot a quiver plot from the given center data.
    
    Parameters:
        center (np.ndarray): 2D vector field with shape (2, height, width).
        rotation_angle (float): Angle to rotate vectors (default=5 degrees).
    """
    X = np.arange(center.shape[2])
    Y = np.arange(center.shape[1])
    U, V = center * pixel_size
    V = -V  

    if rotation_angle is not None:
        # Rotate the vector field
        U, V = rotate_uv(U, V, rotation_angle)
    
    # Compute angle and length
    angles = np.arctan2(V, U)
    lengths = np.sqrt(U**2 + V**2)

    # Normalize color scaling
    max_abs = np.nanpercentile(lengths, 90)
    c = np.array([vector_to_rgb(a, l, max_abs) for a, l in zip(angles.flatten(), lengths.flatten())])
    c = np.clip(c, 0, 1)  # Ensure values stay within [0, 1]

    # Quiver plot settings
    u, v = 2, 2
    length = np.sqrt(u**2 + v**2)
    width = 0.005
    hal = hl = 1. / width * length

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.quiver(X, Y, U, V, color=c.reshape(len(Y), len(X), 3), scale=arrow_scale, edgecolor=None, linewidth=0.8, pivot='mid',
              headaxislength=hal, headlength=hl, headwidth=hl / 3 * 2, width=width)
    
    ax.invert_yaxis()  # Flip the Y-axis visually


def imshow_vetor_field(center, rotation_angle=None, max_abs=1, pixel_size=1, show_arrow=True, arrow_scale=100, filter=False, low_val=1.5, high_val=10):
    # Assuming 'center' is your 2D vector field data
    X = np.arange(center.shape[2])
    Y = np.arange(center.shape[1])
    U, V = center * pixel_size  # Ensure U and V are 2D arrays
    V = -V  # If flipping V is necessary

    if rotation_angle is not None:
        # Rotate the vector field
        U, V = rotate_uv(U, V, rotation_angle)
    
    if filter:
        U = difference_of_gaussians(U, low_val, high_val)
        V = difference_of_gaussians(V, low_val, high_val)
    # Calculate angles and magnitudes
    angles = np.arctan2(V, U)
    lengths = np.sqrt(U**2 + V**2)

    # Generate RGB colors
    c = np.array([vector_to_rgb(a, l, max_abs) for a, l in zip(angles.flatten(), lengths.flatten())])
    c = np.clip(c, 0, 1)  # Ensure color values are valid
    c_2d = c.reshape(len(Y), len(X), 3)  # Reshape to 2D grid
    c_2d[np.isnan(U)] = 1
    # Create figure and plot
    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot colored cells using imshow
    ax.imshow(c_2d, origin='lower', 
            extent=[-0.5, len(X)-0.5, -0.5, len(Y)-0.5], 
            aspect='equal')
    if show_arrow:
        lengths[lengths == 0] = 1  # Avoid division by zero
        U_norm = U / lengths
        V_norm = V / lengths

        # Define arrow properties
        width = 0.005
        arrow_length = 1  # Keeps all arrows the same size
        hal = hl = 1. / width * arrow_length  

        # Plot arrows with normalized direction
        ax.quiver(X, Y, U_norm, V_norm, color='white', scale=arrow_scale, 
                pivot='mid', width=width, headaxislength=hal, 
                headlength=hl, headwidth=hl / 3 * 2)
    ax.invert_yaxis()  # Maintain original Y-axis orientation
    plt.show()

# import cupy as cp
# from cupyx.scipy.fft import rfft, irfft
# from cupyx.scipy.ndimage import uniform_filter as gpu_uniform_filter

# class KikuchiAnalyzerGPU:
#     def __init__(self, pattern_shape, inner_radius=40, bin_width=5):
#         self.pattern_shape = pattern_shape
#         self.inner_radius = inner_radius
#         self.bin_width = bin_width
        
#         # Preallocate GPU memory
#         self.d_input = cp.empty(pattern_shape, dtype=cp.float32)
#         self.d_mask = cp.empty(pattern_shape, dtype=cp.bool_)
#         self.d_normalized = cp.empty(pattern_shape, dtype=cp.float32)
#         self.d_profile = cp.empty(360, dtype=cp.float32)
        
#         # Precompute grid/mask templates
#         y, x = cp.ogrid[:pattern_shape[0], :pattern_shape[1]]
#         self.d_grid = cp.stack([x, y], axis=2)
        
#     def annular_mask_gpu(self, center):
#         """GPU-accelerated annular mask creation"""
#         vectors = self.d_grid - cp.array(center)
#         distances = cp.linalg.norm(vectors, axis=2)
#         mask = (distances >= self.inner_radius) & (distances <= self.inner_radius*2)
#         return mask, distances.max()

#     def normalize_azimuthal_gpu(self, d_input, mask):
#         """GPU-accelerated normalization"""
#         masked = d_input * mask
#         normalized = cp.empty_like(masked)
        
#         # Vectorized radial processing
#         max_radius = int(cp.max(cp.linalg.norm(self.d_grid - cp.array(center), axis=(0,1))))
#         radial_bins = cp.arange(0, max_radius, self.bin_width)
        
#         for r in radial_bins:
#             ring_mask = (distances >= r) & (distances < r + self.bin_width) & mask
#             if cp.any(ring_mask):
#                 ring_vals = masked[ring_mask]
#                 mean = cp.mean(ring_vals)
#                 std = cp.std(ring_vals)
#                 normalized[ring_mask] = (masked[ring_mask] - mean) / (std + 1e-9)
                
#         return normalized

#     def extract_profile_gpu(self, d_normalized, mask, center):
#         """GPU-accelerated azimuthal profile extraction"""
#         vectors = self.d_grid - cp.array(center)
#         angles = cp.degrees(cp.arctan2(vectors[:,:,1], vectors[:,:,0])) % 360
#         angles = angles.astype(cp.int32)
        
#         # Use CuPy's bincount for fast summation
#         weights = mask.astype(cp.float32)
#         profile = cp.bincount(angles.ravel(), weights=(d_normalized*weights).ravel(), minlength=360)
#         return profile / cp.bincount(angles.ravel(), weights=weights.ravel(), minlength=360)

#     def high_pass_filter_gpu(self, profile, cutoff=5):
#         """GPU-accelerated high-pass filtering"""
#         fft = rfft(profile)
#         fft[:cutoff] = 0
#         return irfft(fft, n=360).real

#     def detect_peaks_gpu(self, profile, min_prominence=0.1, min_distance=5):
#         """GPU-accelerated peak detection using difference of Gaussians"""
#         # Custom GPU peak finding implementation
#         smoothed = cp.convolve(profile, cp.hanning(10), mode='same')
#         grad = cp.gradient(smoothed)
#         zero_cross = cp.where(cp.diff(cp.sign(grad)))[0]
        
#         # Filter by prominence and distance
#         prominences = smoothed[zero_cross] - cp.minimum(smoothed[zero_cross-1], smoothed[zero_cross+1])
#         valid = (prominences > min_prominence) & (cp.diff(zero_cross) > min_distance)
#         peaks = zero_cross[valid]
        
#         return sorted(zip(peaks.get(), profile[peaks].get()), key=lambda x: -x[1])

#     def analyze_pattern(self, pattern, initial_center=None, n_bands=6, max_iter=10):
#         """Complete GPU analysis pipeline for a single pattern"""
#         # Transfer data to GPU
#         cp.copyto(self.d_input, cp.asarray(pattern))
        
#         # Initialize center
#         center = initial_center or (pattern.shape[1]//2, pattern.shape[0]//2)
        
#         for _ in range(max_iter):
#             # GPU processing steps
#             mask, radius = self.annular_mask_gpu(center)
#             self.d_normalized = self.normalize_azimuthal_gpu(self.d_input, mask)
#             self.d_profile = self.extract_profile_gpu(self.d_normalized, mask, center)
#             filtered = self.high_pass_filter_gpu(self.d_profile)
#             peaks = self.detect_peaks_gpu(filtered)
            
#             # Hybrid CPU processing for geometric validation
#             cpu_peaks = sorted(peaks, key=lambda x: x[0])
#             crossings = self.calculate_crossings_cpu(cpu_peaks, center, radius)
            
#             # Check convergence (requires CPU processing)
#             converged, new_center = self.check_convergence(center, crossings)
#             if converged:
#                 break
                
#             center = new_center
            
#         return {
#             'center': center,
#             'crossings': crossings,
#             'peaks': peaks,
#             'converged': converged
#         }

#     def analyze_4d_gpu(self, data_4d, batch_size=16, n_jobs=1):
#         """Process 4D dataset with GPU batching"""
#         results = []
#         for batch in tqdm(np.array_split(data_4d, len(data_4d)//batch_size)):
#             # Transfer batch to GPU
#             d_batch = cp.asarray(batch)
#             # Process batch in parallel streams
#             with cp.cuda.Stream() as stream:
#                 batch_results = [self.analyze_pattern(p) for p in d_batch]
#                 stream.synchronize()
#             results.extend([{k: cp.asnumpy(v) for k,v in res.items()} for res in batch_results])
        
#         return np.array(results).reshape(data_4d.shape[:-2])

#     # CPU fallbacks for non-GPU compatible operations
#     def calculate_crossings_cpu(self, peaks, center, radius):
#         """Calculate intersections for a given peak set (modified for stability)"""
#         angles = [p[0] for p in peaks]
#         if len(angles) % 2 != 0:
#             raise ValueError("Peak list must contain even number of elements")
        
#         n = len(angles) // 2
#         theta = np.deg2rad(angles)
#         X = np.column_stack([
#             center[0] + radius * np.cos(theta),
#             center[1] + radius * np.sin(theta)
#         ])
        
#         crossings = []
#         for i in range(n):
#             for j in range(i+1, n):
#                 # Get line pairs (i, i+n) and (j, j+n)
#                 Xi, Xip = X[i], X[i+n]
#                 Xj, Xjp = X[j], X[j+n]
                
#                 # Solve intersection
#                 A = np.column_stack([Xip - Xi, -(Xjp - Xj)])
#                 B = Xj - Xi
#                 try:
#                     t = np.linalg.solve(A, B)
#                 except np.linalg.LinAlgError:
#                     continue
                    
#                 if t[0] >= 0 and t[1] >= 0:  # Check intersection within segments
#                     crossings.append(Xi + t[0]*(Xip - Xi))
        
#         return np.array(crossings)


#     def check_convergence(self, center, crossings):
#         """
#         Check point containment in convex hull and return centroid if outside.
        
#         Parameters:
#             point (np.ndarray): 2D point coordinates (x, y)
#             hull (scipy.spatial.ConvexHull): Convex hull object
            
#         Returns:
#             Union[bool, np.ndarray]: 
#                 - True if point is inside or on the hull boundary
#                 - Hull centroid coordinates if point is outside
#         """
        
#         # Get hull vertices coordinates
#         hull_points = hull.points[hull.vertices]
        
#         # Calculate hull centroid
#         centroid = np.mean(hull_points, axis=0)
        
#         # Check point containment
#         delaunay = Delaunay(hull_points)
#         if delaunay.find_simplex(point) >= 0:
#             return (True, centroid)
#         else:
#             return (False, centroid)
        
        