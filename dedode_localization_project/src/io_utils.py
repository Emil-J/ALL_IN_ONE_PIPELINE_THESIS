"""
I/O utilities for loading and saving data
"""

import json
import csv
import pickle
import h5py
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import warnings


def load_json(filepath: Union[str, Path]) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: Dict, filepath: Union[str, Path], indent: int = 2):
    """Save data to JSON file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def load_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load CSV file as pandas DataFrame"""
    return pd.read_csv(filepath)


def save_csv(df: pd.DataFrame, filepath: Union[str, Path], index: bool = False):
    """Save DataFrame to CSV"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=index)


def append_to_csv(row_dict: Dict, filepath: Union[str, Path], write_header: bool = False):
    """Append a single row to CSV file (for streaming results)"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    mode = 'a' if filepath.exists() and not write_header else 'w'
    
    with open(filepath, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if write_header or mode == 'w':
            writer.writeheader()
        writer.writerow(row_dict)


def load_h5_reference_database(h5_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load reference database from H5 file and return as DataFrame
    
    Expected H5 structure (flexible adapter):
    - features: (N, D) feature vectors
    - metadata fields: lat, lon, tile_x, tile_y, file_path, etc.
    
    Returns:
        DataFrame with columns: file_path, tile_x, tile_y, lat, lon, bounds, features, etc.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 database not found: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        # Try to extract common fields
        data = {}
        
        # Features (required)
        if 'features' in f:
            data['features'] = f['features'][:]
        else:
            raise KeyError("H5 file must contain 'features' dataset")
        
        # Metadata fields (flexible - try multiple common names)
        metadata_mapping = {
            'lat': ['lat', 'latitude', 'center_lat'],
            'lon': ['lon', 'longitude', 'center_lon'],
            'tile_x': ['tile_x', 'x', 'col'],
            'tile_y': ['tile_y', 'y', 'row'],
            'zoom': ['zoom', 'zoom_level', 'z'],
            'file_path': ['file_path', 'path', 'filepath', 'image_path'],
        }
        
        for target_key, possible_keys in metadata_mapping.items():
            for key in possible_keys:
                if key in f:
                    raw_data = f[key][:]
                    # Decode bytes to strings if needed
                    if raw_data.dtype.type is np.bytes_:
                        data[target_key] = np.array([s.decode('utf-8') for s in raw_data])
                    else:
                        data[target_key] = raw_data
                    break
        
        # Optional bounds
        if 'bounds' in f:
            data['bounds'] = f['bounds'][:]
        
        # Store attributes
        attrs = dict(f.attrs)
        if attrs:
            data['_attrs'] = attrs
    
    # Convert to DataFrame
    df = pd.DataFrame({k: v for k, v in data.items() if k != 'features' and k != '_attrs'})
    
    # Store features separately (large array)
    if 'features' in data:
        # Add as column of arrays (or store separately if too large)
        df['features'] = [data['features'][i] for i in range(len(df))]
    
    return df


def save_h5_reference_database(df: pd.DataFrame, h5_path: Union[str, Path]):
    """
    Save reference database DataFrame to H5 file
    
    Expects DataFrame with columns: file_path, tile_x, tile_y, lat, lon, features, etc.
    """
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(h5_path, 'w') as f:
        # Save features as numpy array
        if 'features' in df.columns:
            features = np.vstack(df['features'].values)
            f.create_dataset('features', data=features, compression='gzip')
        
        # Save metadata fields
        for col in df.columns:
            if col == 'features':
                continue
            
            if df[col].dtype == object:
                # String data
                dt = h5py.string_dtype(encoding='utf-8')
                f.create_dataset(col, data=df[col].values.astype(str), dtype=dt)
            else:
                # Numeric data
                f.create_dataset(col, data=df[col].values)
        
        # Save metadata
        f.attrs['num_tiles'] = len(df)
        f.attrs['created'] = pd.Timestamp.now().isoformat()


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(data: Any, filepath: Union[str, Path]):
    """Save data to pickle file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def get_sorted_image_files(directory: Union[str, Path], 
                           pattern: str = "*.jpeg",
                           sort_by_name: bool = True) -> List[Path]:
    """
    Get sorted list of image files from directory
    
    Args:
        directory: Path to directory
        pattern: Glob pattern for matching files
        sort_by_name: If True, sort by filename
    
    Returns:
        Sorted list of Path objects
    """
    directory = Path(directory)
    files = list(directory.glob(pattern))
    
    if sort_by_name:
        files = sorted(files)
    
    return files


def extract_frame_number(filepath: Union[str, Path]) -> Optional[int]:
    """
    Extract frame number from filename like 'Capture00022756.jpeg'
    
    Returns:
        Frame number as int, or None if not found
    """
    filepath = Path(filepath)
    stem = filepath.stem
    
    # Try to extract number from common patterns
    import re
    match = re.search(r'(\d+)', stem)
    if match:
        return int(match.group(1))
    return None


def create_timestamped_filename(prefix: str, extension: str = "") -> str:
    """Create filename with timestamp prefix_YYYYMMDD_HHMMSS.extension"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if extension and not extension.startswith('.'):
        extension = f".{extension}"
    return f"{prefix}_{timestamp}{extension}"


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if needed"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_kml_to_dataframe(kml_path: Union[str, Path]) -> pd.DataFrame:
    """
    Parse KML file containing ground truth coordinates to DataFrame.
    
    Extracts Placemark entries with:
    - name: frame filename (e.g., "Capture00022760.jpeg")
    - coordinates: longitude,latitude,altitude
    
    Args:
        kml_path: Path to KML file
        
    Returns:
        DataFrame with columns: frame_name, latitude, longitude, altitude
        
    Example:
        >>> df = parse_kml_to_dataframe("coords_fixed_100m_east.kml")
        >>> df.head()
           frame_name                  latitude   longitude   altitude
        0  Capture00022760.jpeg       55.688826   9.516491    358.854
    """
    import xml.etree.ElementTree as ET
    
    kml_path = Path(kml_path)
    if not kml_path.exists():
        raise FileNotFoundError(f"KML file not found: {kml_path}")
    
    # Read KML file and remove malformed namespace prefixes (e.g., ns1: without declaration)
    # This handles cases where KML files have undeclared namespace prefixes
    with open(kml_path, 'r', encoding='utf-8') as f:
        kml_content = f.read()
    
    # Remove lines with undeclared namespace prefixes (ns1:, ns2:, etc.)
    kml_content = re.sub(r'<ns\d+:[^>]+>', '', kml_content)  # Remove <ns1:tag>
    kml_content = re.sub(r'</ns\d+:[^>]+>', '', kml_content)  # Remove </ns1:tag>
    
    # Parse cleaned XML
    root = ET.fromstring(kml_content)
    
    # KML uses namespace - handle it flexibly
    namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    # Try without namespace first (in case it's not used)
    try:
        placemarks = root.findall('.//Placemark')
        if not placemarks:  # If empty, try with namespace
            placemarks = root.findall('.//kml:Placemark', namespace)
    except:
        placemarks = root.findall('.//kml:Placemark', namespace)
    
    
    records = []
    for placemark in placemarks:
        # Get name (frame filename) - try without namespace first
        name_elem = placemark.find('.//name')
        if name_elem is None:
            name_elem = placemark.find('.//kml:name', namespace)
        if name_elem is None or not name_elem.text:
            continue
            
        frame_name = name_elem.text.strip()
        
        # Get coordinates (format: "longitude,latitude,altitude") - try without namespace first
        coords_elem = placemark.find('.//coordinates')
        if coords_elem is None:
            coords_elem = placemark.find('.//kml:coordinates', namespace)
        if coords_elem is None or not coords_elem.text:
            continue
            
        coords_text = coords_elem.text.strip()
        try:
            # Parse "lon,lat,alt"
            parts = coords_text.split(',')
            if len(parts) >= 2:
                longitude = float(parts[0])
                latitude = float(parts[1])
                altitude = float(parts[2]) if len(parts) > 2 else 0.0
                
                records.append({
                    "frame_name": frame_name,
                    "latitude": latitude,
                    "longitude": longitude,
                    "altitude": altitude
                })
        except (ValueError, IndexError) as e:
            warnings.warn(f"Failed to parse coordinates for {frame_name}: {coords_text} ({e})")
            continue
    
    if not records:
        raise ValueError(f"No valid placemarks found in {kml_path}")
    
    df = pd.DataFrame(records)
    print(f"Parsed {len(df)} ground truth coordinates from {kml_path.name}")
    
    return df


def convert_kml_to_csv(kml_path: Union[str, Path], 
                       csv_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Convert KML ground truth file to CSV format.
    
    Args:
        kml_path: Path to input KML file
        csv_path: Path to output CSV file (optional - auto-generated if None)
        
    Returns:
        Path to created CSV file
        
    Example:
        >>> csv_path = convert_kml_to_csv("coords_fixed_100m_east.kml")
        >>> print(csv_path)
        coords_fixed_100m_east_ground_truth.csv
    """
    kml_path = Path(kml_path)
    
    # Auto-generate CSV path if not provided
    if csv_path is None:
        csv_path = kml_path.parent / f"{kml_path.stem}_ground_truth.csv"
    else:
        csv_path = Path(csv_path)
    
    # Parse KML and convert to DataFrame
    df = parse_kml_to_dataframe(kml_path)
    
    # Save to CSV
    save_csv(df, csv_path, index=False)
    
    print(f"✅ Converted {len(df)} coordinates")
    print(f"   Input:  {kml_path}")
    print(f"   Output: {csv_path}")
    
    return csv_path
