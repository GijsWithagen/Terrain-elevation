import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.crs import CRS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import os
import random
import pydeck as pdk
from rasterio.windows import from_bounds, bounds as window_bounds
from rasterio.enums import Resampling as ResampleMethod
from rasterio.fill import fillnodata
from collections import defaultdict
from scipy.spatial import KDTree
import traceback


def build_cubical_complex(raster_path, bounding_box=None):
    """
    Builds a cubical complex from a GeoTIFF file.
    """
    with rasterio.open(raster_path) as src:
        dst_crs = CRS.from_epsg(4326)
        window = None
        window_bounds_src = src.bounds

        # Filter to only coordinates within the bounding box
        if bounding_box:
            min_lon, max_lon, min_lat, max_lat = bounding_box
            bbox_src_crs = transform_bounds(dst_crs, src.crs, min_lon, min_lat, max_lon, max_lat)
            window = from_bounds(*bbox_src_crs, transform=src.transform).round_offsets().round_shape()
            window_bounds_src = window_bounds(window, src.transform)

        data = src.read(1, window=window, masked=True)
        src_transform = src.window_transform(window) if window else src.transform

        src_nodata_val = src.nodatavals[0] if src.nodatavals and src.nodatavals[0] is not None else None
        nodata_for_reprojection = src_nodata_val if src_nodata_val is not None else np.nan
        dst_affine, dst_width, dst_height = calculate_default_transform(
            src.crs, dst_crs, data.shape[1], data.shape[0], *window_bounds_src
        )
        reprojected_data = np.full((dst_height, dst_width), nodata_for_reprojection, dtype=data.dtype)
        
        reproject(
            source=data, destination=reprojected_data, src_transform=src_transform,
            src_crs=src.crs, src_nodata=src_nodata_val, dst_transform=dst_affine,
            dst_crs=dst_crs, dst_nodata=nodata_for_reprojection, resampling=Resampling.nearest
        )

        vertices = {}
        valid_mask = ~np.isnan(reprojected_data) & (reprojected_data != nodata_for_reprojection)
        indices = np.argwhere(valid_mask)

        for i, j in indices:
            elevation = reprojected_data[i, j]

            if bounding_box:
                lon, lat = rasterio.transform.xy(dst_affine, i, j)

                if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
                    continue

            vertices[(i, j)] = float(elevation)

        return vertices, reprojected_data, dst_affine

def build_morse_complex(vertices, epsilon=1e-6):
    """
    Constructs a Morse complex, assigning a unique perturbed value to each cell.
    """
    f_v = {v: val + random.uniform(0, epsilon) for v, val in vertices.items()}

    edges = {}
    faces = {}

    for (i, j) in vertices:
        # Horizontal and vertical edges
        for di, dj in [(0, 1), (1, 0)]:
            ni, nj = i + di, j + dj
            neighbor = (ni, nj)
            if neighbor in vertices:
                edge = tuple(sorted([(i, j), neighbor]))
                if edge not in edges:
                    edges[edge] = max(f_v[(i, j)], f_v[neighbor]) + random.uniform(0, epsilon)

        v1 = (i, j)
        v2 = (i + 1, j)
        v3 = (i, j + 1)
        v4 = (i + 1, j + 1)
        if all(v in vertices for v in [v1, v2, v3, v4]):
            e1 = tuple(sorted([v1, v2]))
            e2 = tuple(sorted([v1, v3]))
            e3 = tuple(sorted([v2, v4]))
            e4 = tuple(sorted([v3, v4]))

            for e in [e1, e2, e3, e4]:
                if e not in edges:
                    edges[e] = max(f_v[e[0]], f_v[e[1]]) + random.uniform(0, epsilon)

            face_key = (i, j)
            face_edges = [e1, e2, e3, e4]
            faces[face_key] = {
                "value": max(edges[e] for e in face_edges) + random.uniform(0, epsilon),
                "edges": face_edges
            }

    return {
        "vertex_function": f_v,
        "edges": edges,
        "faces": faces
    }

def compute_gradient_field(morse_complex):
    """
    Computes the discrete gradient field by pairing cells.
    """
    f_v = morse_complex['vertex_function']
    edges = morse_complex['edges']
    faces = morse_complex['faces']

    # Sort all cells by Morse value
    all_vertices = [(v, 0, f_v[v]) for v in f_v]
    all_edges = [(e, 1, edges[e]) for e in edges]
    all_faces = [(f, 2, faces[f]['value']) for f in faces]
    all_cells = all_vertices + all_edges + all_faces
    all_cells.sort(key=lambda x: x[2])

    gradient_pairs = {}
    used_cells = set()
    vertex_to_edges = defaultdict(list)
    edge_to_faces = defaultdict(list)

    for edge in edges:
        v1, v2 = edge
        vertex_to_edges[v1].append(edge)
        vertex_to_edges[v2].append(edge)

    for face_key, face_data in faces.items():
        for edge in face_data['edges']:
            edge_to_faces[edge].append(face_key)

    # Process pairing
    for cell, dim, _ in all_cells:
        if cell in used_cells:
            continue

        if dim == 0:  # Vertex -> Edge
            unpaired_edges = [e for e in vertex_to_edges[cell] if e not in used_cells]
            if len(unpaired_edges) == 1:
                pair = unpaired_edges[0]
                gradient_pairs[cell] = pair
                used_cells.update([cell, pair])

        elif dim == 1:  # Edge -> Face
            unpaired_faces = [f for f in edge_to_faces[cell] if f not in used_cells]
            if len(unpaired_faces) == 1:
                pair = unpaired_faces[0]
                gradient_pairs[cell] = pair
                used_cells.update([cell, pair])

    # Extract critical cells
    paired_cells = set(gradient_pairs.keys()) | set(gradient_pairs.values())
    critical_0, critical_1, critical_2 = [], [], []

    for cell, dim, _ in all_cells:
        if cell not in paired_cells:
            if dim == 0:
                critical_0.append(cell)
            elif dim == 1:
                critical_1.append(cell)
            elif dim == 2:
                critical_2.append(cell)

    return {
        "critical_cells": {
            "0_minima": critical_0,
            "1_saddles": critical_1,
            "2_maxima": critical_2
        },
    }

def visualize_filtered_critical_points(
    vertices,
    reprojected_data,
    dst_affine,
    filtered_minima=None,
    filtered_maxima=None,
    output_html_file='output/critical_points_map.html',
    subsample_factor=5,
    z_scale_factor=1.5
):
    """
    Creates a 3D Pydeck map to visualize the terrain and its critical points.
    Only shows filtered minima and maxima.
    """
    
    all_points_data = []
    critical_points_data = []

    # Terrain sampling
    rows, cols = np.indices(reprojected_data.shape)
    rows_sub = rows[::subsample_factor, ::subsample_factor]
    cols_sub = cols[::subsample_factor, ::subsample_factor]
    elevations_sub = reprojected_data[::subsample_factor, ::subsample_factor]
    valid_mask = ~np.isnan(elevations_sub)

    # Get coordinates for terrain
    longitudes, latitudes = rasterio.transform.xy(dst_affine, rows_sub[valid_mask], cols_sub[valid_mask])
    elevations = elevations_sub[valid_mask]
    
    # Normalize colors using terrain colormap
    norm = matplotlib.colors.Normalize(vmin=np.percentile(elevations, 5), vmax=np.percentile(elevations, 95))
    cmap = plt.get_cmap('Greys')  

    for lon, lat, elev in zip(longitudes, latitudes, elevations):
        color_rgba = cmap(norm(elev))
        all_points_data.append({
            'longitude': lon,
            'latitude': lat,
            'elevation': elev * z_scale_factor,
            'color': (np.array(color_rgba[:3]) * 255).astype(int).tolist()
        })

    # Add filtered minima in blue
    if filtered_minima:
        for r, c in filtered_minima:
            lon, lat = rasterio.transform.xy(dst_affine, r, c)
            elev = vertices.get((r, c), 0)
            critical_points_data.append({
                'longitude': lon,
                'latitude': lat,
                'elevation': (elev + 2) * z_scale_factor,
                'type': 'Minima',
                'color': [0, 0, 255]
            })

    # Add filtered maxima in red
    if filtered_maxima:
        for r, c in filtered_maxima:
            lon, lat = rasterio.transform.xy(dst_affine, r + 0.5, c + 0.5)
            elev = vertices.get((r, c), 0)
            critical_points_data.append({
                'longitude': lon,
                'latitude': lat,
                'elevation': (elev + 2) * z_scale_factor,
                'type': 'Maxima',
                'color': [255, 0, 0]
            })

    df_terrain = pd.DataFrame(all_points_data)
    df_critical = pd.DataFrame(critical_points_data)

    # Pydeck layers
    terrain_layer = pdk.Layer(
        'PointCloudLayer',
        data=df_terrain,
        get_position='[longitude, latitude, elevation]',
        get_color='color',
        point_size=1
    )

    critical_points_layer = pdk.Layer(
        'ScatterplotLayer',
        data=df_critical,
        get_position='[longitude, latitude, elevation]',
        get_fill_color='color',
        get_radius=2,
        pickable=True
    )

    view_state = pdk.ViewState(
        longitude=df_terrain['longitude'].mean(),
        latitude=df_terrain['latitude'].mean(),
        zoom=15,
        pitch=50
    )

    tooltip = {"html": "<b>{type}</b>"}

    r = pdk.Deck(
        layers=[terrain_layer, critical_points_layer],
        initial_view_state=view_state,
        map_provider="mapbox",
        api_keys={'mapbox': os.environ.get('MAPBOX_API_KEY')},
        tooltip=tooltip
    )

    os.makedirs(os.path.dirname(output_html_file), exist_ok=True)
    r.to_html(output_html_file)
    print(f"3D map with filtered critical points saved to: {output_html_file}")

def pair_extrema_with_saddles(analysis_results, vertices, k_neighbors=5):
    """
    Pair extrema (minima, maxima) with nearby saddle points using spatial KDTree for performance. Only checks the k nearest neighbours.
    """
    minima = analysis_results['critical_cells'].get('0_minima', [])
    maxima = analysis_results['critical_cells'].get('2_maxima', [])
    saddles = analysis_results['critical_cells'].get('1_saddles', [])

    if not saddles:
        print("No saddle points.")
        return {'min_saddle_pairs': [], 'max_saddle_pairs': []}

    # Compute midpoints of saddle edges and store average location and elevation
    saddle_points_info = []
    saddle_coords_for_tree = []

    for s1_rc, s2_rc in saddles:
        # Calculate midpoint coordinates for the edge
        mid_r = (s1_rc[0] + s2_rc[0]) / 2
        mid_c = (s1_rc[1] + s2_rc[1]) / 2
        
        # Calculate average elevation for the edge
        mid_elev = (vertices.get(s1_rc, np.nan) + vertices.get(s2_rc, np.nan)) / 2
        
        if np.isnan(mid_elev):
            print(f"WARN: Could not get elevation for one or both vertices of saddle {s1_rc}-{s2_rc}. Skip")
            continue

        saddle_points_info.append({'rc': (mid_r, mid_c), 'elevation': mid_elev})
        saddle_coords_for_tree.append((mid_r, mid_c))

    if not saddle_coords_for_tree:
        print("No valid saddle coordinates.")
        return {'min_saddle_pairs': [], 'max_saddle_pairs': []}
        
    # KDTree for fast spatial lookup of coordinates
    tree = KDTree(saddle_coords_for_tree)

    def find_best_pair_for_extremum(extrema_list, kind='min'):
        """
        Helper function to find the best saddle pair for a list of extrema.
        """
        results = []
        for ext_rc in extrema_list:
            ext_r, ext_c = ext_rc
            ext_elev = vertices.get(ext_rc, np.nan)
            
            if np.isnan(ext_elev):
                print(f"Warning: Could not get elevation for extremum {ext_rc}. Skipping.")
                continue

            # Query k nearest saddles
            _, indices = tree.query((ext_r, ext_c), k=k_neighbors)

            best_saddle_info_idx = None 
            best_persistence_val = -1
            
            # Iterate over all k nearest neighbours and find the best one
            for i, idx in enumerate(indices):
                # Check if index is within bounds
                if idx >= len(saddle_points_info) or idx < 0:
                    continue
                
                current_saddle_info = saddle_points_info[idx]
                current_saddle_elev = current_saddle_info['elevation']
                current_persistence = abs(ext_elev - current_saddle_elev)

                # Check if current is better than current best
                is_suitable_saddle = False
                if kind == 'min':
                    if current_saddle_elev >= ext_elev:
                        if best_saddle_info_idx is None or current_saddle_elev < saddle_points_info[best_saddle_info_idx]['elevation']:
                            is_suitable_saddle = True
                elif kind == 'max':
                    if current_saddle_elev <= ext_elev:
                        if best_saddle_info_idx is None or current_saddle_elev > saddle_points_info[best_saddle_info_idx]['elevation']:
                            is_suitable_saddle = True
                
                # Update the best option if current is better
                if is_suitable_saddle:
                    best_saddle_info_idx = idx
                    best_persistence_val = current_persistence

            # Add the best option to the results
            if best_saddle_info_idx is not None:
                best_saddle_rc = saddle_points_info[best_saddle_info_idx]['rc'] 
                results.append((ext_rc, best_saddle_rc, best_persistence_val))
           
        print(f"Paired {len(results)} {kind} with saddles.")
        return results

    min_saddle_pairs = find_best_pair_for_extremum(minima, 'min')
    max_saddle_pairs = find_best_pair_for_extremum(maxima, 'max')

    return {
        'min_saddle_pairs': min_saddle_pairs,
        'max_saddle_pairs': max_saddle_pairs
    }

def filter_persistent_pairs(pairs, persistence_threshold=10):
    """
    Filters minima–saddle and maxima–saddle pairs based on persistence.
    """
    min_pairs = pairs.get('min_saddle_pairs', [])
    max_pairs = pairs.get('max_saddle_pairs', [])

    filtered_minima = [min_cell for (min_cell, _, persistence) in min_pairs if persistence >= persistence_threshold]
    filtered_maxima = [max_cell for (max_cell, _, persistence) in max_pairs if persistence >= persistence_threshold]

    return {
        'filtered_minima': filtered_minima,
        'filtered_maxima': filtered_maxima
    }

def find_min_max(raster_path, bounding_box=None, output=None):
    try:
        vertices, reprojected_data, dst_affine = build_cubical_complex(
            raster_path=raster_path,
            bounding_box=bounding_box
        )

        if not vertices:
            print("\nWARN: No vertices found")
            return

        # Perform the steps described by Celine.
        morse_complex = build_morse_complex(vertices)
        analysis_results = compute_gradient_field(morse_complex)
        paired_extrema = pair_extrema_with_saddles(analysis_results, vertices)

        # idea: store paired_extrema to json and try multiple persistence thresholds

        filtered = filter_persistent_pairs(paired_extrema, persistence_threshold=7) # Used 7 for now, looks good imo

        # Visualize the results on the map.
        visualize_filtered_critical_points(
            vertices,
            reprojected_data,
            dst_affine,
            filtered_minima=filtered['filtered_minima'],
            filtered_maxima=filtered['filtered_maxima'],
            output_html_file=f"output/{os.path.splitext(os.path.basename(raster_path))[0]}-min_max.html" if output is None else output
        )
    except FileNotFoundError:
        print(f"\nERROR: Input file not found at '{input_raster}'.")
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred: {e}")
        traceback.print_exc()

def fill_missing_values(raster_path):
    """
    Fills missing values in a raster file with the mean of the surrounding pixels.
    """
    with rasterio.open("datasets/dsm/" + raster_path) as src:
        data = src.read(1)
        msk = src.read_masks()

        
        filled_data = fillnodata(data, mask=msk, max_search_distance=100, smoothing_iterations=0)

        # Write the filled data back to the raster file
        with rasterio.open("datasets/dsm/FILL_" + raster_path, 'w', **src.meta) as dst:
            dst.write(filled_data, 1)
            dst.close()
        src.close()

if __name__ == '__main__':
    if not os.environ.get('MAPBOX_API_KEY'):
        os.environ['MAPBOX_API_KEY'] = 'pk.eyJ1IjoiZ2lqc3dpdGhhZ2VuIiwiYSI6ImNtYmF6bWw4OTA2Z3QyanNvcnFqd3NqZGMifQ.XKPEq0D0yqxZsMLJQMhTBw'
    
    input_raster = "2024_R_51GN1.TIF"

    atlas_bbox = (5.484705, 5.486861, 51.446943, 51.448714)
    campus_bbox= (5.482617, 5.494170, 51.445511, 51.450016)
    waters_man_bbox= (5.496532, 5.503750, 51.447776, 51.456504)
    
    fill_missing_values(input_raster)

    # find_min_max(input_raster, atlas_bbox, , "output/atlas_min_max.html")
    # find_min_max(input_raster, campus_bbox, "output/campus_min_max.html")
    find_min_max("datasets/dsm/FILL_" + input_raster, waters_man_bbox, "output/100inter_water_min_max.html")

    