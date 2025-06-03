import folium
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
import numpy as np
import pandas as pd # Added for 3D data handling
import pydeck # Added for 3D visualization
import matplotlib.pyplot as plt
import matplotlib.colors
import os
import pydeck as pdk

def create_overlay_image(data_array, cmap_name='terrain', nodata_val=None):
    """
    Converts a 2D numpy array (raster data) into an RGBA image array using a colormap.
    Nodata values are made transparent.

    Args:
        data_array (np.ndarray): The 2D input data.
        cmap_name (str): Name of the matplotlib colormap to use.
        nodata_val (float/int, optional): Value representing nodata in data_array. 
                                          These pixels will be made transparent.

    Returns:
        np.ndarray: An (height, width, 4) RGBA image array (uint8).
    """
    data_float = data_array.astype(np.float32)

    masked_data = np.ma.array(data_float, mask=False)
    
    if nodata_val is not None:
        if np.isnan(nodata_val):
            masked_data.mask = np.isnan(data_float)
        else: 
            masked_data.mask = (data_float == nodata_val)

    if masked_data.mask.all():
        return np.zeros((data_array.shape[0], data_array.shape[1], 4), dtype=np.uint8)

    valid_data = masked_data.compressed()
    if valid_data.size == 0: 
         return np.zeros((data_array.shape[0], data_array.shape[1], 4), dtype=np.uint8)

    vmin = np.min(valid_data)
    vmax = np.max(valid_data)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    rgba_data = cmap(norm(data_float))

    if nodata_val is not None and masked_data.mask.any(): 
         rgba_data[masked_data.mask, 3] = 0.0 

    rgba_data_uint8 = (rgba_data * 255).astype(np.uint8)
    
    return rgba_data_uint8

def display_tifs_on_map(tif_file_paths, output_html_file='elevation_map.html', 
                        map_cmap='terrain', map_opacity=0.7):
    netherlands_center = [52.1326, 5.2913]
    eindhoven_center = [51.439530, 5.478377]


    folium_map = folium.Map(location=eindhoven_center, zoom_start=13, tiles="OpenStreetMap")

    print(f"Processing {len(tif_file_paths)} TIF file(s)...")

    for tif_path in tif_file_paths:
        if not os.path.exists(tif_path):
            print(f"Warning: File not found, skipping: {tif_path}")
            continue
        
        try:
            print(f"Processing: {tif_path}")
            with rasterio.open(tif_path) as src:
                src_crs = src.crs
                dst_crs = CRS.from_epsg(4326)

                transform, width, height = calculate_default_transform(
                    src_crs, dst_crs, src.width, src.height, *src.bounds
                )
                
                src_nodata_val = src.nodatavals[0] if src.nodatavals and src.nodatavals[0] is not None else None
                
                if src_nodata_val is not None:
                    nodata_for_reprojection_and_overlay = src_nodata_val
                elif np.issubdtype(src.read(1).dtype, np.floating):
                    nodata_for_reprojection_and_overlay = np.nan 
                else: 
                    nodata_for_reprojection_and_overlay = 0 

                reprojected_array = np.full((height, width), 
                                            nodata_for_reprojection_and_overlay, 
                                            dtype=src.read(1).dtype)

                reproject(
                    source=rasterio.band(src, 1), 
                    destination=reprojected_array,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src_nodata_val, 
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    dst_nodata=nodata_for_reprojection_and_overlay, 
                    resampling=Resampling.nearest 
                )

                min_lon, min_lat, max_lon, max_lat = rasterio.transform.array_bounds(
                    height, width, transform
                )
                folium_bounds = [[min_lat, min_lon], [max_lat, max_lon]]

                overlay_image_rgba = create_overlay_image(
                    reprojected_array, 
                    cmap_name=map_cmap, 
                    nodata_val=nodata_for_reprojection_and_overlay # Pass the consistent nodata value
                )
                
                img_overlay = folium.raster_layers.ImageOverlay(
                    image=overlay_image_rgba, 
                    bounds=folium_bounds,
                    opacity=map_opacity,
                    name=os.path.basename(tif_path),
                    interactive=True,
                    cross_origin=False, 
                    zindex=1 
                )
                img_overlay.add_to(folium_map)
                print(f"Successfully added {os.path.basename(tif_path)} to map.")

        except Exception as e:
            print(f"Error processing file {tif_path}: {e}")
            import traceback
            traceback.print_exc()

    if len(folium_map._children) > 0: 
        folium.LayerControl().add_to(folium_map)
    
    try:
        folium_map.save(output_html_file)
        print(f"\nMap successfully saved to: {output_html_file}")
        print("You can open this HTML file in your web browser.")
    except Exception as e:
        print(f"Error saving map: {e}")


def create_3d_point_cloud_map(
        tif_file_paths, 
        output_html_file_3d='dsm_3d_map.html',
        map_cmap='terrain', 
        subsample_factor=20, 
        z_scale_factor=1,
        bounding_box=None
    ):
    all_points_data = []
    all_elevations_for_norm = []

    print(f"\nProcessing {len(tif_file_paths)} TIF file(s) for 3D map...")

    for tif_path in tif_file_paths:
        if not os.path.exists(tif_path):
            print(f"Warning (3D map): File not found, skipping: {tif_path}")
            continue
        try:
            print(f"Processing (3D map): {tif_path}")
            with rasterio.open(tif_path) as src:
                dst_crs = CRS.from_epsg(4326)
                dst_affine, dst_width, dst_height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )
                src_nodata_val = src.nodatavals[0] if src.nodatavals and src.nodatavals[0] is not None else None

                nodata_for_reprojection = (
                    src_nodata_val if src_nodata_val is not None
                    else (np.nan if np.issubdtype(src.read(1).dtype, np.floating) else 0)
                )

                reprojected_data = np.full((dst_height, dst_width), nodata_for_reprojection, dtype=src.read(1).dtype)

                reproject(
                    source=rasterio.band(src, 1),
                    destination=reprojected_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src_nodata_val,
                    dst_transform=dst_affine,
                    dst_crs=dst_crs,
                    dst_nodata=nodata_for_reprojection,
                    resampling=Resampling.nearest
                )

                rows, cols = np.indices(reprojected_data.shape)
                rows_sub = rows[::subsample_factor, ::subsample_factor].ravel()
                cols_sub = cols[::subsample_factor, ::subsample_factor].ravel()
                elevations_sub = reprojected_data[rows_sub, cols_sub]

                valid_mask = ~np.isnan(elevations_sub) if np.isnan(nodata_for_reprojection) else (elevations_sub != nodata_for_reprojection)

                rows_sub_valid = rows_sub[valid_mask]
                cols_sub_valid = cols_sub[valid_mask]
                elevations_sub_valid = elevations_sub[valid_mask]

                longitudes, latitudes = rasterio.transform.xy(dst_affine, rows_sub_valid, cols_sub_valid)

                for lon, lat, elev in zip(longitudes, latitudes, elevations_sub_valid):
                    if bounding_box:
                        min_lon, max_lon, min_lat, max_lat = bounding_box
                        if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
                            continue  # Skip points outside bounding box
                    all_points_data.append({
                        'longitude': lon,
                        'latitude': lat,
                        'elevation_raw': elev
                    })
                    all_elevations_for_norm.append(elev)

            print(f"Finished processing {os.path.basename(tif_path)} for 3D map.")
        except Exception as e:
            print(f"Error processing file {tif_path} for 3D map: {e}")
            import traceback
            traceback.print_exc()

    valid_elev_array = np.array(all_elevations_for_norm)
    if valid_elev_array.size == 0:
        print("No valid elevation values found after processing all files. Cannot create 3D map.")
        return

    vmin_color = np.percentile(valid_elev_array, 2)
    vmax_color = np.percentile(valid_elev_array, 98)
    if vmin_color == vmax_color:
        vmin_color -= 0.5
        vmax_color += 0.5

    norm_elev_for_color = matplotlib.colors.Normalize(vmin=vmin_color, vmax=vmax_color)
    cmap_3d = plt.get_cmap(map_cmap)

    for point in all_points_data:
        color_rgba_float = cmap_3d(norm_elev_for_color(point['elevation_raw']))
        point['color'] = (np.array(color_rgba_float[:3]) * 255).astype(np.uint8).tolist()
        point['elevation'] = point['elevation_raw'] * z_scale_factor

    df_3d = pd.DataFrame(all_points_data)
    if df_3d.empty:
        print("DataFrame for 3D map is empty. Cannot generate 3D map.")
        return

    point_cloud_layer = pdk.Layer(
        'PointCloudLayer',
        data=df_3d,
        get_position='[longitude, latitude, elevation]',
        get_color='color',
        get_normal=[0, 0, 1],
        auto_highlight=True,
        pickable=True,
        point_size=3
    )

    initial_view_state = pdk.ViewState(
        longitude=df_3d['longitude'].mean(),
        latitude=df_3d['latitude'].mean(),
        zoom=8,
        pitch=45,
        bearing=0
    )

    tooltip = {
        "html": "<b>Elevation:</b> {elevation_raw} m",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }

    try:
        r = pdk.Deck(
            layers=[point_cloud_layer],
            initial_view_state=initial_view_state,
            map_provider="mapbox",
            api_keys={
                'mapbox': os.environ.get('MAPBOX_API_KEY')
            },
            tooltip=tooltip,
        )
        r.to_html(output_html_file_3d)
        print(f"3D Point Cloud Map successfully saved to: {output_html_file_3d}")
    except Exception as e:
        print(f"Error creating or saving 3D map with Pydeck: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    if not os.environ.get('MAPBOX_API_KEY'):
        print("environ 'MAPBOX_API_KEY' not set")
        exit(1)

    tif_dsm_files = [
        "datasets/dsm/2024_R_51BZ2.TIF",
        "datasets/dsm/2024_R_51DN2.TIF",
        "datasets/dsm/2024_R_51EZ1.TIF",
        "datasets/dsm/2024_R_51GN1.TIF",
    ]

    tif_dtm_files = [
        "datasets/dtm/2024_M5_51BZ2.TIF",
        "datasets/dtm/2024_M5_51DN2.TIF",
        "datasets/dtm/2024_M5_51EZ1.TIF",
        "datasets/dtm/2024_M5_51GN1.TIF",
    ]

    # display_tifs_on_map(
    #     tif_dsm_files, 
    #     output_html_file='output/netherlands_dsm_2d_map.html', 
    #     map_cmap='terrain', 
    #     map_opacity=0.75
    # )

    # display_tifs_on_map(
    #     tif_dtm_files, 
    #     output_html_file='output/netherlands_dtm_2d_map.html', 
    #     map_cmap='terrain', 
    #     map_opacity=0.75
    # )
    
    create_3d_point_cloud_map(
        [
         "datasets/dsm/2024_R_51GN1.TIF",
        ],
        output_html_file_3d='output/atlas_3d.html',
        map_cmap='terrain',
        subsample_factor=1, 
        z_scale_factor=1,
        bounding_box=(5.484705, 5.486861, 51.446943, 51.448714)
    )

    create_3d_point_cloud_map(
        [
         "datasets/dsm/2024_R_51GN1.TIF",
        ],
        output_html_file_3d='output/campus_3d.html',
        map_cmap='terrain',
        subsample_factor=1, 
        z_scale_factor=1,
        bounding_box=(5.482617, 5.494170, 51.445511,51.450016)
    )

    # create_3d_point_cloud_map(
    #     tif_dtm_files,
    #     output_html_file_3d='output/netherlands_dtm_3d_map.html',
    #     map_cmap='terrain',
    #     subsample_factor=20, 
    #     z_scale_factor=1 
    # )