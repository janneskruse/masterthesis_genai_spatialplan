"""
===============================================================================
Classification of building structural types based on OSM data for the region. 
================================================================================
"""
##### Import libraries ######
# System
# import time

# Data handling
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import momepy


def classify_building_shapes(
    osm_gdf: gpd.GeoDataFrame,
    buildings_xr: xr.DataArray,
    street_blocks_gdf: gpd.GeoDataFrame,
    shapes_gdf: gpd.GeoDataFrame,
    utm_crs: str
    ) -> gpd.GeoDataFrame:
    """
    Classify building shapes into structural types based on OSM data.
    
    Parameters:
    - osm_gdf: GeoDataFrame containing OSM data with building and street information.
    - buildings_xr: xarray DataArray containing building shapes.
    - street_blocks_gdf: GeoDataFrame containing street block information.
    - shapes_gdf: GeoDataFrame containing building shape geometries.
    - utm_crs: Coordinate reference system to use for spatial operations (e.g., "EPSG:32633").
    
    Returns:
    - gpd.GeoDataFrame with an additional column "structural_type" indicating the classified type.
    """
    
    # =============================
    # Define the datasets
    # =============================
    buildings_gdf = osm_gdf[osm_gdf["building"].notnull()].copy()
    residential_gdf  = osm_gdf[osm_gdf["landuse"] == "residential"].copy()

    streets_gdf = osm_gdf[osm_gdf["highway"].notnull()].copy()
    streets_gdf = streets_gdf[streets_gdf.geometry.type == "LineString"]
    
    # Remove columns with more than 50% NaNs
    streets_gdf = streets_gdf.dropna(axis=1, thresh=len(streets_gdf) * 0.5)
    
    exclude_types = ['cycleway', 'path', 'pedestrian', 'service', 'footway', 
                        'construction', 'track', 'steps', 'bridleway', 'corridor', 
                        'elevator', 'platform']
    main_streets_gdf = streets_gdf[~streets_gdf["highway"].isin(exclude_types)]

    include_types = ['footway', 'service', 'living_street']
    building_streets_gdf = streets_gdf[streets_gdf["highway"].isin(include_types)]
    
    
    # =============================
    # Process the attributes
    # =============================
    
    # assign residential label to shapes that intersect with residential landuse
    residential_shapes_gdf = gpd.sjoin(shapes_gdf, residential_gdf, how="inner", predicate="intersects")
    residential_shapes_gdf = residential_shapes_gdf[["geometry", "label"]]
    residential_shapes_gdf["residential"] = 1
    # drop duplicates in residential_shapes_gdf
    residential_shapes_gdf = residential_shapes_gdf.drop_duplicates(subset="label")
    # merge residential column back to shapes_gdf
    shapes_gdf = shapes_gdf.merge(residential_shapes_gdf[["label", "residential"]], left_on="label", right_on="label", how="left") 
    
    
    # Convert GDFs to UTM
    main_streets_gdf = main_streets_gdf.to_crs(utm_crs)
    building_streets_gdf = building_streets_gdf.to_crs(utm_crs)
    street_blocks_gdf = street_blocks_gdf.to_crs(utm_crs)
    shapes_gdf = shapes_gdf.to_crs(utm_crs)
    buildings_gdf = buildings_gdf.to_crs(utm_crs)
    building_heights_gdf = building_heights_gdf.to_crs(utm_crs)

    # Buildings crossed by streets
    streets_in_buildings = gpd.sjoin(
        building_streets_gdf[["geometry"]],
        buildings_gdf[["geometry"]],
        how="inner",
        predicate="crosses",
    )
    crossed_indices = streets_in_buildings["index_right"].unique()
    buildings_gdf["crossed_by_street"] = buildings_gdf.index.isin(crossed_indices)

    # Nearest street for every building
    buildings_gdf["street_index"] = momepy.get_nearest_street(buildings_gdf, main_streets_gdf)

    # Building count and street count
    bld_in_shapes = gpd.sjoin(
        buildings_gdf[["geometry", "street_index", "crossed_by_street"]],
        shapes_gdf[["geometry"]],
        how="inner",
        predicate="intersects",
    )
    building_count = bld_in_shapes.groupby("index_right").size().rename("building_count")
    street_count = bld_in_shapes.groupby("index_right")["street_index"].nunique().rename("street_count")
    crossed_count = bld_in_shapes.groupby("index_right")["crossed_by_street"].sum().rename("buildings_crossed_by_street")

    shapes_gdf["building_count"] = shapes_gdf.index.map(building_count).fillna(0).astype(int)
    shapes_gdf["street_count"] = shapes_gdf.index.map(street_count).fillna(0).astype(int)
    shapes_gdf["buildings_crossed_by_street"] = shapes_gdf.index.map(crossed_count).fillna(0).astype(int)

    # Max building height per shape
    heights_in_shapes = gpd.sjoin(
        building_heights_gdf[["geometry", "height"]],
        shapes_gdf[["geometry"]],
        how="inner",
        predicate="intersects",
    )
    max_height = heights_in_shapes.groupby("index_right")["height"].max().rename("max_building_height")
    shapes_gdf["max_building_height"] = shapes_gdf.index.map(max_height)

    # Morphometric statistics
    shapes_gdf["convexity"] = momepy.convexity(shapes_gdf)
    shapes_gdf["elongation"] = momepy.elongation(shapes_gdf)
    shapes_gdf["area"] = shapes_gdf.geometry.area


    # Mean convexity of nearest shapes within 100m
    shapes_buffered = shapes_gdf[["geometry"]].copy()
    shapes_buffered["geometry"] = shapes_gdf.geometry.centroid.buffer(100)

    neighbors = gpd.sjoin(
        shapes_gdf[["geometry", "convexity"]],
        shapes_buffered,
        how="inner",
        predicate="intersects",
    )
    # exclude self-joins
    neighbors = neighbors[neighbors.index != neighbors["index_right"]]
    mean_neighbor_convexity = neighbors.groupby("index_right")["convexity"].mean().rename("mean_neighbor_convexity")
    shapes_gdf["mean_neighbor_convexity"] = shapes_gdf.index.map(mean_neighbor_convexity)



    # calculate convex hull of all buildings and compare to street block convex hull 
    street_blocks_gdf["block_convexity"] = momepy.convexity(street_blocks_gdf)
    #intersect buildings with street blocks to count number of buildings per block
    bld_in_blocks = gpd.sjoin(
        buildings_gdf[["geometry"]],
        street_blocks_gdf[["geometry"]],
        how="inner",
        predicate="intersects",
    )
    buildings_per_block = bld_in_blocks.groupby("index_right").size().rename("building_count")
    street_blocks_gdf["block_building_count"] = street_blocks_gdf.index.map(buildings_per_block).fillna(0).astype(int)


    dense_block_idx = street_blocks_gdf[street_blocks_gdf["block_building_count"] >= 4].index

    # Aggregate building geometries per block -> union -> convex hull
    bld_hulls = (
        bld_in_blocks.loc[bld_in_blocks["index_right"].isin(dense_block_idx)]
        .groupby("index_right")["geometry"]
        .apply(lambda geoms: geoms.union_all().convex_hull.area)
        .rename("buildings_hull_area")
    )

    # Block convex hull area
    block_hull_area = street_blocks_gdf.loc[dense_block_idx, "geometry"].convex_hull.area.rename("block_hull_area")

    # Ratio: how much of the block's convex hull is filled by buildings' convex hull
    street_blocks_gdf["hull_ratio"] = (
        street_blocks_gdf.index.map(bld_hulls) / street_blocks_gdf.index.map(block_hull_area)
    )

    # Block area
    street_blocks_gdf["block_area"] = street_blocks_gdf.geometry.area

    # Map block attributes to shapes using a random point inside the shape
    shape_points = shapes_gdf[["geometry"]].copy()
    shape_points["geometry"] = shapes_gdf.geometry.sample_points(size=1, rng=42)

    shapes_to_blocks = gpd.sjoin(
        shape_points,
        street_blocks_gdf[["geometry", "hull_ratio", "block_building_count", "block_area"]],
        how="left",
        predicate="within",
    )

    # Count shapes per block
    block_shape_count = (
        shapes_to_blocks.dropna(subset=["index_right"])
        .groupby("index_right").size()
        .rename("block_shape_count")
    )
    shapes_to_blocks["block_shape_count"] = shapes_to_blocks["index_right"].map(block_shape_count)

    # Sum of all shape areas per block
    shapes_to_blocks["shape_area"] = shapes_to_blocks.index.map(shapes_gdf.geometry.area)

    block_total_shape_area = (
        shapes_to_blocks.dropna(subset=["index_right"])
        .groupby("index_right")["shape_area"]
        .sum()
        .rename("block_total_shape_area")
    )
    shapes_to_blocks["block_total_shape_area"] = shapes_to_blocks["index_right"].map(block_total_shape_area)
    shapes_to_blocks["block_shape_density"] = shapes_to_blocks["block_total_shape_area"] / shapes_to_blocks["block_area"]

    # De-duplicate (keep block with highest hull_ratio if centroid is on a boundary)
    block_attrs = (
        shapes_to_blocks
        .sort_values("hull_ratio", ascending=False)
        .groupby(level=0)
        .first()
        [["hull_ratio", "block_building_count", "block_shape_count", "block_shape_density"]]
    )

    shapes_gdf["hull_ratio"] = shapes_gdf.index.map(block_attrs["hull_ratio"]).fillna(0)
    shapes_gdf["block_building_count"] = shapes_gdf.index.map(block_attrs["block_building_count"]).fillna(0).astype(int)
    shapes_gdf["block_shape_count"] = shapes_gdf.index.map(block_attrs["block_shape_count"]).fillna(0).astype(int)
    shapes_gdf["block_shape_density"] = shapes_gdf.index.map(block_attrs["block_shape_density"]).fillna(0)
    
    # convert back to original CRS
    shapes_gdf = shapes_gdf.to_crs("EPSG:4326")
    
    # =============================
    # Classify to structural types
    # =============================
    
    shapes_gdf["structural_type"] = classify_shapes(shapes_gdf)

    print("Classification results:")
    print(shapes_gdf["structural_type"].value_counts())
    
    return shapes_gdf
    
def classify_shapes(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
        Classify building shapes into 
        urban design structural types ("Städtebauliche Strukturtypen") 
        using vectorized conditions.
        
        The classification is oriented on the "Städtebauliche Strukturtypen" defined by
        Heller (2010) and Reicher (2019) and adapted to the available data attributes.
        
    Parameters:
    - gdf: GeoDataFrame containing building shape geometries and attributes.
    
    Returns:
    - pd.Series with the classified structural type for each shape.
        
    Reference:
        Heller, M. (2010) Städtebauliche Strukturen. Lecture slides. 
            Institut für Raum- und Landschaftsentwicklung, Professur für Raumentwicklung, 
            available at: https://berndscholl.ch/wp-content/uploads/2018/07/St%C3%A4dtische-Strukturen.pdf.
        
        Reicher, C. (2019) Grundlagen, Bausteine und Aufgaben des Städtebaus: Schnelleinstieg für Architekten und Planer.
            Wiesbaden: Springer Vieweg. https://doi.org/10.1007/978-3-658-25659-3
    """
    
    # Pre-compute boolean masks
    has_crossing = gdf["buildings_crossed_by_street"] > 0
    is_residential = gdf["residential"] == 1
    
    # Conditions in priority order (first match wins)
    conditions = [
        
        # Zeile: elongated, few buildings, taller than single family
        (gdf["building_count"] <= 2) & (gdf["convexity"] > 0.7) & (gdf["max_building_height"] > 9) 
        & ((gdf["building_count"] > 1) & (gdf["elongation"] < 0.45) | (gdf["building_count"] == 1) & (gdf["elongation"] < 0.5)),
        
        # Passage: streets cross buildings, compact shape
        has_crossing & (gdf["convexity"] > 0.7),
        
        # Ensemble: streets cross buildings, irregular shape, building count < 10
        has_crossing & (gdf["convexity"] <= 0.7)& (gdf["building_count"] < 10),
        
        # Block: hull_ratio close to 1  and height > 8m
        (gdf["hull_ratio"] > 0.8) & (gdf["building_count"] >= 4) & (gdf["max_building_height"] > 8),
        
        # Wohnsiedlung: residential, compact, low-rise
        is_residential & (gdf["convexity"] > 0.65) 
        & (gdf["building_count"] <= 2) & ((gdf["max_building_height"] < 20) | gdf["max_building_height"].isna()),
        
        # Solitär: single building, compact
        (gdf["building_count"] == 1) & (gdf["convexity"] > 0.65) 
        & ((gdf["max_building_height"] < 20) | (gdf["area"] > 1000)),
        
        # Cluster: block shape count and max building count 10
        (gdf["block_shape_count"] >= 6) & (gdf["building_count"] < 10),
        
        # Hof: medium convexity, multiple street orientations
        (gdf["convexity"].between(0.45, 0.75)) & (gdf["street_count"] > 3),
        
        # Reihe: multiple buildings along few streets, compact
        (gdf["building_count"] > 2) 
        & ((gdf["street_count"] < 4) | (gdf["convexity"] > 0.65)),
        
        # Cluster: low convexity neighborhood
        (gdf["convexity"] < 0.65) & (gdf["mean_neighbor_convexity"] < 0.65),
    ]
    
    choices = [
        "Zeile",
        "Passage",
        "Ensemble",
        "Block",
        "Wohnsiedlung",
        "Solitär",
        "Cluster",
        "Hof",
        "Reihe",
        "Cluster",
    ]
    
    return np.select(conditions, choices, default="Cluster")