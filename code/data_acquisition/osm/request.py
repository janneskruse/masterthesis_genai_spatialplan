######## Script to aquire and pre-process OSM data for the region to an Xarray raster image cube #######

##### Import libraries ######
# system
import time
import logging
import traceback
import re
import random

# data manipulation
import overpass
import pandas as pd
import geopandas as gpd

# Suppress logs for the Overpass API requests
def suppress_overpass_logs():
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)
    logging.getLogger("requests").setLevel(logging.CRITICAL)
    logging.getLogger("osm2geojson").setLevel(logging.WARNING)

def restore_logs():
    # Restore the default logging level
    logging.getLogger("urllib3").setLevel(logging.NOTSET)
    logging.getLogger("requests").setLevel(logging.NOTSET)
    logging.getLogger("osm2geojson").setLevel(logging.NOTSET)
        
# Initialize the Overpass API
overpass_api = overpass.API(debug=False, timeout=900)

# Create Overpass query for tags
def create_query(bbox, tags):
    query_parts = []
    
    south, west, north, east = bbox[1], bbox[0], bbox[3], bbox[2]

    for tag, value in tags.items():
        osm_graph_type = "wr"  # nwr, node, way, etc.
        
        if tag in {"building","highway","railway","waterway"}:
            osm_graph_type = "way"
            
        if value is True:
            # If true, use tag
            query_parts.append(f'{osm_graph_type}["{tag}"]({south},{west},{north},{east});')
        elif isinstance(value, list):
            # If list, use a regex match for multiple possible values
            vals = [str(v) for v in value]
            if len(vals) == 1:
                v = vals[0]
                query_parts.append(f'{osm_graph_type}["{tag}"="{v}"]({south},{west},{north},{east});')
            else:
                pat = "|".join(re.escape(v) for v in vals)
                query_parts.append(f'{osm_graph_type}["{tag}"~"{pat}"]({south},{west},{north},{east});')
        else:
            # If single string, use an exact match
            query_parts.append(f'{osm_graph_type}["{tag}"="{value}"]({south},{west},{north},{east});')

            
    #join to a single query
    query = "("+"\n".join(query_parts) +");"
    # members = "(._;>>;);"
    # return query + members + "out body geom;"
    return query


def fetch_overpass_data(bbox, tags, retries=10, delay=2):
    
    query = create_query(bbox, tags)
    
    for attempt in range(retries):
        try:
            suppress_overpass_logs()
            response = overpass_api.Get(query, responseformat="geojson", verbosity='geom')
            restore_logs()
            
            # print(f"Fetched {len(response['features'])} features for bbox {bbox}")
            
            gdf = gpd.GeoDataFrame.from_features(response['features']) 
            
            #drop nodes column
            if 'nodes' in gdf.columns:
                gdf = gdf.drop(columns=['nodes'])

            #create columns for tags
            if 'tags' in gdf.columns:
                # new dataframe from tags
                expanded_tags = pd.json_normalize(gdf['tags'])
                
                # Only keep columns with at least 50% data for tags
                threshold = len(expanded_tags) * 1/len(tags)*0.5
                columns_to_drop = [col for col in expanded_tags.columns if 
                                expanded_tags[col].count() < threshold and 
                                col not in ["natural", "water", "boundary", "landuse", "building", "highway", "waterway", "leisure"]]
                expanded_tags = expanded_tags.drop(columns=columns_to_drop)
                gdf = gdf.drop(columns=['tags'])
                
                # concatenate with original dataframe
                gdf = pd.concat([gdf, expanded_tags], axis=1)
                
                # for duplicated column names, keep the first one
                gdf = gdf.loc[:,~gdf.columns.duplicated()]

            return gdf
        
        except overpass.errors.ServerLoadError as e:
            # server loaded, use return time to wait
            sleep_s = (e.args[0] if e.args else 10) + random.uniform(0.5, 2.0)
            print(f"ServerLoadError on attempt {attempt+1}. Retrying after delay.", sleep_s)
            time.sleep(sleep_s)

        except overpass.errors.MultipleRequestsError:
            print(f"MultipleRequestsError on attempt {attempt+1}. Retrying after delay.")
            # collided with another job, wait with jitter
            time.sleep(2.0 + random.uniform(0.2, 1.5))
        
        except Exception as e:
            etype = type(e).__name__
            args = getattr(e, "args", ())
            resp = getattr(e, "response", None)
            resp_info = ""
            if resp is not None:
                try:
                    resp_info = f" | HTTP {resp.status_code}: {resp.text[:500]}"
                except Exception:
                    pass
            print(f"Error fetching data for bbox {bbox}: {etype} args={args}{resp_info}")
            print(traceback.format_exc())
            time.sleep(delay)
            
    print(f"Failed to fetch data after {retries} attempts.")
    return None