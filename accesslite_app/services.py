import osmnx as ox
import math
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import networkx as nx
import psycopg2
import json
from shapely.geometry import mapping


def epsg_calc(lat, lon):
    try:
        # Validate input coordinates
        if not (-90 <= lat <= 90):
            raise ValueError(f"Invalid latitude: {lat}")
        if not (-180 <= lon <= 180):
            raise ValueError(f"Invalid longitude: {lon}")

        # Create a GeoDataFrame for the given point
        point = gpd.GeoDataFrame(
            {'geometry': [Point(lon, lat)]},
            crs="EPSG:4326"  # WGS84
        )
        # Estimate the best UTM CRS
        best_crs = point.estimate_utm_crs()
        # Extract the EPSG code as an integer
        epsg_code = best_crs.to_epsg()
        if epsg_code is None:
            raise ValueError("Could not determine a valid EPSG code from the UTM CRS")
        print(f"Best UTM CRS determined: EPSG:{epsg_code}")
        return epsg_code

    except Exception as e:
        print(f"Error determining EPSG code: {e}")
        raise ValueError(f"Failed to calculate EPSG code: {e}")

def bbox_calc(lat, lon, time_budget, travel_speed):
    max_distance = travel_speed * time_budget  # In meters

    # Convert to degrees
    delta_lat = max_distance / 111000
    delta_lon = max_distance / (111320 * math.cos(math.radians(lat)))
    # Define bounding box
    north, south = lat + delta_lat, lat - delta_lat
    east, west = lon + delta_lon, lon - delta_lon
    bbox= (west, south, east, north) 
    return bbox   #(left, bottom, right, top). 


def features_calc(lat, lon, bbox, epsg, bounding_poly,features_to_fetch):
    # Define OSM tag mappings
    tag_map = {
        "water": {"natural": "water"},
        "park": {"leisure": "park"},
        "school": {"amenity": "school"},
        "university": {"amenity": "university"},
        "hospital": {"amenity": "hospital"},
        "forest": {"landuse": "forest"},
        "place_of_worship": {"amenity": "place_of_worship"},
        "playground": {"leisure": "playground"}
    }
    tags = {}
    for feature in features_to_fetch:
        if feature in tag_map:
            for key, value in tag_map[feature].items():
                if key in tags:
                    if isinstance(tags[key], list):
                        tags[key].append(value)
                    else:
                        tags[key] = [tags[key], value]
                else:
                    tags[key] = value
    features_gdf = ox.features_from_bbox(bbox=bbox, tags=tags)

    # Center point for distance calculations
    center = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
    center_proj = center.to_crs(epsg)
    center_point = center_proj.geometry.iloc[0]

    # Initialize results
    results = []

    for feature in features_to_fetch:
        sub_filter = tag_map.get(feature)
        if not sub_filter:
            continue

        # Filter features of this type
        mask = pd.Series([True] * len(features_gdf), index=features_gdf.index)
        for key, val in sub_filter.items():
            if key in features_gdf.columns:
                mask &= (features_gdf[key] == val)
        gdf_sub = features_gdf[mask]


        if gdf_sub.empty:
            results.append({
                "feature": feature,
                "count": 0,
                "total_area_m2": 0,
                "nearest_dist_m": None,
                "mean_dist_m": None
            })
            continue

        # Project to match isochrone CRS
        gdf_sub = gdf_sub.to_crs(epsg)
        gdf_sub = gdf_sub[gdf_sub.geometry.intersects(bounding_poly)]

        if gdf_sub.empty:
            results.append({
                "feature": feature,
                "count": 0,
                "total_area_m2": 0,
                "nearest_dist_m": None,
                "mean_dist_m": None
            })
            continue

        # Area (for polygons)
        if gdf_sub.geom_type.isin(['Polygon', 'MultiPolygon']).any():
            area = gdf_sub.geometry.area.sum()
        else:
            area = 0

        # Distance calculations
        dists = gdf_sub.geometry.centroid.distance(center_point)
        results.append({
            "feature": feature,
            "count": len(gdf_sub),
            "total_area_m2": area,
            "nearest_dist_m": dists.min(),
            "mean_dist_m": dists.mean()
        })

    feature_summary = {
    item["feature"]: {
        "count": item["count"],
        "total_area_m2": item["total_area_m2"],
        "nearest_dist_m": item["nearest_dist_m"],
        "mean_dist_m": item["mean_dist_m"]
    }
    for item in results
    }

    return feature_summary

def sociodemo_calc(bounding_poly):

  
    poly_geojson = json.dumps(mapping(bounding_poly))

    # Connect to Supabase (Postgres)
    # conn = psycopg2.connect(
    #     dbname="postgres",
    #     user="postgres",
    #     password="7fxL0xfuw9w6PfPd",
    #     host="db.fiuxnvanhbujhwdwygto.supabase.co",
    #     port="5432",
    # )

    conn = psycopg2.connect(
        user='postgres.fiuxnvanhbujhwdwygto',
        password='7fxL0xfuw9w6PfPd',
        host='aws-0-us-west-1.pooler.supabase.com',
        port='6543',
        dbname='postgres'
    )
    cursor = conn.cursor()

    # PostGIS query: intersect and weight ACS data by area
    query = f"""
    SELECT
    SUM(total *
      (ST_Area(ST_Intersection(geometry, iso.geom)::geography) / ST_Area(geometry::geography))) AS estimated_total,
    SUM(total_male *
      (ST_Area(ST_Intersection(geometry, iso.geom)::geography) / ST_Area(geometry::geography))) AS estimated_male,
    SUM(total_female *
      (ST_Area(ST_Intersection(geometry, iso.geom)::geography) / ST_Area(geometry::geography))) AS estimated_female
FROM
    idaho_acs_bg,
    (SELECT ST_SetSRID(ST_GeomFromGeoJSON('{poly_geojson}'), 4326) AS geom) AS iso
WHERE
    ST_Intersects(geometry, iso.geom);
    """

    cursor.execute(query)
    row = cursor.fetchone()
    conn.close()

    return {
        "total_popl": row[0],
        "total_male": row[1],
        "total_female": row[2]
    }

    
def main_calc(lat, lon, time_budget, mode, features_to_fetch):
    # Validate input coordinates
    if not (-90 <= lat <= 90):
        raise ValueError(f"Invalid latitude: {lat}")
    if not (-180 <= lon <= 180):
        raise ValueError(f"Invalid longitude: {lon}")

    # Travel speed
    if mode == 'walk':
        travel_speed = 80  # m/min (about 3 mph)
    else:  # bike
        travel_speed = 270  # m/min (about 10 mph)

    # Get network
    bbox = bbox_calc(lat, lon, time_budget, travel_speed)
    print("BBOX:", bbox)
    if not all(isinstance(coord, (int, float)) for coord in bbox):
        raise ValueError("Invalid bounding box coordinates")
    G = ox.graph_from_bbox(bbox, network_type=mode)
    if not G.nodes or not G.edges:
        raise ValueError("Graph is empty; check bbox or network_type")

    # Get center node and project graph
    gdf_nodes = ox.graph_to_gdfs(G, edges=False)
    center_node = ox.distance.nearest_nodes(G, lon, lat, return_dist=False)
    if center_node not in G.nodes:
        raise ValueError("Center node not found in graph")
    
    epsg = epsg_calc(lat, lon)
    print(f"Lat: {lat}, Lon: {lon}, EPSG: {epsg}")
    
    # Validate EPSG code
    try:
        import pyproj
        pyproj.CRS.from_epsg(epsg)
    except Exception as e:
        raise ValueError(f"Invalid EPSG code {epsg}: {e}")
    
    # Project graph
    try:
        G = ox.project_graph(G, to_crs=epsg)
    except Exception as e:
        raise ValueError(f"Failed to project graph to EPSG:{epsg}: {e}")

    # Add edge time attribute
    meters_per_minute = travel_speed * 1000 / 60  # km per hour to m per minute
    for u, v, k, data in G.edges(data=True, keys=True):
        length = data.get('length', 0)
        if length <= 0:
            raise ValueError(f"Invalid edge length for edge ({u}, {v}, {k}): {length}")
        data['time'] = length / meters_per_minute

    # Make isochrone polygons
    subgraph = nx.ego_graph(G, center_node, radius=time_budget, distance='time')
    print("Subgraph nodes:", len(subgraph.nodes))
    if len(subgraph.nodes) < 3:
        raise ValueError("Subgraph has fewer than 3 nodes; cannot form a convex hull")

    node_points = [Point((data['x'], data['y'])) for node, data in subgraph.nodes(data=True)]
    if len(node_points) < 3:
        raise ValueError(f"Insufficient points for convex hull: {len(node_points)}")

    poly = gpd.GeoSeries(node_points).unary_union.convex_hull
    if poly.is_empty or not poly.is_valid:
        raise ValueError("Convex hull is empty or invalid")

    poly_proj = gpd.GeoSeries([poly], crs=epsg)
    bounding_poly_proj = poly_proj.iloc[0]  # Polygon for features_calc
    bounding_poly_4326 = poly_proj.to_crs(epsg=4326)  # GeoSeries for to_json
    print("BP->", bounding_poly_4326.iloc[0])

    if not bounding_poly_4326.iloc[0].is_valid:
        raise ValueError("Transformed polygon (EPSG:4326) is invalid")
    if not bounding_poly_proj.is_valid:
        raise ValueError("Projected polygon is invalid")

    geojson_str = bounding_poly_4326.to_json()
    print("GeoJSON:", geojson_str)
    geojson_data = json.loads(geojson_str)
    if not geojson_data.get("features"):
        raise ValueError("GeoJSON has no features")
    bounding_poly_geojson = geojson_data["features"][0]["geometry"]

    # Calculate feature proximities
    proximity_data = features_calc(lat, lon, bbox, epsg, bounding_poly_proj, features_to_fetch)

    # Calculate socio-demographic data
    socio_demo_data = sociodemo_calc(bounding_poly_4326.iloc[0])  # Assuming Polygon

    # Final result dictionary
    return {
        **proximity_data,
        **socio_demo_data,
        "bounding_poly_geojson": bounding_poly_geojson
    }