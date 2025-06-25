import osmnx as ox
import math
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import networkx as nx
import psycopg2
import json
from shapely.geometry import mapping
from contextlib import contextmanager

def epsg_calc(lat, lon):
   
    try:
        # Create a GeoDataFrame for the given point
        point = gpd.GeoDataFrame(
            {'geometry': [Point(lon, lat)]},
            crs="EPSG:4326"  # WGS84
        )

        # Estimate the best UTM CRS
        best_crs = point.estimate_utm_crs()
        print(f"Best UTM CRS determined: {best_crs.to_string()}")
        return best_crs

    except Exception as e:
        print(f"An error occurred while determining the best EPSG: {e}")
        return None

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


def features_calc(lat, lon, bbox, epsg, bounding_poly, features_to_fetch):
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

            # Convert results to DataFrame and include bounding_poly as GeoJSON
        df = pd.DataFrame(results)
        proximity_data = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": mapping(bounding_poly),  # Isochrone polygon as GeoJSON
                "properties": df.to_dict(orient='records')  # Embed feature stats
            }]
        }

    return proximity_data

# Context manager for database connection
@contextmanager
def get_db_connection():
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="7fxL0xfuw9w6PfPd",
        host="db.fiuxnvanhbujhwdwygto.supabase.co",
        port="5432"
    )
    try:
        yield conn
    finally:
        conn.close()


def sociodemo_calc(bounding_poly):
    # Convert polygon to GeoJSON string
    poly_geojson = json.dumps(mapping(bounding_poly))

    # Define query columns and structure
    columns = [
        "geoid", "total", "total_male", "total_female", "original_area",
        "intersect_area", "area_fraction", "estimated_total", "estimated_male",
        "estimated_female"
    ]

    # Parameterized query with correct table qualification
    query = """
    SELECT
        geoid,
        total,
        total_male,
        total_female,
        ST_Area(idaho_acs_bg.geometry::geography) AS original_area,
        ST_Area(ST_Intersection(idaho_acs_bg.geometry, iso.geom)::geography) AS intersect_area,
        (ST_Area(ST_Intersection(idaho_acs_bg.geometry, iso.geom)::geography) / ST_Area(idaho_acs_bg.geometry::geography)) AS area_fraction,
        total * (ST_Area(ST_Intersection(idaho_acs_bg.geometry, iso.geom)::geography) / ST_Area(idaho_acs_bg.geometry::geography)) AS estimated_total,
        total_male * (ST_Area(ST_Intersection(idaho_acs_bg.geometry, iso.geom)::geography) / ST_Area(idaho_acs_bg.geometry::geography)) AS estimated_male,
        total_female * (ST_Area(ST_Intersection(idaho_acs_bg.geometry, iso.geom)::geography) / ST_Area(idaho_acs_bg.geometry::geography)) AS estimated_female
    FROM
        public.idaho_acs_bg idaho_acs_bg,
        (SELECT ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326) AS geom) AS iso
    WHERE
        ST_Intersects(idaho_acs_bg.geometry, iso.geom);
    """

    # Execute query and fetch results
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SET search_path TO public, tiger;")
            cursor.execute(query, (poly_geojson,))
            rows = cursor.fetchall()
            df = pd.DataFrame(rows, columns=columns)
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        raise
    finally:
        if 'cursor' in locals():
            cursor.close()

    return df


# def sociodemo_calc(bounding_poly):
#     # Convert polygon to GeoJSON string
#     poly_geojson = json.dumps(mapping(bounding_poly))

#     # Connect to Supabase (Postgres)
#     conn = psycopg2.connect(
#         dbname="postgres",
#         user="postgres",
#         password="7fxL0xfuw9w6PfPd",
#         host="db.fiuxnvanhbujhwdwygto.supabase.co",
#         port="5432"
#     )
#     cursor = conn.cursor()
#     print(poly_geojson)
#     # PostGIS query: intersect and weight ACS data by area
#     query = f"""
#     SELECT
#        geoid,
#        total,
#        total_male,
#        total_female,
#        ST_Area(geometry::geography) AS original_area,
#        ST_Area(ST_Intersection(geometry, iso.geom)::geography) AS intersect_area,
#        (ST_Area(ST_Intersection(geometry, iso.geom)::geography) / ST_Area(geometry::geography)) AS area_fraction,
#        total *
#          (ST_Area(ST_Intersection(geometry, iso.geom)::geography) / ST_Area(geometry::geography)) AS estimated_total,
#        total_male *
#          (ST_Area(ST_Intersection(geometry, iso.geom)::geography) / ST_Area(geometry::geography)) AS estimated_male,
#        total_female *
#          (ST_Area(ST_Intersection(geometry, iso.geom)::geography) / ST_Area(geometry::geography)) AS estimated_female
#    FROM
#        idaho_acs_bg,
#        (SELECT ST_SetSRID(ST_GeomFromGeoJSON('{poly_geojson}'), 4326) AS geom) AS iso
#    WHERE
#        ST_Intersects(geometry, iso.geom);
#     """

#     cursor.execute(query)
#     rows = cursor.fetchall()
#     columns = [desc[0] for desc in cursor.description]
#     conn.close()

#     # Convert to pandas DataFrame
#     df = pd.DataFrame(rows, columns=columns)
#     return df
    
# def main_calc(lat, lon, time_budget, mode, features_to_fetch):
#     if mode == 'walk':
#         travel_speed = 80 #m/min (about 3 mph)
#     else: #bike
#         travel_speed = 270 #m/min (about 10 mph)
#     #get network
#     bbox = bbox_calc(lat, lon, time_budget, travel_speed)
#     G = ox.graph_from_bbox(bbox, network_type=mode)

#     # Get the isochrone
#     gdf_nodes = ox.graph_to_gdfs(G, edges=False)
#     center_node = ox.distance.nearest_nodes(G,lon,lat, return_dist=False) #center_node is just a node ID not affected by projection
#     #get best epsg
#     epsg=epsg_calc(lat,lon)
#     #G = ox.project_graph(G, to_crs={'init': 'epsg:2277'})
#     G = ox.project_graph(G, to_crs=epsg)

#     # add an edge attribute for time in minutes required to traverse each edge
#     meters_per_minute = travel_speed * 1000 / 60 #km per hour to m per minute
#     for u, v, k, data in G.edges(data=True, keys=True):
#             data['time'] = data['length'] / meters_per_minute

#     # make the isochrone polygons
#     subgraph = nx.ego_graph(G, center_node, radius=time_budget, distance='time')
#     node_points = [Point((data['x'], data['y'])) for node, data in subgraph.nodes(data=True)]
#     bounding_poly = gpd.GeoSeries(node_points).unary_union.convex_hull
#     #main_calc_data=[subgraph,bounding_poly]

#     #Calculate feature proximities
#     features_to_fetch=["water","school","park"]

#     proximity_data = features_calc(lat, lon, bbox, epsg, bounding_poly, features_to_fetch)
#     #Calculate socio demographic data

#     socio_demo_data = sociodemo_calc(bounding_poly)
#     #Calculate Job access
#     #Job_access_data= job_calc()

    
#     return proximity_data, socio_demo_data

def main_calc(lat, lon, time_budget, mode, features_to_fetch):
    if mode == 'walk':
        travel_speed = 80  # m/min (about 3 mph)
    else:  # bike
        travel_speed = 270  # m/min (about 10 mph)
    # Get network
    bbox = bbox_calc(lat, lon, time_budget, travel_speed)
    G = ox.graph_from_bbox(bbox, network_type=mode)

    # Get the isochrone
    gdf_nodes = ox.graph_to_gdfs(G, edges=False)
    center_node = ox.distance.nearest_nodes(G, lon, lat, return_dist=False)
    epsg = epsg_calc(lat, lon)
    G = ox.project_graph(G, to_crs=epsg)

    # Add an edge attribute for time in minutes
    meters_per_minute = travel_speed * 1000 / 60
    for u, v, k, data in G.edges(data=True, keys=True):
        data['time'] = data['length'] / meters_per_minute

    # Make the isochrone polygon with validation
    subgraph = nx.ego_graph(G, center_node, radius=time_budget, distance='time')
    if subgraph.number_of_nodes() > 2:  # Ensure enough nodes for a polygon
        node_points = [Point((data['x'], data['y'])) for node, data in subgraph.nodes(data=True)]
        bounding_poly = gpd.GeoSeries(node_points).unary_union.convex_hull
        if not bounding_poly.is_valid:
            bounding_poly = bounding_poly.buffer(0)  # Repair invalid geometry
    else:
        # Fallback to a small buffer around the center if too few nodes
        center_point = Point(lon, lat)
        bounding_poly = center_point.buffer(100)  # 100m buffer as fallback

    # Calculate feature proximities
    proximity_data = features_calc(lat, lon, bbox, epsg, bounding_poly, features_to_fetch)

    # Calculate socio-demographic data
    socio_demo_data = sociodemo_calc(bounding_poly)

    return proximity_data, socio_demo_data