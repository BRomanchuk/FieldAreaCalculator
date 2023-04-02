import numpy as np
import pandas as pd

from shapely import Polygon, LineString, Point
from shapely.ops import unary_union
from shapely import wkt

import utm
import ast



def from_latlon(latitude, longitude, geozone_num, geozone_let):
    """
    convert lat, lon to (x, y) coordinates (in meters)
    :param latitude:  list or np.array (shape: (n,))
    :param longitude: list or np.array (shape: (n,))
    :param geozone_num: geozone integer
    :param geozone_let: geozone letter
    :return: np.array of (x, y) coordinates (shape: (n, 2))
    """
    xy = np.array([
        utm.from_latlon(
            latitude=lat, 
            longitude=lon, 
            force_zone_number=geozone_num,
            force_zone_letter=geozone_let
        )[:2] for lat, lon in zip(latitude, longitude)
    ])
    return xy



def get_subfields_xy(field_bounds: list, geozone_num, geozone_let):
    """
    convert GPS bounds of field into (x, y) coordinates (in meters)
    :param field_bounds: list with GPS bounds of subfields
    :return: list of (x, y) bounds of the subfields
    """
    subfields = [np.array(subfield).T for subfield in field_bounds]
    bounds_xy = [from_latlon(field[1], field[0], geozone_num, geozone_let) for field in subfields]
    return bounds_xy



def get_geozone(latitude, longitude):
    """
    :param latitude: 
    :param longitude: 
    :return: UTM geozone number and letter
    """
    geozone_num, geozone_let = utm.from_latlon(latitude, longitude)[2:]
    return geozone_num, geozone_let



def get_polygon_from_bounds(field_bounds, geozone_num, geozone_let):
    """
    get field polygon from string representation of its bounds
    :param field_bounds_str: string representation of the bounds of the field
    :param geozone_num: utm geozone number
    :param geozone_let: utm geozone letter
    :return: shapely field polygon
    """
    # convert GPS field bounds to XY and create field_polygon
    subfields_xy = get_subfields_xy(field_bounds, geozone_num, geozone_let)
    field_polygon = unary_union([
        Polygon(subfield).buffer(0) if len(subfield) > 1 else None for subfield in subfields_xy
    ])
    return field_polygon



def denoise_track(track_xy, simplification_m):
    """
    get denoised track 
    :param track_xy: XY-coordinates of the original track
    :param simplification_m: simplification tolerance
    :return: denoised track 
    """
    # create track LineString object
    track = LineString(track_xy)
    # simplify the track
    simplified_track = track.simplify(simplification_m)
    # project each point of the original track onto the simplified track
    projected_points = [
        simplified_track.interpolate(
            simplified_track.project(
                Point(x, y)
            )
        )
        for x, y in track_xy
    ]
    return projected_points



def get_area_and_distance(
    tool_width : float,
    gps_m_deviation : float,
    
    time : list,
    track_lat : list,
    track_lon : list,
    
    field_bounds : list,
    
    encoded_field_polygon : str = None,
    encoded_calculated_track : str = None,
    
    gps_geozone_num : int = None,
    gps_geozone_let : str = None,
    
    track_polygon_simplification_m = 0.3
):
    """
    Calculates path distances and processed field area

    Parameters
    ----------
    tool_width : float
        Tool-width in meters.
    gps_m_deviation : float 
        GPS deviation in meters.
    time : array-like
        List of timestamps.
    track_lat : array-like
        List of track latitudes.
    track_lon : array-like
        List of track longitudes.
    field_bounds : list of lists
        List of lists with raw GPS field bounds !(lon, lat)!
    encoded_field_polygon : str
        Encoded version of the polygon in XY-system.
    encoded_calculated_track : str
        Encoded version of the calculated track in XY-system.
    gps_geozone_num : int
        GPS geozone number (UTM).
    gps_geozone_let : str
        GPS geozone letter (UTM).
    simplification_m : float
        Tolerance for the simplification of the calculated track.
    
    Returns
    -------
    distance_and_area_calculator : dict
        Dict with list of path distances and areas, 
        and updated parameters of the track and field:
            encoded_field_polygon,
            encoded_calculated_track,
            gps_geozone_num,
            gps_geozone_let.

    """
    
    # init field_polygon
    field_polygon = None
    
    # calculate track buffer as the sum of tool-width and GPS deviation
    buffer = (tool_width + gps_m_deviation) / 2
    
    # if GPS geozone is not defined
    if gps_geozone_num is None or gps_geozone_let is None:
        # define geozone num and let
        gps_geozone_num, gps_geozone_let = get_geozone(track_lat[0], track_lon[0])
        # get field polygon from string bounds
        field_polygon = get_polygon_from_bounds(field_bounds, gps_geozone_num, gps_geozone_let)
        # simplify the field polygon to fasten the computations
        field_polygon = field_polygon.simplify(track_polygon_simplification_m*2)
        # create encoded version of the field polygon
        encoded_field_polygon = field_polygon.wkt
        
    # convert GPS coordinates of the track 
    track_xy = from_latlon(track_lat, track_lon, gps_geozone_num, gps_geozone_let)
    
    # if the field polygon isn't defined
    if field_polygon is None:
        # and its encoded version isn't defined
        if encoded_field_polygon is None:
            # get field polygon from string bounds
            field_polygon = get_polygon_from_bounds(field_bounds, gps_geozone_num, gps_geozone_let)
            # simplify the field polygon to fasten the computations
            field_polygon = field_polygon.simplify(track_polygon_simplification_m*2)
            # create encoded version of the field polygon
            encoded_field_polygon = field_polygon.wkt
        else:
            # load field polygon from encoded version
            field_polygon = wkt.loads(encoded_field_polygon)
    
    
    # dublicate the first point of the track
    track_xy = np.concatenate([track_xy[:1], track_xy])
    
    # denoise track 
    track_xy = denoise_track(track_xy, gps_m_deviation)
    

    # if the encoded calculated track representation isn't defined
    if encoded_calculated_track is None:
        # create empty calculated track 
        calculated_track = LineString([])
    else:
        # load calculated track from encoded version
        calculated_track = wkt.loads(encoded_calculated_track)
        
        
    # create path distance array and cumulative area array
    path_distance = np.zeros(len(time))
    cumulative_field_processed = np.concatenate([[calculated_track.area], np.zeros(len(time))])
    

    # calculation loop : for two consecutive track points (i, i+1)
    # !! ADD index_from !!
    for i in range(len(time)):
        # create LineString object as a subtrack
        subtrack = LineString(track_xy[i : i+2])
        # compute distance between these points, and add to the list
        path_distance[i] = subtrack.intersection(field_polygon).length

        # if the vehicle has no tool, continue calculating only path distance
        if tool_width == 0:
            continue
        
        # if the vehicle has the tool, calculate processed area
        
        # add buffer to the subtrack, and intersect it with the field polygon
        subfield = subtrack.buffer(buffer) \
                           .intersection(field_polygon) 
        # add this intersection to the preprocessed track
        calculated_track = calculated_track.union(subfield)
        
        # create simplified track to fasten the computations
        simplified = calculated_track.simplify(track_polygon_simplification_m, preserve_topology=False)
        # if the difference in areas is small, use simplified track as a new calculated track
        if np.abs(simplified.area - calculated_track.area) < (tool_width * gps_m_deviation) and cumulative_field_processed[i] < simplified.area:
            calculated_track = simplified
            
        # add new cumulative area to the list
        cumulative_field_processed[i+1] = calculated_track.area   

    # calculate processed area for each timestamp
    field_processed = cumulative_field_processed[1:] - cumulative_field_processed[:-1]

    # create output dictionary
    distance_and_area_calculator = {
        'encoded_field_polygon' : encoded_field_polygon,
        'encoded_calculated_track' : calculated_track.wkt,

        'gps_geozone_num' : gps_geozone_num,
        'gps_geozone_let' : gps_geozone_let,
        
        'path_distance' : path_distance,
        'field_processed' : field_processed
    }
    
    return distance_and_area_calculator