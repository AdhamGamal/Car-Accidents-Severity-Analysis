import pyproj
from geopy.geocoders import Nominatim
from tqdm import tqdm


# ****************************************************************************
# ****************************************************************************
def convert_bng_to_latlon(easting, northing):
    """
    Converts British National Grid (BNG) coordinates to latitude and longitude.

    Parameters:
    - easting (float): Easting coordinate value in the British National Grid.
    - northing (float): Northing coordinate value in the British National Grid.

    Returns:
    tuple: A tuple containing latitude and longitude values.
    """
    bng = pyproj.Proj(init="epsg:27700")
    lon, lat = bng(easting, northing, inverse=True)
    return lat, lon


# ****************************************************************************
# ****************************************************************************
def get_address_from_coordinates(geolocator, easting, northing):
    """
    Retrieves the address information from geographical coordinates using a geolocator.

    Parameters:
    - geolocator: An instance of a geolocator object (e.g., Nominatim).
    - easting (float): Easting coordinate value.
    - northing (float): Northing coordinate value.

    Returns:
    str: The address corresponding to the given coordinates, or "Location not found" if not available.
    """
    lat, lon = convert_bng_to_latlon(easting, northing)
    location = geolocator.reverse((lat, lon), language='en')
    return location.address if location else "Location not found"


# ****************************************************************************
# ****************************************************************************
def convert_coordinates_to_locations(df, easting_col, northing_col):
    """
    Converts British National Grid (BNG) coordinates in a DataFrame to human-readable locations.

    Parameters:
    - df: Pandas DataFrame containing BNG coordinates.
    - easting_col (str): Name of the column containing easting coordinates.
    - northing_col (str): Name of the column containing northing coordinates.

    Returns:
    list: A list of human-readable locations corresponding to the given coordinates.
    """
    geolocator = Nominatim(user_agent="reverse_geocoding_example")
    locations = []
    for easting, northing in tqdm(zip(df[easting_col], df[northing_col]), total=len(df)):
        address = get_address_from_coordinates(geolocator, easting, northing)
        locations.append(address)
    
    return locations


# ****************************************************************************
# ****************************************************************************
def format_time(df, column_name):
    """
    Formats time values in a DataFrame column to a standard time format.

    Parameters:
    - df: Pandas DataFrame.
    - column_name (str): Name of the column containing time values.

    Returns:
    pandas.Series: A Series with formatted time values.
    """
    return df[column_name].astype(str).apply(lambda x: f'00:{int(x):02d}' if len(x) < 3 else f'{int(x[:-2]):02d}:{int(x[-2:]):02d}')


# ****************************************************************************
# ****************************************************************************
def map_vehicle_type(vehicle):
    """
    Maps a vehicle type based on a string containing the vehicle's description.

    Parameters:
    - vehicle (str): String containing the vehicle's description.

    Returns:
    str: The mapped vehicle type ('Car', 'Motorcycle', 'Bus', or 'Other').
    """
    vehicle = vehicle.lower()
    if 'car' in vehicle:
        return 'Car'
    elif 'cycle' in vehicle:
        return 'Motorcycle'
    elif 'bus' in vehicle:
        return 'Bus'
    else:
        return 'Other'
