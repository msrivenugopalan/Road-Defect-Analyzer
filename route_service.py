
import time
from functools import lru_cache
import certifi
import requests


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_URL = "https://router.project-osrm.org/route/v1/driving"

HEADERS = {
    "User-Agent": "RoadSenseAI-Local-PoC/1.0"
}


@lru_cache(maxsize=32)
def geocode_place(place_name):
    """Convert a place name into latitude and longitude."""

    response = requests.get(
        NOMINATIM_URL,
        params={
            "q": place_name,
            "format": "jsonv2",
            "limit": 1,
        },
        headers=HEADERS,
        timeout=20,
    )

    response.raise_for_status()
    results = response.json()

    if not results:
        raise ValueError(
            f"Could not find the location: {place_name}"
        )

    return {
        "name": results[0]["display_name"],
        "latitude": float(results[0]["lat"]),
        "longitude": float(results[0]["lon"]),
    }


def get_driving_route(start_place, destination_place):
    """Get a driving route and its road coordinates."""

    start = geocode_place(start_place)

    # Respect the public geocoding service's request limit.
    time.sleep(1.1)

    destination = geocode_place(destination_place)

    coordinates = (
        f"{start['longitude']},{start['latitude']};"
        f"{destination['longitude']},{destination['latitude']}"
    )

    
    response = requests.get(
        f"{OSRM_URL}/{coordinates}",
        params={
            "overview": "full",
            "geometries": "geojson",
            "steps": "true",
        },
        headers=HEADERS,
        timeout=30,
        verify=certifi.where(),
    )


    response.raise_for_status()
    data = response.json()

    if data.get("code") != "Ok" or not data.get("routes"):
        raise ValueError(
            f"Could not calculate a driving route: "
            f"{data.get('message', data.get('code', 'Unknown error'))}"
        )

    route = data["routes"][0]

    # OSRM returns coordinates as [longitude, latitude].
    # Convert them into named fields for our simulator.
    route_coordinates = [
        {
            "latitude": latitude,
            "longitude": longitude,
        }
        for longitude, latitude
        in route["geometry"]["coordinates"]
    ]

    road_names = list(dict.fromkeys(
        step["name"]
        for leg in route["legs"]
        for step in leg["steps"]
        if step.get("name")
    ))

    return {
        "start": start,
        "destination": destination,
        "coordinates": route_coordinates,
        "distance_km": route["distance"] / 1000,
        "duration_minutes": route["duration"] / 60,
        "road_names": road_names,
    }
