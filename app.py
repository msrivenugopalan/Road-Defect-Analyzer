
import time

import pandas as pd
import streamlit as st
import folium
import pydeck as pdk
from streamlit_folium import st_folium
import numpy as np
import altair as alt
import base64
from route_service import get_driving_route
from simulator import simulate_vehicles
from location_matcher import group_events_by_location
from jerk_detector import compare_detectors
from evaluation import evaluate_detectors
from pathlib import Path
import streamlit.components.v1 as components
from database import (
    initialize_database,
    save_simulation_run,
    load_vehicle_samples,
    save_road_defect_events,
    load_recent_road_defect_events,
    load_reports,
    find_csv_import_by_hash,
    register_csv_import,
    save_csv_road_defect_events,
    dismiss_hotspot,
    load_dismissed_hotspots,
    restore_dismissed_hotspot,
    save_injected_pothole_config,
    load_injected_pothole_config,
    get_next_injected_defect_id,
)
import io
import hashlib
from defect_detector import analyse_samples
from defect_detector import detect_defects
from hotspot_engine import find_hotspots
from reporting import send_reports_for_hotspots
from folium.plugins import TimestampedGeoJson
import dataclasses

from config import DetectionConfig, HotspotConfig, MailConfig

st.set_page_config(
    page_title="Road Defect Analyzer",
    layout="wide",
)

def format_timestamp_column(df, columns):
    df = df.copy()

    for column in columns:
        if column in df.columns:
            df[column] = (
                pd.to_datetime(df[column], errors="coerce")
                .dt.strftime("%Y-%m-%d %H:%M:%S")
            )

    return df
def get_route_signature(route):
    """Create an identity for the currently selected route."""

    if not route or not route.get("coordinates"):
        return None

    coordinates = [
        (
            round(float(point["latitude"]), 7),
            round(float(point["longitude"]), 7),
        )
        for point in route["coordinates"]
    ]

    return hashlib.sha256(
        repr(coordinates).encode("utf-8")
    ).hexdigest()

def render_html_report_link(report_path, report_id=None):
    """Display the saved HTML report path as plain text."""

    if not report_path:
        st.caption("No saved HTML report path is available.")
        return

    path = Path(str(report_path))

    if not path.is_file():
        st.warning(f"Saved HTML report file not found: {path}")
        return

    st.write("Saved to:", str(path))

def distance_along_route_to_point(route, latitude, longitude):
    from simulator import _prepare_route

    coordinates, cumulative_distances, _ = _prepare_route(route)

    # Convert coordinates to a local metre-based plane.
    mean_lat = sum(p["latitude"] for p in coordinates) / len(coordinates)
    metres_per_degree_lat = 111_320.0
    metres_per_degree_lon = (
        111_320.0 * np.cos(np.radians(mean_lat))
    )

    def xy(lat, lon):
        return (
            lon * metres_per_degree_lon,
            lat * metres_per_degree_lat,
        )

    point_x, point_y = xy(latitude, longitude)
    best_distance_sq = float("inf")
    best_route_distance = 0.0

    for i in range(len(coordinates) - 1):
        start = coordinates[i]
        end = coordinates[i + 1]

        x1, y1 = xy(start["latitude"], start["longitude"])
        x2, y2 = xy(end["latitude"], end["longitude"])

        dx = x2 - x1
        dy = y2 - y1
        segment_length_sq = dx * dx + dy * dy

        if segment_length_sq == 0:
            fraction = 0.0
        else:
            fraction = (
                (point_x - x1) * dx + (point_y - y1) * dy
            ) / segment_length_sq
            fraction = max(0.0, min(1.0, fraction))

        projected_x = x1 + fraction * dx
        projected_y = y1 + fraction * dy

        distance_sq = (
            (point_x - projected_x) ** 2
            + (point_y - projected_y) ** 2
        )

        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            segment_length = (
                cumulative_distances[i + 1] - cumulative_distances[i]
            )
            best_route_distance = (
                cumulative_distances[i] + fraction * segment_length
            )

    return best_route_distance


def prepare_speed_graph_data(telemetry_data, defect_positions):
    """Generate a smooth speed curve with dips only at defect locations."""

    required = {"distance_along_route_m", "speed_mps"}

    if telemetry_data is None or telemetry_data.empty:
        return pd.DataFrame()

    if not required.issubset(telemetry_data.columns):
        return pd.DataFrame()

    source = telemetry_data[
        ["distance_along_route_m", "speed_mps"]
    ].copy()

    source["distance_along_route_m"] = pd.to_numeric(
        source["distance_along_route_m"], errors="coerce"
    )
    source["speed_mps"] = pd.to_numeric(
        source["speed_mps"], errors="coerce"
    )

    source = (
        source.dropna()
        .sort_values("distance_along_route_m")
        .drop_duplicates(subset=["distance_along_route_m"])
        .reset_index(drop=True)
    )

    if len(source) < 2:
        return pd.DataFrame()

    distances = source["distance_along_route_m"].to_numpy(dtype=float)
    speeds = source["speed_mps"].to_numpy(dtype=float)

    min_distance = float(distances.min())
    max_distance = float(distances.max())

    if max_distance <= min_distance:
        return pd.DataFrame()

    # Keep only defect positions inside the route.
    valid_positions = sorted({
        float(position)
        for position in defect_positions
        if (
            position is not None
            and np.isfinite(float(position))
            and min_distance <= float(position) <= max_distance
        )
    })

    # Create a uniformly spaced distance axis.
    smooth_distances = np.linspace(
        min_distance,
        max_distance,
        3000,
    )

    # Interpolate the original telemetry.
    smooth_speeds = np.interp(
        smooth_distances,
        distances,
        speeds,
    )
    # Remove the artificial starting acceleration from the graph.
    # Estimate the initial baseline using positive speeds from the
    # first 30 metres instead of using the initial zero-speed sample.
    initial_window_m = 30.0

    initial_mask = (
        (distances <= min_distance + initial_window_m)
        & (speeds > 0)
    )

    if initial_mask.any():
        initial_speed = float(
            np.percentile(speeds[initial_mask], 80)
        )
    else:
        initial_speed = float(
            np.percentile(speeds[speeds > 0], 80)
        )

    smooth_speeds = np.maximum(
        smooth_speeds,
        initial_speed,
    )

    
    # Smooth small fluctuations without changing simulator data.
    spacing_m = (
        smooth_distances[1] - smooth_distances[0]
    )

    sigma_m = 2.5
    sigma_points = max(sigma_m / spacing_m, 0.5)

    kernel_radius = max(
        2,
        int(np.ceil(4 * sigma_points)),
    )

    kernel_x = np.arange(
        -kernel_radius,
        kernel_radius + 1,
    )

    gaussian_kernel = np.exp(
        -0.5 * (kernel_x / sigma_points) ** 2
    )

    gaussian_kernel /= gaussian_kernel.sum()

    padded_speeds = np.pad(
        smooth_speeds,
        (kernel_radius, kernel_radius),
        mode="edge",
    )

    smooth_speeds = np.convolve(
        padded_speeds,
        gaussian_kernel,
        mode="valid",
    )

    # Remove artificial speed reduction near the destination.
    # Use the same baseline approach as the starting point.
    # Add a subtle speed dip at the destination only.
    destination_dip_radius_m = 20.0
    destination_dip_factor = 0.90

    destination_distance = np.abs(
        smooth_distances - max_distance
    )

    destination_normalized = np.clip(
        destination_distance / destination_dip_radius_m,
        0.0,
        1.0,
    )

    destination_multiplier = (
        destination_dip_factor
        + (1.0 - destination_dip_factor)
        * (
            0.5
            - 0.5 * np.cos(
                np.pi * destination_normalized
            )
        )
    )

    smooth_speeds = smooth_speeds * destination_multiplier

    # Apply rounded speed reductions only at defect locations.
    dip_radius_m = 20.0
    minimum_speed_factor = 0.65

    speed_multiplier = np.ones(len(smooth_distances))

    for position in valid_positions:

        distance_from_defect = np.abs(
            smooth_distances - position
        )

        normalized_distance = np.clip(
            distance_from_defect / dip_radius_m,
            0.0,
            1.0,
        )

        defect_multiplier = (
            minimum_speed_factor
            + (1.0 - minimum_speed_factor)
            * (
                0.5
                - 0.5 * np.cos(
                    np.pi * normalized_distance
                )
            )
        )

        speed_multiplier = np.minimum(
            speed_multiplier,
            defect_multiplier,
        )

    # No artificial starting-point or destination slowdown.
    graph_speeds = smooth_speeds * speed_multiplier

    return pd.DataFrame({
        "distance_along_route_m": smooth_distances,
        "speed_mps": graph_speeds,
    })



def build_folium_route_map(
    route,
    current_positions=None,
    animation_samples=None,
):
    """Build one Leaflet map for the route, vehicles, defect, and hotspots."""
    middle = route["coordinates"][len(route["coordinates"]) // 2]

    route_map = folium.Map(
        location=[middle["latitude"], middle["longitude"]],
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    route_points = [
        [point["latitude"], point["longitude"]]
        for point in route["coordinates"]
    ]

    folium.PolyLine(
        route_points,
        color="#2878dc",
        weight=5,
        opacity=0.85,
        tooltip="Driving route",
    ).add_to(route_map)

    folium.Marker(
        route_points[0],
        tooltip="Start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(route_map)

    folium.Marker(
        route_points[-1],
        tooltip="Destination",
        icon=folium.Icon(color="red", icon="flag"),
    ).add_to(route_map)
    
    
    
    
    

    



    # Latest position of each vehicle.
    if (
        current_positions is not None
        and not current_positions.empty
        and {"vehicle_id", "latitude", "longitude"}.issubset(
            current_positions.columns
        )
    ):
        for _, vehicle in current_positions.dropna(
            subset=["latitude", "longitude"]
        ).iterrows():
            folium.CircleMarker(
                location=[vehicle["latitude"], vehicle["longitude"]],
                radius=6,
                color="white",
                weight=1,
                fill=True,
                fill_color="#36a66b",
                fill_opacity=1,
                tooltip=f"Vehicle: {vehicle['vehicle_id']}",
            ).add_to(route_map)

    
    
    
    # Draw injected defects at their permanent GPS coordinates.
    potholes = st.session_state.get("pothole_configs", [])

    for display_number, pothole in enumerate(potholes, start=1):
        display_name = f"Pothole {display_number}"
        latitude = pothole.get("latitude")
        longitude = pothole.get("longitude")

        # Skip old defects until their coordinates are established.
        if latitude is None or longitude is None:
            continue

        latitude = float(latitude)
        longitude = float(longitude)

        distance_m = float(
            pothole.get("distance_along_route_m", 0.0)
        )

        folium.CircleMarker(
            location=[latitude, longitude],
            radius=9,
            color="#ffffff",
            weight=2,
            fill=True,
            fill_color="#fa7e02",
            fill_opacity=1.0,
            popup=display_name,
            tooltip=(
                f"{pothole.get('defect_type', 'pothole').title()} "
                f"— {pothole.get('severity', 'medium')} severity "
                f"— {latitude:.6f}, {longitude:.6f}"
            ),
        ).add_to(route_map)




    # Persistent hotspots, excluding dismissed ones.
    hotspots = st.session_state.get("road_hotspots", pd.DataFrame())
    show_hotspots = st.session_state.get("show_road_defect_hotspots", True)

    if (
        show_hotspots
        and not hotspots.empty
        and {"latitude", "longitude", "hotspot_id", "defect_type"}.issubset(
            hotspots.columns
        )
    ):
        for _, hotspot in hotspots.iterrows():
            folium.CircleMarker(
                location=[hotspot["latitude"], hotspot["longitude"]],
                radius=9,
                color="white",
                weight=2,
                fill=True,
                fill_color="#f28c28",
                fill_opacity=0.9,
                tooltip=(
                    f"Hotspot {hotspot['hotspot_id']} | "
                    f"{hotspot['defect_type']} | "
                    f"Severity: {hotspot.get('max_severity', 'unknown')}"
                ),
                popup=str(hotspot["hotspot_id"]),
            ).add_to(route_map)

        
    
    # Add the animation control as a Folium script component.
    if (
        animation_samples is not None
        and not animation_samples.empty
        and {"vehicle_id", "timestamp", "latitude", "longitude"}.issubset(
            animation_samples.columns
        )
    ):
        import json
        from branca.element import MacroElement
        from jinja2 import Template

        animation_data = animation_samples.dropna(
            subset=["vehicle_id", "timestamp", "latitude", "longitude"]
        ).copy()

        animation_data["timestamp"] = pd.to_datetime(
            animation_data["timestamp"], errors="coerce"
        )
        animation_data["latitude"] = pd.to_numeric(
            animation_data["latitude"], errors="coerce"
        )
        animation_data["longitude"] = pd.to_numeric(
            animation_data["longitude"], errors="coerce"
        )
        animation_data = animation_data.dropna(
            subset=["timestamp", "latitude", "longitude"]
        ).sort_values("timestamp")
        print("Vehicle IDs in animation data:", sorted(animation_data["vehicle_id"].astype(str).unique()))
        print("Samples per vehicle:", animation_data.groupby("vehicle_id").size().to_dict())
        for vehicle_id, vehicle_data in animation_data.groupby("vehicle_id"):
            print(
                vehicle_id,
                "samples =", len(vehicle_data),
                "start =", (
                    vehicle_data.iloc[0]["latitude"],
                    vehicle_data.iloc[0]["longitude"],
                ),
                "end =", (
                    vehicle_data.iloc[-1]["latitude"],
                    vehicle_data.iloc[-1]["longitude"],
                ),
            )
        
        tracks = {}
        for vehicle_id, vehicle_data in animation_data.groupby("vehicle_id"):
            tracks[str(vehicle_id)] = [
                [
                    float(row["latitude"]),
                    float(row["longitude"]),
                    float(row["timestamp"].timestamp()),
                ]
                for _, row in vehicle_data.iterrows()
            ]

        tracks = {
            vehicle_id: points
            for vehicle_id, points in tracks.items()
            if len(points) >= 2
        }
        print("Vehicles included in animation:", sorted(tracks.keys()))
        print("Number of moving vehicles:", len(tracks))
        print(
            "Points per vehicle:",
            {vehicle_id: len(points) for vehicle_id, points in tracks.items()}
        )
        
        for vehicle_id, points in tracks.items():
            print(
                vehicle_id,
                "first point =", points[0][:2],
                "last point =", points[-1][:2],
                "total points =", len(points),
            )


        if tracks:
            class VehicleAnimation(MacroElement):
                def __init__(self, track_data):
                    super().__init__()
                    self._name = "VehicleAnimation"
                    self.track_data = track_data
                    self._template = Template("""
                    {% macro script(this, kwargs) %}
                    (function () {
                        var map = {{ this._parent.get_name() }};
                        var tracks = {{ this.track_data | tojson }};
                        console.log("Vehicle IDs received by animation:", Object.keys(tracks));
                        console.log("Track point counts:", Object.fromEntries(
                            Object.entries(tracks).map(([id, points]) => [id, points.length])
                        ));
                        var markers = {};
                        var playing = false;
                        var startTime = 0;
                        var pausedTime = 0;
                        var frame = null;
                        var duration = 15000;
                        var button;

                        var control = L.control({position: "topright"});

                        control.onAdd = function () {
                            var box = L.DomUtil.create("div", "leaflet-bar");
                            button = L.DomUtil.create("button", "", box);
                            button.type = "button";
                            button.textContent = "▶ Play vehicles";
                            button.style.cssText =
                                "padding:8px 10px;background:white;" +
                                "border:0;cursor:pointer;white-space:nowrap;";

                            L.DomEvent.disableClickPropagation(box);
                            L.DomEvent.on(button, "click", function () {
                                if (playing) {
                                    playing = false;
                                    pausedTime += performance.now() - startTime;
                                    if (frame !== null) {
                                        cancelAnimationFrame(frame);
                                        frame = null;
                                    }
                                    button.textContent = "▶ Resume vehicles";
                                } else {
                                    
                                    if (Object.keys(markers).length === 0) {
                                        Object.keys(tracks).forEach(function (id, index) {
                                            var points = tracks[id];

                                            if (!points || points.length < 2) return;

                                            // Give each vehicle a distinct marker colour.
                                            var colors = [
                                                "#16a34a", "#2563eb", "#dc2626", "#9333ea",
                                                "#ea580c", "#0891b2", "#c026d3", "#4d7c0f",
                                                "#be123c", "#4338ca", "#0f766e", "#a16207"
                                            ];

                                            var vehicleIcons = [
                                                "🚗", "🚙", "🚕", "🚐", "🚓",
                                                "🚑", "🚒", "🚌", "🚎", "🚚",
                                                "🚛", "🚜", "🏎️", "🏍️", "🛵",
                                                "🚲", "🚔", "🚍", "🚖", "🚘"
                                            ];

                                            var vehicleIcon = L.divIcon({
                                                className: "vehicle-marker",
                                                html: `
                                                    <div style="
                                                        width: 32px;
                                                        height: 32px;
                                                        border-radius: 50%;
                                                        background: ${colors[index % colors.length]};
                                                        border: 2px solid white;
                                                        display: flex;
                                                        align-items: center;
                                                        justify-content: center;
                                                        box-shadow: 0 2px 6px rgba(0,0,0,0.35);
                                                        font-size: 19px;
                                                        line-height: 1;
                                                    ">🚗</div>
                                                `,
                                                iconSize: [32, 32],
                                                iconAnchor: [16, 16]
                                            });

                                            var iconIndex = Object.keys(tracks)
                                                .sort()
                                                .indexOf(id) % vehicleIcons.length;

                                            var vehicleIcon = L.divIcon({
                                                className: "vehicle-marker",
                                                html: `
                                                    <div style="
                                                        width: 32px;
                                                        height: 32px;
                                                        border-radius: 50%;
                                                        background: ${colors[index % colors.length]};
                                                        border: 2px solid white;
                                                        display: flex;
                                                        align-items: center;
                                                        justify-content: center;
                                                        box-shadow: 0 2px 6px rgba(0,0,0,0.35);
                                                        font-size: 19px;
                                                        line-height: 1;
                                                    ">${vehicleIcons[iconIndex]}</div>
                                                `,
                                                iconSize: [32, 32],
                                                iconAnchor: [16, 16]
                                            });

                                            markers[id] = L.marker(
                                                [points[0][0], points[0][1]],
                                                {
                                                    icon: vehicleIcon,
                                                    zIndexOffset: 1000
                                                }
                                            ).bindTooltip("Vehicle: " + id, {
                                                permanent: false,
                                                direction: "top"
                                            }).addTo(map);
                                        });
                                    }

                                    console.log("Vehicle markers created after Play:", Object.keys(markers));
                                    playing = true;
                                    console.log("Vehicle markers created after Play:", Object.keys(markers));
                                    startTime = performance.now();
                                    button.textContent = "Ⅱ Pause vehicles";
                                    frame = requestAnimationFrame(update);
                                }
                            });

                            return box;
                        };

                        function update(now) {
                            if (!playing) return;

                            var elapsed = pausedTime + now - startTime;
                            var progress = Math.min(1, elapsed / duration);
                            if (progress === 0 || progress >= 1) {
                                console.log("Animation progress:", progress);
                                console.log(
                                    "Current vehicle positions:",
                                    Object.fromEntries(
                                        Object.keys(markers).map(function (id) {
                                            var p = markers[id].getLatLng();
                                            return [id, [p.lat, p.lng]];
                                        })
                                    )
                                );
                            }

                            Object.keys(tracks).forEach(function (id) {
                                var points = tracks[id];
                                var marker = markers[id];
                                if (!marker) return;

                                
                            var animationStart = Math.min.apply(
                                null,
                                Object.keys(tracks).map(function (id) {
                                    return tracks[id][0][2];
                                })
                            );
                            var animationEnd = Math.max.apply(
                                null,
                                Object.keys(tracks).map(function (id) {
                                    return tracks[id][tracks[id].length - 1][2];
                                })
                            );
                            var target = animationStart +
                                progress * (animationEnd - animationStart);


                                var left = points[0];
                                var right = points[points.length - 1];

                                for (var i = 1; i < points.length; i++) {
                                    if (points[i][2] >= target) {
                                        left = points[i - 1];
                                        right = points[i];
                                        break;
                                    }
                                }

                                var span = right[2] - left[2];
                                var fraction = span > 0
                                    ? Math.max(0, Math.min(1, (target - left[2]) / span))
                                    : 0;

                                marker.setLatLng([
                                    left[0] + (right[0] - left[0]) * fraction,
                                    left[1] + (right[1] - left[1]) * fraction
                                ]);
                            });

                            if (progress < 1) {
                                frame = requestAnimationFrame(update);
                            } else {
                                playing = false;
                                pausedTime = 0;
                                frame = null;
                                button.textContent = "▶ Replay vehicles";
                            }
                        }

                        control.addTo(map);
                        console.log("Vehicle markers created:", Object.keys(markers));
                    })();
                    {% endmacro %}
                    """)

            route_map.add_child(VehicleAnimation(tracks))

     
        # Include fixed defect coordinates when fitting the map view.
    map_bounds_points = list(route_points)

    for pothole in st.session_state.get("pothole_configs", []):
        latitude = pothole.get("latitude")
        longitude = pothole.get("longitude")

        if latitude is not None and longitude is not None:
            map_bounds_points.append([
                float(latitude),
                float(longitude),
            ])

    if map_bounds_points:
        route_map.fit_bounds(map_bounds_points, padding=[30, 30])

    return route_map



DEFAULT_SIMULATION_CONFIG = {
    "target_speed_kmh": 36.0,
    "max_acceleration_mps2": 3.0,
    "max_braking_mps2": 7.0,
    "max_jerk_mps3": 2.0,
    "timestep_seconds": 0.2,
}


# Default simulation configuration
if "simulation_config" not in st.session_state:
    st.session_state.simulation_config = DEFAULT_SIMULATION_CONFIG.copy()


@st.dialog("Configure Simulation")
def configure_simulation():
    config = st.session_state.simulation_config

    target_speed = st.slider(
        "Target Speed (km/h)",
        min_value=5.0,
        max_value=100.0,
        value=float(config["target_speed_kmh"]),
        step=1.0,
    )

    max_acceleration = st.slider(
        "Maximum Acceleration (m/s²)",
        min_value=0.5,
        max_value=5.0,
        value=float(config["max_acceleration_mps2"]),
        step=0.5,
    )

    max_braking = st.slider(
        "Maximum Braking (m/s²)",
        min_value=1.0,
        max_value=10.0,
        value=float(config["max_braking_mps2"]),
        step=0.5,
    )

    max_jerk = st.slider(
        "Maximum Jerk (m/s³)",
        min_value=0.5,
        max_value=5.0,
        value=float(config["max_jerk_mps3"]),
        step=0.5,
    )

    
    timestep_options = [0.05, 0.1, 0.2, 0.5]

    timestep = st.selectbox(
            "Simulation Time Step (seconds)",
            options=timestep_options,
            index=timestep_options.index(
                float(config["timestep_seconds"])
            ) if float(config["timestep_seconds"]) in timestep_options else 2,
    )


    if st.button("Save Settings", use_container_width=True):
        st.session_state.simulation_config = {
            "target_speed_kmh": target_speed,
            "max_acceleration_mps2": max_acceleration,
            "max_braking_mps2": max_braking,
            "max_jerk_mps3": max_jerk,
            "timestep_seconds": timestep,
        }
        st.success("Simulation settings saved.")
        st.rerun()



def filter_dismissed_hotspots(hotspots, radius_m=10.0):
    """Filter dismissed locations by coordinates and defect type."""
    from geo import haversine_m

    if hotspots is None or hotspots.empty:
        return hotspots.copy() if hotspots is not None else pd.DataFrame()

    dismissed = load_dismissed_hotspots()

    if dismissed.empty:
        return hotspots.copy()

    keep = []
    filtered_out = []
    for _, hotspot in hotspots.iterrows():
        kind = str(hotspot.get("defect_type", "")).strip().casefold()
        latitude = float(hotspot["latitude"])
        longitude = float(hotspot["longitude"])
        hidden = False

        matching = dismissed[
            dismissed["defect_type"].astype(str).str.strip().str.casefold()
            == kind
        ]

        for _, item in matching.iterrows():
            if haversine_m(
                latitude, longitude,
                float(item["latitude"]), float(item["longitude"]),
            ) <= radius_m:
                hidden = True
                break

        keep.append(not hidden)

    return hotspots.loc[keep].copy().reset_index(drop=True)


initialize_database()

# Load persistent road-defect evidence whenever the app reruns.
hotspot_config = HotspotConfig()

recent_defect_events = load_recent_road_defect_events(
    window_days=hotspot_config.window_days
)

# Match the timestamp column expected by the hotspot engine.
if (
    not recent_defect_events.empty
    and "event_timestamp" in recent_defect_events.columns
):
    recent_defect_events = recent_defect_events.rename(
        columns={"event_timestamp": "timestamp"}
    )

# Distinguish vehicle instances from different simulation runs.
if not recent_defect_events.empty:
    recent_defect_events["vehicle_key"] = (
    recent_defect_events["run_id"].astype(str)
    + " "
    + recent_defect_events["vehicle_id"].astype(str)
)
# Rebuild hotspots from the persisted evidence.
road_hotspots, _ = find_hotspots(
    recent_defect_events,
    radius_m=hotspot_config.radius_m,
    min_vehicles=hotspot_config.min_vehicles,
    window_days=hotspot_config.window_days,
)

# Keep the loaded evidence available to the dashboard.
st.session_state.road_defect_events = recent_defect_events
road_hotspots = filter_dismissed_hotspots(road_hotspots)
st.session_state.road_hotspots = road_hotspots

st.title("Road Defect Analyzer")
st.caption("Simulated Vehicle Monitoring and Road-Defect Analysis")



# ---------------- ROAD ROUTE ----------------

st.subheader("Select a Road Route")

# Keep the retrieved route when Streamlit reruns the app.
if "route_result" not in st.session_state:
    st.session_state.route_result = None

if "pothole_configs" not in st.session_state:
    saved_config = load_injected_pothole_config()

    if isinstance(saved_config, list):
        st.session_state["pothole_configs"] = saved_config
    elif isinstance(saved_config, dict):
        # Migrate the existing single-defect configuration.
        st.session_state["pothole_configs"] = [saved_config]
    else:
        st.session_state["pothole_configs"] = []





col1, col2 = st.columns(2)

with col1:
    start_place = st.text_input(
        "Start Location",
        value="Tambaram Railway Station, Chennai, India",
        key="route_start_place",
    )

with col2:
    destination_place = st.text_input(
        "Destination",
        value="Guindy Railway Station, Chennai, India",
        key="route_destination_place",
    )

if st.button("Get Route", key="get_route_button"):
    if not start_place.strip() or not destination_place.strip():
        st.warning("Please enter both the start and destination.")
    else:
        try:
            with st.spinner("Finding the driving route..."):
                route = get_driving_route(
                    start_place.strip(),
                    destination_place.strip(),
                )

            st.session_state.route_result = route

            # Clear samples belonging to the previously selected route.
            st.session_state.samples = pd.DataFrame()
            st.session_state.samples_route_signature = None
            st.session_state.pop("map_selected_defect_coordinates", None)
            st.session_state.pop("map_selected_defect_distance", None)
            st.session_state.pop("last_defect_map_click", None)
            st.session_state.pop("selected_injected_defect_id", None)
            # Clear the previous run's dashboard results.
            st.session_state.events = None
            st.session_state.detection_results = None
            st.session_state.locations = None
            st.session_state["telemetry_data"] = pd.DataFrame()
            st.session_state["telemetry_defect_enabled"] = False

            st.success("Driving route retrieved successfully!")

        except Exception as error:
            st.error(f"Could not retrieve the route: {error}")

route = st.session_state.route_result

if route is not None:
    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Distance",
        f"{route['distance_km']:.2f} km",
    )

    metric2.metric(
        "Estimated Driving Time",
        f"{route['duration_minutes']:.1f} min",
    )

    



    
    show_hotspots = st.checkbox(
        "Show Road-Defect Hotspots",
        value=True,
        key="show_road_defect_hotspots",
    )

    st.subheader("Driving Route Map")

    
    route_map_placeholder = st.empty()

    saved_samples = st.session_state.get("samples")
    ordered_samples = pd.DataFrame()
    latest_positions = pd.DataFrame()

    current_route_signature = get_route_signature(route)

    saved_route_signature = st.session_state.get(
        "samples_route_signature"
    )

    # Use samples only when they belong to the displayed route.
    if (
        saved_samples is not None
        and not saved_samples.empty
        and saved_route_signature == current_route_signature
        and {
            "vehicle_id", "timestamp", "latitude", "longitude"
        }.issubset(saved_samples.columns)
    ):
        ordered_samples = saved_samples.copy()

        ordered_samples["timestamp"] = pd.to_datetime(
            ordered_samples["timestamp"],
            errors="coerce",
        )

        ordered_samples = ordered_samples.dropna(
            subset=["timestamp", "latitude", "longitude"]
        ).sort_values("timestamp")

        latest_positions = ordered_samples.groupby(
            "vehicle_id"
        ).tail(1)

    

    if route is not None:
        current_positions = latest_positions

        
        animation_is_playing = st.session_state.get(
            "play_vehicle_animation", False
        )

        map_result = st_folium(
            build_folium_route_map(
                route,
                current_positions,
                animation_samples=ordered_samples,
            ),
            height=520,
            use_container_width=True,
            key="roadsense_main_map",
            returned_objects=["last_object_clicked_popup", "last_clicked"],
        )
        
        # Show a remove button below the map when an injected
        # defect marker is selected.
        clicked_popup = (
            map_result.get("last_object_clicked_popup")
            if map_result else None
        )

        configured_defects = st.session_state.get(
            "pothole_configs", []
        )

        
        # Map visible pothole labels to their permanent database IDs.
        defect_ids_by_display_name = {
            f"Pothole {index}": str(defect["defect_id"])
            for index, defect in enumerate(configured_defects, start=1)
        }

        # Identify the clicked pothole using its visible map label.
        if clicked_popup:
            clicked_popup = str(clicked_popup).strip()

            if clicked_popup in defect_ids_by_display_name:
                st.session_state["selected_injected_defect_id"] = (
                    defect_ids_by_display_name[clicked_popup]
                )


        selected_defect_id = st.session_state.get(
            "selected_injected_defect_id"
        )

        selected_defect = next(
            (
                defect for defect in configured_defects
                if str(defect["defect_id"]) == selected_defect_id
            ),
            None,
        )

        if selected_defect is not None:
            selected_display_number = next(
                (
                    index
                    for index, defect in enumerate(configured_defects, start=1)
                    if str(defect["defect_id"]) == selected_defect_id
                ),
                None,
            )

            st.info(
                f"Selected: Pothole {selected_display_number} — "
                f"{selected_defect['distance_along_route_m']:.1f} m"
            )

            if st.button(
                f"Remove Pothole {selected_display_number}",
                key="remove_selected_injected_defect",
            ):
                updated_defects = [
                    defect for defect in configured_defects
                    if str(defect["defect_id"]) != selected_defect_id
                ]

                st.session_state["pothole_configs"] = updated_defects
                save_injected_pothole_config(updated_defects)
                st.session_state.pop(
                    "selected_injected_defect_id", None
                )

                st.rerun()

        
        


        # Handle a click on the map to place a defect.
        clicked_point = (
            map_result.get("last_clicked") if map_result else None
        )
        

        if (
            st.session_state.get("place_defect_on_map", False)
            and clicked_point
        ):
            click_key = (
                round(clicked_point["lat"], 6),
                round(clicked_point["lng"], 6),
            )

            if st.session_state.get("last_defect_map_click") != click_key:
                selected_distance = distance_along_route_to_point(
                    route,
                    clicked_point["lat"],
                    clicked_point["lng"],
                )
                

                st.session_state["last_defect_map_click"] = click_key
                st.session_state["map_selected_defect_distance"] = round(
                    selected_distance, 1
                )
                st.session_state["pothole_distance_input"] = round(
                    selected_distance, 1
                )
                st.session_state["map_selected_defect_coordinates"] = click_key

        if st.session_state.get("place_defect_on_map", False):
            st.caption(
                "Defect placement is enabled. Click the desired location "
                "on the route map to set the distance."
            )

            selected_coordinates = st.session_state.get(
                "map_selected_defect_coordinates"
            )
            selected_distance = st.session_state.get(
                "map_selected_defect_distance"
            )

            if selected_coordinates and selected_distance is not None:
                st.caption(
                    f"Selected coordinates: {selected_coordinates[0]:.6f}, "
                    f"{selected_coordinates[1]:.6f}"
                )
                st.caption(
                    f"Nearest position along route: {selected_distance:.1f} m"
                )

        # Remove a hotspot using its popup identifier.
        clicked_popup = (
            map_result.get("last_object_clicked_popup")
            if map_result else None
        )
        
        


        if clicked_popup:
            hotspots = st.session_state.get("road_hotspots", pd.DataFrame())

            if (
                not hotspots.empty
                and "hotspot_id" in hotspots.columns
                and hotspots["hotspot_id"].astype(str).eq(
                    str(clicked_popup)
                ).any()
            ):
                st.session_state["selected_hotspot_id"] = str(clicked_popup)

        selected_hotspot_id = st.session_state.get("selected_hotspot_id")

        if selected_hotspot_id:
            hotspots = st.session_state.get("road_hotspots", pd.DataFrame())

            if (
                not hotspots.empty
                and "hotspot_id" in hotspots.columns
            ):
                selected_rows = hotspots[
                    hotspots["hotspot_id"].astype(str).eq(
                        selected_hotspot_id
                    )
                ]

                if not selected_rows.empty:
                    selected_hotspot = selected_rows.iloc[0]

                    st.caption(
                        f"Selected hotspot: {selected_hotspot_id} — "
                        f"{selected_hotspot['defect_type']}"
                    )

                    if st.button(
                        "Remove Selected Hotspot",
                        key="remove_selected_hotspot",
                    ):
                        dismiss_hotspot(
                            latitude=float(selected_hotspot["latitude"]),
                            longitude=float(selected_hotspot["longitude"]),
                            defect_type=str(selected_hotspot["defect_type"]),
                        )
                        st.session_state.pop("road_hotspots", None)
                        st.session_state.pop("selected_hotspot_id", None)
                        st.rerun()

        
        if ordered_samples.empty:
            st.info("Start a simulation to display vehicle positions.")

    else:
        st.info("Retrieve a driving route to display the map.")

    
    




    

if route is not None:
    with st.expander("Road Names Along the Route", expanded=False):
        road_names = route.get("road_names") or []

        if road_names:
            for road_name in road_names:
                st.write(f"- {road_name}")
        else:
            st.info(
                "No road names were returned for this route. "
                "Some roads may not have names in OpenStreetMap."
            )


# ------------ END ROAD ROUTE ------------


if "samples" not in st.session_state:
    st.session_state.samples = pd.DataFrame()

if "events" not in st.session_state:
    st.session_state.events = None

if "locations" not in st.session_state:
    st.session_state.locations = None

if "detection_results" not in st.session_state:
    st.session_state.detection_results = None

if "road_defect_events" not in st.session_state:
    st.session_state.road_defect_events = pd.DataFrame()

if "road_hotspots" not in st.session_state:
    st.session_state.road_hotspots = pd.DataFrame()

if "report_results" not in st.session_state:
    st.session_state.report_results = []
# Reporting table for the latest simulation only.
if "current_run_report_hotspots" not in st.session_state:
    st.session_state.current_run_report_hotspots = pd.DataFrame()


st.header("Live Simulation")
mode = "Live Simulation"

vehicle_count = st.slider(
    "Number of Vehicles",
    min_value=1,
    max_value=20,
    value=10,
    key="vehicle_count_selector",
)

st.subheader("Controlled Road Defect")

place_defect_on_map = st.checkbox(
    "Place Defect on Map",
    value=False,
    key="place_defect_on_map",
)

selected_route = st.session_state.get("route_result")



enable_pothole = st.checkbox(
    "Simulate a pothole",
    value=False,
    key="enable_pothole",
)

if "pothole_distance_input" not in st.session_state:
    st.session_state["pothole_distance_input"] = 100.0

pothole_distance_m = st.number_input(
    "Pothole Distance Along Route (m)",
    min_value=0.0,
    step=10.0,
    disabled=not enable_pothole,
    key="pothole_distance_input",
)


pothole_severity = st.selectbox(
    "Pothole Severity",
    ["low", "medium", "high"],
    index=1,
    disabled=not enable_pothole,
)

if st.button("Add Defect to Simulation", key="add_defect_button"):
    selected_route = st.session_state.get("route_result")

    if selected_route is None:
        st.warning("Please retrieve a driving route first.")
    else:
        from simulator import _prepare_route, _get_position_at_distance

        coordinates, cumulative_distances, route_distance_m = (
            _prepare_route(selected_route)
        )

        # If the user clicked the map, use that exact selected location.
        selected_coordinates = st.session_state.get(
            "map_selected_defect_coordinates"
        )

        if selected_coordinates is not None:
            latitude = float(selected_coordinates[0])
            longitude = float(selected_coordinates[1])

            # Store the route distance as supporting simulation metadata.
            distance_m = distance_along_route_to_point(
                selected_route, latitude, longitude
            )
        else:
            # Otherwise, calculate the GPS location at the entered distance.
            distance_m = float(pothole_distance_m)

            if not 0 <= distance_m < route_distance_m:
                st.warning(
                    "Defect distance must be within the selected route."
                )
                st.stop()

            position = _get_position_at_distance(
                coordinates,
                cumulative_distances,
                distance_m,
            )
            latitude = float(position["latitude"])
            longitude = float(position["longitude"])

        potholes = st.session_state["pothole_configs"]

        defect_id = get_next_injected_defect_id()

        new_defect = {
            "defect_id": defect_id,
            "defect_type": "pothole",
            "distance_along_route_m": float(distance_m),
            "severity": pothole_severity,
            "latitude": latitude,
            "longitude": longitude,
        }

        updated_potholes = potholes + [new_defect]

        save_injected_pothole_config(updated_potholes)
        st.session_state["pothole_configs"] = updated_potholes

        # Clear the old map selection so it isn't reused accidentally.
        st.session_state.pop(
            "map_selected_defect_coordinates", None
        )
        st.session_state.pop(
            "map_selected_defect_distance", None
        )
        st.session_state.pop("last_defect_map_click", None)

        st.success(
            f"Added {new_defect['defect_id']} at fixed GPS coordinates "
            f"{latitude:.6f}, {longitude:.6f}."
        )
        st.rerun()





st.caption(
    "Each vehicle has its own ID and movement samples. "
    "The five-vehicle threshold applies to distinct vehicles "
    "at a suspected location, not to the selected vehicle count."
)

if st.button("Configure Simulation", use_container_width=True):
    configure_simulation()

if st.button("Start Simulation", type="primary"):
    selected_route = st.session_state.get("route_result")

    
    
    # Preserve every saved pothole configuration.
    configured_defects = [
        defect.copy()
        for defect in st.session_state.get("pothole_configs", [])
    ]

   

    

    selected_route = st.session_state.get("route_result")

    if enable_pothole and configured_defects and selected_route is None:
        st.warning("Please retrieve a driving route first.")
        st.stop()

    # Only inject saved defects when the checkbox is enabled.
    defect_configs = []
    defect_config = None

    if enable_pothole and selected_route is not None:
        from simulator import _prepare_route, _get_position_at_distance
        from geo import haversine_m

        coordinates, cumulative_distances, route_distance_m = (
            _prepare_route(selected_route)
        )

        
        for defect in configured_defects:
            latitude = defect.get("latitude")
            longitude = defect.get("longitude")

            # Recover coordinates for older defects that lack GPS values.
            if (
                latitude is None
                or longitude is None
                or pd.isna(latitude)
                or pd.isna(longitude)
            ):
                saved_distance = defect.get("distance_along_route_m")

                if (
                    saved_distance is None
                    or pd.isna(saved_distance)
                    or not 0 <= float(saved_distance) < route_distance_m
                ):
                    st.warning(
                        f"{defect['defect_id']} has no valid GPS coordinates "
                        "or usable saved route distance. Re-add this defect "
                        "using its original route."
                    )
                    continue

                position = _get_position_at_distance(
                    coordinates,
                    cumulative_distances,
                    float(saved_distance),
                )

                latitude = float(position["latitude"])
                longitude = float(position["longitude"])

            # Validate BOTH old and newly configured defects against the route.
            distance_m = distance_along_route_to_point(
                selected_route,
                float(latitude),
                float(longitude),
            )

            if not 0 <= distance_m < route_distance_m:
                continue

            nearest_position = _get_position_at_distance(
                coordinates,
                cumulative_distances,
                distance_m,
            )

            offset_m = haversine_m(
                float(latitude),
                float(longitude),
                float(nearest_position["latitude"]),
                float(nearest_position["longitude"]),
            )

            # Only simulate defects within 30 m of the selected route.
            if offset_m <= 30.0:
                run_defect = defect.copy()
                run_defect["latitude"] = float(latitude)
                run_defect["longitude"] = float(longitude)
                run_defect["distance_along_route_m"] = float(distance_m)
                defect_configs.append(run_defect)


        if enable_pothole and configured_defects and not defect_configs:
            st.info(
                "None of the configured defects are within 30 m of this route. "
                "Their saved locations remain unchanged."
            )
        # Pass validated potholes to the simulator.
        defect_config = defect_configs if defect_configs else None


    

    # Generate samples regardless of whether defects are configured.

    with st.spinner("Simulating vehicle movement..."):
        samples, _ = simulate_vehicles(
            vehicle_count,
            route=selected_route,
            config=st.session_state.simulation_config,
            defect_config=defect_config,
        )

    # Ensure the simulator returned a DataFrame.
    if samples is None:
        st.error("The simulator did not return vehicle samples.")
        st.stop()

    if not isinstance(samples, pd.DataFrame):
        st.error("The simulator returned an unexpected data format.")
        st.stop()

    if samples.empty:
        st.warning("No vehicle samples were generated.")
        st.stop()

    
    # ---------------- SAVE VEHICLE SPEED TELEMETRY ----------------

    telemetry_data = pd.DataFrame()

    required_telemetry_columns = {
        "vehicle_id",
        "distance_along_route_m",
        "speed_mps",
    }

    if required_telemetry_columns.issubset(samples.columns):

        # Preserve the existing first-vehicle telemetry behaviour.
        target_vehicle_id = samples["vehicle_id"].iloc[0]

        telemetry_data = samples[
            samples["vehicle_id"] == target_vehicle_id
        ].copy()

        telemetry_data["distance_along_route_m"] = pd.to_numeric(
            telemetry_data["distance_along_route_m"],
            errors="coerce",
        )

        telemetry_data["speed_mps"] = pd.to_numeric(
            telemetry_data["speed_mps"],
            errors="coerce",
        )

        telemetry_data = (
            telemetry_data
            .dropna(subset=[
                "distance_along_route_m",
                "speed_mps",
            ])
            .sort_values("distance_along_route_m")
        )

    # Preserve all telemetry samples.
    st.session_state["telemetry_data"] = telemetry_data.copy()

    # Store only defects actually injected into this simulation.
    st.session_state["telemetry_defect_positions"] = [
        float(defect["distance_along_route_m"])
        for defect in defect_configs
        if (
            defect.get("distance_along_route_m") is not None
            and pd.notna(defect.get("distance_along_route_m"))
        )
    ]
      
    # ---------------- VEHICLE SPEED TELEMETRY ----------------

    telemetry_data = st.session_state.get(
        "telemetry_data",
        pd.DataFrame()
    )

    telemetry_is_pothole = st.session_state.get(
        "telemetry_defect_enabled",
        False
    )

    # Optional jerk diagnostics. Missing columns will not crash the app.
    if "jerk_mps3" in samples.columns:
        debug_columns = [
            "vehicle_id",
            "timestamp",
            "speed_mps",
            "acceleration_mps2",
            "jerk_mps3",
            "distance_along_route_m",
        ]
        available_columns = [
            column for column in debug_columns
            if column in samples.columns
        ]

        spike_indices = samples["jerk_mps3"].abs().nlargest(10).index

        

    # Run both detection methods for every simulation.
    detection_results = compare_detectors(samples)

    if detection_results is None:
        st.error("The detector did not return results.")
        st.stop()

    

    # Keep events detected by either method.
    rule_mask = (
        detection_results["rule_detected"].fillna(False).astype(bool)
        if "rule_detected" in detection_results.columns
        else pd.Series(False, index=detection_results.index)
    )
    ml_mask = (
        detection_results["ml_detected"].fillna(False).astype(bool)
        if "ml_detected" in detection_results.columns
        else pd.Series(False, index=detection_results.index)
    )

    events = detection_results[rule_mask | ml_mask].copy()

    

    # Preserve the real-time processing option.
    if mode == "Real-Time Simulation":
        progress = st.progress(0, text="Starting simulation...")
        live_status = st.empty()
        live_table = st.empty()

        if "vehicle_id" in samples.columns:
            vehicle_ids = samples["vehicle_id"].dropna().unique()
            total = len(vehicle_ids)
            processed_ids = []

            for index, vehicle_id in enumerate(vehicle_ids, start=1):
                processed_ids.append(vehicle_id)

                visible_samples = samples[
                    samples["vehicle_id"].isin(processed_ids)
                ]

                live_status.write(
                    f"Processing {vehicle_id} — {index} of {total} vehicles"
                )
                live_table.dataframe(
                    visible_samples,
                    use_container_width=True,
                    hide_index=True,
                )
                progress.progress(
                    index / total,
                    text=f"Simulation progress: {index / total:.0%}",
                )
                time.sleep(0.25)

        progress.empty()
        live_status.empty()
        live_table.empty()

    # Group detected events by location.
    locations = group_events_by_location(
        events,
        radius_metres=hotspot_config.radius_m,
        
    )
    # Save the simulation and samples to SQLite.
    simulation_config = st.session_state.simulation_config

    database_config = {
        "simulation_mode": mode,
        "vehicle_count": vehicle_count,
        "target_speed_kmh": simulation_config["target_speed_kmh"],
        "max_acceleration_mps2": simulation_config["max_acceleration_mps2"],
        "max_braking_mps2": simulation_config["max_braking_mps2"],
        "max_jerk_mps3": simulation_config["max_jerk_mps3"],
        "timestep_seconds": simulation_config["timestep_seconds"],
        "driving_behavior": "automatic",
        "route_start": (
            selected_route.get("start", {}).get("name")
            if selected_route else None
        ),
        "route_destination": (
            selected_route.get("destination", {}).get("name")
            if selected_route else None
        ),
    }

    run_id = save_simulation_run(database_config, samples)
    
    
    # Detect road defects using vertical acceleration.
    defect_events = detect_defects(samples)
    
    
    

    hotspot_config = HotspotConfig()

    # Save the current run's events to SQLite.
    saved_event_count = save_road_defect_events(
        run_id,
        defect_events,
    )

    # Load evidence from all recent simulation runs.
    recent_defect_events = load_recent_road_defect_events(
        window_days=hotspot_config.window_days
    )
    # Match the column name expected by hotspot_engine.py.
    if (
        not recent_defect_events.empty
        and "event_timestamp" in recent_defect_events.columns
    ):
        recent_defect_events = recent_defect_events.rename(
            columns={"event_timestamp": "timestamp"}
        )

    # Identify each simulated vehicle instance uniquely.
    if not recent_defect_events.empty:
        recent_defect_events["vehicle_key"] = (
            recent_defect_events["run_id"].astype(str)
            + " "
            + recent_defect_events["vehicle_id"].astype(str)
        )

    
    # Reporting is based on defects injected into THIS simulation run.
    # Historical hotspot dismissals must not hide newly injected defects.
    from geo import haversine_m
    # Use only defects configured and validated for this simulation run.
    current_run_defects = list(defect_configs) if defect_configs else []
    injected_hotspots = []
    report_evidence = recent_defect_events.copy()

    # Use evidence from the current run only.
    if not report_evidence.empty and "run_id" in report_evidence.columns:
        report_evidence = report_evidence[
            report_evidence["run_id"].astype(str) == str(run_id)
        ].copy()

    for defect in defect_configs:
        defect_id = str(defect["defect_id"])
        latitude = float(defect["latitude"])
        longitude = float(defect["longitude"])
        defect_type = str(defect.get("defect_type", "pothole"))
        severity = str(defect.get("severity", "medium"))

        # Associate current-run detections with this injected defect.
        matching_events = report_evidence.copy()

        if not matching_events.empty:
            matching_events = matching_events.dropna(
                subset=["latitude", "longitude"]
            ).copy()

            matching_events["_distance_m"] = matching_events.apply(
                lambda event: haversine_m(
                    latitude,
                    longitude,
                    float(event["latitude"]),
                    float(event["longitude"]),
                ),
                axis=1,
            )

            matching_events = matching_events[
                matching_events["_distance_m"] <= 30.0
            ].copy()

        vehicle_ids = (
            matching_events["vehicle_id"].dropna().astype(str).unique().tolist()
            if not matching_events.empty and "vehicle_id" in matching_events.columns
            else []
        )

        timestamps = (
            pd.to_datetime(matching_events["timestamp"], errors="coerce").dropna()
            if not matching_events.empty and "timestamp" in matching_events.columns
            else pd.Series(dtype="datetime64[ns]")
        )

        injected_hotspots.append({
            "hotspot_id": defect_id,
            "defect_id": defect_id,
            "defect_type": defect_type,
            "latitude": latitude,
            "longitude": longitude,
            "distinct_vehicle_count": len(vehicle_ids),
            "vehicle_ids": vehicle_ids,
            "event_count": len(matching_events),
            "first_seen": timestamps.min() if not timestamps.empty else pd.NaT,
            "last_seen": timestamps.max() if not timestamps.empty else pd.NaT,
            "max_severity": severity,
            "spread_m": 0.0,
            "priority_score": 1.0,
            "report_eligible": (
                len(vehicle_ids) >= hotspot_config.min_vehicles
            ),
        })

    # One eligible location per valid injected defect.
    hotspots = pd.DataFrame(injected_hotspots)
    eligible_total = (
        int(hotspots["report_eligible"].sum())
        if not hotspots.empty
        else 0
    )

    st.write("Defects injected this run:", len(defect_configs))
    st.write("Eligible Locations for Report:", eligible_total)

    # Retrieve existing reports and create reports only for new eligible defects.
    mail_config = MailConfig.from_env()
    report_results = []

    saved_reports_for_lookup = load_reports(limit=1000)

    # Find defect IDs that already have a saved report.
    existing_report_ids = set()

    if (
        not saved_reports_for_lookup.empty
        and "defect_id" in saved_reports_for_lookup.columns
    ):
        existing_reports = saved_reports_for_lookup[
            saved_reports_for_lookup["status"].isin(["sent", "dry_run"])
            & saved_reports_for_lookup["defect_id"].notna()
        ].copy()

        existing_report_ids = set(
            existing_reports["defect_id"].astype(str)
        )

        # Retrieve the latest saved report for each configured defect.
        for defect in configured_defects:
            defect_id = str(defect["defect_id"])

            matching_reports = existing_reports[
                existing_reports["defect_id"].astype(str) == defect_id
            ]

            if not matching_reports.empty:
                previous_report = matching_reports.sort_values(
                    ["created_at", "report_id"],
                    ascending=False,
                ).iloc[0]

                report_results.append({
                    "status": "existing_report",
                    "report_id": previous_report["report_id"],
                    "outbox_path": previous_report.get("outbox_path"),
                    "defect_id": defect_id,
                    "detail": "Previously generated report retrieved.",
                })

    # Only consider currently eligible defects for new report creation.
    if not hotspots.empty:
        new_report_hotspots = hotspots[
            hotspots["report_eligible"].fillna(False).astype(bool)
            & ~hotspots["defect_id"].astype(str).isin(existing_report_ids)
        ].copy()

        if not new_report_hotspots.empty:
            new_report_results = send_reports_for_hotspots(
                new_report_hotspots,
                report_evidence,
                mail_config,
                data_source="simulation",
                run_id=run_id,
            )

            report_results.extend(new_report_results)

    # Keep historical evidence separate from the current-run report table.
    st.session_state.road_defect_events = recent_defect_events
    st.session_state.road_hotspots = hotspots
    st.session_state.current_run_report_hotspots = hotspots.copy()
    st.session_state.report_results = report_results
    st.session_state.eligible_injected_count = eligible_total


    # Store the results for the dashboard and route map.
    st.session_state.samples = samples

    # Remember which route generated these samples.
    st.session_state.samples_route_signature = (
        get_route_signature(selected_route)
    )

    st.session_state.events = events
    st.session_state.detection_results = detection_results
    st.session_state.locations = locations
    st.session_state.simulation_mode = mode

    st.success("Simulation completed successfully!")

    # Refresh the map so it immediately uses the new simulation samples.
    st.rerun()



# ---------------- END MOVING VEHICLE ----------------
# ---------------- VEHICLE SPEED TELEMETRY ----------------



with st.expander(
    "Vehicle Speed Telemetry",
    expanded=False,
):

    telemetry_data = st.session_state.get(
        "telemetry_data", pd.DataFrame()
    )

    # Use all currently active injected defects from the map.
    # This includes defects saved during previous simulations.
    defect_positions = [
        float(defect["distance_along_route_m"])
        for defect in st.session_state.get("pothole_configs", [])
        if (
            defect.get("distance_along_route_m") is not None
            and pd.notna(defect.get("distance_along_route_m"))
        )
    ]

    if not telemetry_data.empty:

        st.caption(
            "Vehicle speed variation along the route, "
            "including configured defect slowdowns and destination slowdown."
        )

        chart_data = prepare_speed_graph_data(
            telemetry_data,
            defect_positions,
        )

        if not chart_data.empty:
            speed_chart = (
                alt.Chart(chart_data)
                .mark_line(
                    color="#2878dc",
                    strokeWidth=3,
                    interpolate="monotone",
                )
                .encode(
                    x=alt.X(
                        "distance_along_route_m:Q",
                        title="Distance Along Route (m)",
                    ),
                    y=alt.Y(
                        "speed_mps:Q",
                        title="Vehicle Speed (m/s)",
                        scale=alt.Scale(zero=True),
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "distance_along_route_m:Q",
                            title="Distance (m)",
                            format=".1f",
                        ),
                        alt.Tooltip(
                            "speed_mps:Q",
                            title="Speed (m/s)",
                            format=".2f",
                        ),
                    ],
                )
                .properties(
                    height=320,
                    title="Vehicle Speed Along the Route",
                )
                .interactive()
            )

            st.altair_chart(
                speed_chart,
                use_container_width=True,
            )
        else:
            st.info("No valid telemetry points are available.")

    else:
        st.info("Run a simulation to generate vehicle telemetry.")




samples = st.session_state.samples
events = st.session_state.events
locations = st.session_state.locations
detection_results = st.session_state.detection_results

st.divider()
st.header("Road-Defect Reporting")

road_hotspots = st.session_state.road_hotspots
report_hotspots = st.session_state.current_run_report_hotspots
report_results = st.session_state.report_results

with st.expander("Manage dismissed hotspot markers", expanded=False):
    if not road_hotspots.empty:
        hotspot_options = list(range(len(road_hotspots)))

        def hotspot_label(index):
            item = road_hotspots.iloc[index]
            return (
                f"{item.get('defect_type', 'Unknown')} ? "
                f"{float(item['latitude']):.5f}, "
                f"{float(item['longitude']):.5f}"
            )

        selected_index = st.selectbox(
            "Visible hotspot to dismiss",
            options=hotspot_options,
            format_func=hotspot_label,
            key="dismiss_hotspot_selection",
        )
        confirm_dismiss = st.checkbox(
            "Confirm hiding this marker",
            key="confirm_hotspot_dismissal",
        )

        if st.button("Dismiss Selected Marker", key="dismiss_hotspot_button"):
            if not confirm_dismiss:
                st.warning("Check the confirmation box before dismissing.")
            else:
                item = road_hotspots.iloc[selected_index]
                dismiss_hotspot(
                    item["latitude"],
                    item["longitude"],
                    item["defect_type"],
                )
                st.session_state.pop("dismiss_hotspot_selection", None)
                st.session_state.pop("confirm_hotspot_dismissal", None)
                st.rerun()
    else:
        st.caption("There are no visible hotspots to dismiss.")

    active_dismissals = load_dismissed_hotspots()
    if active_dismissals.empty:
        st.caption("There are no dismissed markers to restore.")
    else:
        dismissal_ids = active_dismissals["dismissal_id"].astype(int).tolist()

        def dismissal_label(dismissal_id):
            item = active_dismissals[
                active_dismissals["dismissal_id"] == dismissal_id
            ].iloc[0]
            return (
                f"#{dismissal_id} ? {item['defect_type']} ? "
                f"{float(item['latitude']):.5f}, "
                f"{float(item['longitude']):.5f}"
            )

        selected_dismissal = st.selectbox(
            "Dismissed marker to restore",
            options=dismissal_ids,
            format_func=dismissal_label,
            key="restore_hotspot_selection",
        )

        if st.button("Restore Selected Marker", key="restore_hotspot_button"):
            restore_dismissed_hotspot(selected_dismissal)
            st.session_state.pop("restore_hotspot_selection", None)
            st.rerun()

if report_hotspots.empty:
    st.info(
        "No current-run eligibility results yet. Run a simulation to evaluate configured defects."
    )
else:
    
    display_hotspots = report_hotspots[
        [
            "defect_type",
            "latitude",
            "longitude",
            "distinct_vehicle_count",
            "vehicle_ids",
            "event_count",
            "first_seen",
            "last_seen",
            "max_severity",
            "spread_m",
            "report_eligible",
        ]
    ].copy()

    # Add simple sequential display names.
    display_hotspots.insert(
        0,
        "Pothole",
        [f"Pothole {i}" for i in range(1, len(display_hotspots) + 1)],
    )


    display_hotspots["report_status"] = display_hotspots[
        "report_eligible"
    ].map({
        True: "Eligible for report",
        False: "Awaiting more vehicles",
    })

    display_hotspots = display_hotspots.drop(
        columns="report_eligible"
    )
    
    for column in ("first_seen", "last_seen"):
        if column in display_hotspots.columns:
            display_hotspots[column] = (
                pd.to_datetime(
                    display_hotspots[column], errors="coerce"
                ).dt.strftime("%Y-%m-%d %H:%M:%S")
            )

    st.dataframe(
        display_hotspots,
        use_container_width=True,
        hide_index=True,
        column_config={
            "latitude": st.column_config.NumberColumn(
                "Latitude",
                format="%.6f",
            ),
            "longitude": st.column_config.NumberColumn(
                "Longitude",
                format="%.6f",
            ),
            "spread_m": st.column_config.NumberColumn(
                "Location spread (m)",
                format="%.1f",
            ),
        },
    )
    
    

if report_results:
    for result in report_results:
        status = result.get("status", "unknown")
        report_id = result.get("report_id")
        report_path = result.get("outbox_path")

        if status == "existing_report":
            st.success(
                f"Previously generated report retrieved. "
                f"Report ID: {report_id}"
            )

            render_html_report_link(
                report_path,
                report_id,
            )   

        elif status == "dry_run":
            st.success(
                f"Report created. Report ID: {report_id}"
            )

            render_html_report_link(
                report_path,
                report_id,
            )

        elif status == "duplicate":
            st.info(
                "A matching report already exists; "
                "the existing report will be reused."
            )

            render_html_report_link(
                result.get("outbox_path"),
                result.get("report_id"),
            )

        elif status == "skipped_cap":
            st.warning(
                "Report skipped: the maximum number of reports "
                "allowed in this batch has been reached."
            )

        else:
            st.warning(f"Report status: {status}")

            detail = result.get("detail")
            if detail:
                st.error(f"Reason: {detail}")

else:
    st.caption(
        "No report was created during this simulation. "
        "A report requires an eligible hotspot."
    )

st.divider()
st.header("Saved Report History")

saved_reports = load_reports(limit=100)

if saved_reports.empty:
    st.info("No saved reports found in the database.")
else:
    st.caption(f"Loaded {len(saved_reports)} saved report(s), up to 100.")
    
    history_columns = [
        "report_id",
        "created_at",
        "defect_type",
        "severity",
        "distinct_vehicles",
        "event_count",
        "status",
        "latitude",
        "longitude",
        "run_id",
        "error",
        "outbox_path",
        "defect_id",
    ]

    st.dataframe(
        saved_reports[history_columns],
        use_container_width=True,
        height=400,
        hide_index=True,
        column_config={
            "report_id": "Report ID",
            
            "hotspot_id": "Hotspot ID",
            "defect_type": "Defect Type",
            "distinct_vehicle_count": "Distinct Vehicles",
            "vehicle_ids": "Supporting Vehicles",
            "event_count": "Evidence Events",
            "first_seen": "First Detected",
            "last_seen": "Last Detected",
            "max_severity": "Maximum Severity",

            "created_at": "Created At",
            "defect_type": "Defect Type",
            "severity": "Severity",
            "distinct_vehicles": "Distinct Vehicles",
            "event_count": "Evidence Events",
            "status": "Saved Status",
            "latitude": st.column_config.NumberColumn(
                "Latitude",
                format="%.6f",
            ),
            "longitude": st.column_config.NumberColumn(
                "Longitude",
                format="%.6f",
            ),
            "run_id": "Simulation Run",
            "outbox_path": "Saved Email File",
            "error": "Error Details",
        },
    )

st.divider()

st.header("Simulation Overview")

samples = st.session_state.get("samples", pd.DataFrame())
events = st.session_state.get("events")
detection_results = st.session_state.get("detection_results")

if samples is None or samples.empty or "vehicle_id" not in samples.columns:
    col1, col2, col3 = st.columns(3)
    col1.metric("Vehicles", 0)
    col2.metric("Movement Samples", 0)
    col3.metric("Jerk Events", 0)

    st.info("Choose a simulation mode and start the simulation.")

else:
    vehicle_total = samples["vehicle_id"].nunique()
    sample_total = len(samples)
    event_total = len(events) if events is not None else 0

    report_hotspots = st.session_state.get(
        "current_run_report_hotspots", pd.DataFrame()
    )
    fleet_hotspots = st.session_state.get(
        "road_hotspots", pd.DataFrame()
    )

    injected_eligible_total = (
        int(report_hotspots["report_eligible"].fillna(False).astype(bool).sum())
        if not report_hotspots.empty and "report_eligible" in report_hotspots.columns
        else 0
    )

    fleet_eligible_total = (
        int(fleet_hotspots["report_eligible"].fillna(False).astype(bool).sum())
        if not fleet_hotspots.empty and "report_eligible" in fleet_hotspots.columns
        else 0
    )

    # Count all defect locations currently displayed on the map.
    eligible_total = len(
        st.session_state.get("pothole_configs", [])
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Vehicles", vehicle_total)
    col2.metric("Movement Samples", sample_total)
    col3.metric("Jerk Events", event_total)
    col4.metric("Eligible Locations for Report", eligible_total)
    

    # Individual vehicle condition
    st.divider()
    st.header("Individual Vehicle Condition")

    vehicle_ids = sorted(samples["vehicle_id"].dropna().unique())

    if vehicle_ids:
        selected_vehicle = st.selectbox(
            "Select a vehicle",
            vehicle_ids,
        )

        vehicle_samples = samples[
            samples["vehicle_id"] == selected_vehicle
        ].copy()

        if events is not None and not events.empty and "vehicle_id" in events.columns:
            vehicle_events = events[
                events["vehicle_id"] == selected_vehicle
            ]
        else:
            vehicle_events = pd.DataFrame()

        if vehicle_events.empty:
            st.success("No jerk events detected for this vehicle.")
        else:
            st.warning(
                f"{len(vehicle_events)} jerk event(s) detected "
                f"for {selected_vehicle}."
            )

        if detection_results is not None and "vehicle_id" in detection_results.columns:
            vehicle_detection_results = detection_results[
                detection_results["vehicle_id"] == selected_vehicle
            ].copy()

            display_results = format_timestamp_column(
                vehicle_detection_results,
                ["timestamp", "event_timestamp", "first_seen", "last_seen"]
            )

            st.dataframe(
                display_results,
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("View all movement samples"):
        st.dataframe(
            samples,
            use_container_width=True,
            hide_index=True,
        )


# CSV Sensor-Log Analysis
st.header("CSV Sensor-Log Analysis")

st.caption(
    "Analyse vehicle sensor logs independently of the simulation workflow. "
    "Uploaded events are not added to hotspot reporting."
)

uploaded_file = st.file_uploader(
    "Upload a vehicle sensor CSV",
    type=["csv"],
    key="sensor_csv_upload",
)


if uploaded_file is not None:
    try:
        csv_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(csv_bytes).hexdigest()

        existing_import = find_csv_import_by_hash(file_hash)

        if existing_import:
            st.warning(
                "This exact CSV file has already been imported. "
                f"Original upload: {existing_import['imported_at']}"
            )
        else:
            csv_samples = pd.read_csv(io.BytesIO(csv_bytes))

            required_columns = {
                "vehicle_id",
                "timestamp",
                "latitude",
                "longitude",
                "sensor_vertical_acceleration_mps2",
            }

            missing_columns = required_columns - set(csv_samples.columns)

            if missing_columns:
                st.error(
                    "CSV is missing required columns: "
                    + ", ".join(sorted(missing_columns))
                )
            else:
                csv_samples["timestamp"] = pd.to_datetime(
                    csv_samples["timestamp"], errors="coerce"
                )

                for column in (
                    "latitude",
                    "longitude",
                    "sensor_vertical_acceleration_mps2",
                ):
                    csv_samples[column] = pd.to_numeric(
                        csv_samples[column], errors="coerce"
                    )

                valid_coordinates = (
                    csv_samples["latitude"].between(-90, 90)
                    & csv_samples["longitude"].between(-180, 180)
                )

                valid_rows = (
                    csv_samples["vehicle_id"].notna()
                    & csv_samples["timestamp"].notna()
                    & valid_coordinates
                    & csv_samples[
                        "sensor_vertical_acceleration_mps2"
                    ].notna()
                )

                invalid_count = int((~valid_rows).sum())
                csv_samples = csv_samples.loc[valid_rows].copy()

                if invalid_count:
                    st.warning(
                        f"Skipped {invalid_count} row(s) with missing "
                        "or invalid required values."
                    )

                if csv_samples.empty:
                    st.error("No valid sensor samples were found.")
                else:
                    annotated_samples, defect_events = analyse_samples(
                        csv_samples
                    )

                    import_id = register_csv_import(
                        original_filename=uploaded_file.name,
                        file_hash=file_hash,
                        sample_count=len(annotated_samples),
                        event_count=len(defect_events),
                    )
                    saved_events = save_csv_road_defect_events(
                        import_id,
                        defect_events,
                    )
                    st.success(
                        f"Saved {saved_events} new road-defect event(s) to the database."
                    )

                    csv_hotspots, csv_triggers = find_hotspots(
                        defect_events,
                        radius_m=10.0,
                        min_vehicles=5,
                        window_days=30,
                    )

                    st.success(
                        f"Analysed {len(annotated_samples)} valid samples "
                        f"from {annotated_samples['vehicle_id'].nunique()} "
                        "vehicle(s)."
                    )

                    st.subheader("Detected Road-Defect Events")
                    st.metric("Events detected", len(defect_events))
                    st.subheader("CSV Hotspot Results")

                    if csv_hotspots.empty:
                        st.info("No suspected road-defect locations found.")
                    else:
                        st.dataframe(
                            csv_hotspots,
                            use_container_width=True,
                            hide_index=True,
                        )

                        eligible_count = int(
                            csv_hotspots["report_eligible"].sum()
                        )
                        st.metric("Eligible hotspots", eligible_count)

                    if defect_events.empty:
                        st.info("No road-defect events were detected.")
                    else:
                        st.dataframe(
                            defect_events,
                            use_container_width=True,
                            hide_index=True,
                        )

                    st.subheader("Annotated Sensor Samples")
                    st.dataframe(
                        annotated_samples,
                        use_container_width=True,
                        hide_index=True,
                    )

    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Could not analyse this CSV: {exc}")
