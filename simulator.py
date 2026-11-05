from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


MAX_SPEED_MPS = 15.0
MAX_ACCELERATION_MPS2 = 3.0
MAX_BRAKING_MPS2 = 7.0
DRAG_COEFFICIENT = 0.03
SIMULATION_DT_SECONDS = 0.2


DEFAULT_SIMULATION_CONFIG = {
    "target_speed_kmh": 36.0,
    "max_acceleration_mps2": 3.0,
    "max_braking_mps2": 7.0,
    "max_jerk_mps3": 2.0,
    "timestep_seconds": 0.2,
}
def _haversine_metres(point_a, point_b):
    """Calculate the distance between two geographic points."""

    earth_radius = 6_371_000

    lat1 = np.radians(point_a["latitude"])
    lon1 = np.radians(point_a["longitude"])
    lat2 = np.radians(point_b["latitude"])
    lon2 = np.radians(point_b["longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return float(
        2 * earth_radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    )


def _prepare_route(route):
    """Calculate cumulative distances along the supplied route."""

    coordinates = route["coordinates"]

    if len(coordinates) < 2:
        raise ValueError(
            "The route must contain at least two coordinates."
        )

    cumulative_distances = [0.0]

    for index in range(1, len(coordinates)):
        segment_distance = _haversine_metres(
            coordinates[index - 1],
            coordinates[index],
        )

        cumulative_distances.append(
            cumulative_distances[-1] + segment_distance
        )

    total_distance = cumulative_distances[-1]

    if total_distance <= 0:
        raise ValueError("The route must have a positive distance.")

    return coordinates, cumulative_distances, total_distance


def _get_position_at_distance(
    coordinates,
    cumulative_distances,
    distance_metres,
):
    """Interpolate a geographic position at a route distance."""

    distance_metres = float(
        np.clip(
            distance_metres,
            0.0,
            cumulative_distances[-1],
        )
    )

    segment_index = int(
        np.searchsorted(
            cumulative_distances,
            distance_metres,
            side="right",
        ) - 1
    )

    segment_index = max(
        0,
        min(segment_index, len(coordinates) - 2),
    )

    start_distance = cumulative_distances[segment_index]
    end_distance = cumulative_distances[segment_index + 1]

    segment_length = end_distance - start_distance

    fraction = (
        0.0
        if segment_length <= 0
        else (distance_metres - start_distance) / segment_length
    )

    start_point = coordinates[segment_index]
    end_point = coordinates[segment_index + 1]

    return {
        "latitude": (
            start_point["latitude"]
            + fraction
            * (end_point["latitude"] - start_point["latitude"])
        ),
        "longitude": (
            start_point["longitude"]
            + fraction
            * (end_point["longitude"] - start_point["longitude"])
        ),
    }


def _simulate_pothole_response(
    distance_along_route_m,
    defect_config=None,
):
    """Generate sensor response for one or more configured potholes."""

    if not defect_config:
        return 0.0, False, None, None

    # Support both the existing single-pothole dictionary
    # and a list of pothole dictionaries.
    if isinstance(defect_config, dict):
        potholes = [defect_config]
    else:
        potholes = list(defect_config)

    total_sensor_response = 0.0
    encountered_potholes = []

    severity_amplitude = {
        "low": 2.0,
        "medium": 4.0,
        "high": 6.0,
    }

    for pothole in potholes:
        if pothole.get("defect_type") != "pothole":
            continue

        defect_distance_m = float(
            pothole["distance_along_route_m"]
        )
        severity = str(
            pothole.get("severity", "medium")
        ).lower()

        if severity not in severity_amplitude:
            raise ValueError(
                "Pothole severity must be low, medium, or high."
            )

        distance_from_defect_m = (
            distance_along_route_m - defect_distance_m
        )

        encounter_window_m = float(
            pothole.get("encounter_window_m", 2.0)
        )
        response_width_m = float(
            pothole.get("response_width_m", 1.0)
        )

        if response_width_m <= 0 or encounter_window_m < 0:
            raise ValueError(
                "Response width must be positive and "
                "encounter window non-negative."
            )

        # Each pothole contributes its own sensor response.
        response = severity_amplitude[severity] * np.exp(
            -0.5
            * (distance_from_defect_m / response_width_m) ** 2
        )

        total_sensor_response += float(response)

        if abs(distance_from_defect_m) <= encounter_window_m:
            encountered_potholes.append(
                (
                    abs(distance_from_defect_m),
                    pothole,
                )
            )

    if not encountered_potholes:
        return total_sensor_response, False, None, None

    # Use the nearest encountered pothole for the single
    # ground-truth ID/type fields on this sample.
    _, nearest_pothole = min(
        encountered_potholes,
        key=lambda item: item[0],
    )

    return (
        total_sensor_response,
        True,
        nearest_pothole.get("defect_id", "pothole_1"),
        nearest_pothole.get("defect_type", "pothole"),
    )




def _automatic_controls(
    speed_mps,
    remaining_distance_m,
    target_speed_mps=10.0,
    max_acceleration_mps2=3.0,
    max_braking_mps2=7.0,
):
    # Stop at the destination.
    if remaining_distance_m <= 1.0:
        return 0.0, 1.0

    # Calculate the speed difference.
    speed_error = target_speed_mps - speed_mps

    if speed_error > 0.1:
        throttle = min(
            speed_error / max(max_acceleration_mps2, 0.1),
            1.0,
        )
        brake = 0.0
    elif speed_error < -0.1:
        throttle = 0.0
        brake = min(
            abs(speed_error) / max(max_braking_mps2, 0.1),
            1.0,
        )
    else:
        throttle = 0.0
        brake = 0.0

    return throttle, brake



def _simulate_one_vehicle(
    vehicle_id,
    route_data,
    start_time,
    vehicle_start_offset_seconds=0,
    controls=None,
    config=None,
    defect_config=None,
):
    """Simulate vehicle motion with bounded acceleration and jerk."""

    coordinates, cumulative_distances, total_distance = route_data

    config = {
        **DEFAULT_SIMULATION_CONFIG,
        **(config or {}),
    }

    max_speed_mps = float(config["target_speed_kmh"]) / 3.6
    max_acceleration_mps2 = float(config["max_acceleration_mps2"])
    max_braking_mps2 = float(config["max_braking_mps2"])
    max_jerk_mps3 = float(config["max_jerk_mps3"])
    dt = float(config["timestep_seconds"])

    if (
        max_speed_mps <= 0
        or max_acceleration_mps2 <= 0
        or max_braking_mps2 <= 0
        or max_jerk_mps3 <= 0
        or dt <= 0
    ):
        raise ValueError(
            "All simulation limits and the time step must be positive."
        )

    elapsed_seconds = 0.0
    distance_along_route_m = 0.0
    speed_mps = 0.0
    previous_acceleration_mps2 = 0.0

    records = []

    position = _get_position_at_distance(
        coordinates,
        cumulative_distances,
        distance_along_route_m,
    )

    records.append({
        "vehicle_id": vehicle_id,
        "timestamp": start_time + timedelta(
            seconds=vehicle_start_offset_seconds
        ),
        "latitude": position["latitude"],
        "longitude": position["longitude"],
        "distance_along_route_m": distance_along_route_m,
        "speed_mps": speed_mps,
        "acceleration_mps2": 0.0,
        "jerk_mps3": 0.0,
        "ground_truth_jerk": False,
        "ground_truth_defect": False,
        "defect_id": None,
        "ground_truth_defect_type": None,
        "sensor_vertical_acceleration_mps2": 0.0,
        "throttle": 0.0,
        "brake": 0.0,
    })

    max_steps = max(
        1,
        int(total_distance / max(dt, 0.001) * 10) + 1000,
    )

    for _ in range(max_steps):
        remaining_distance_m = (
            total_distance - distance_along_route_m
        )

        if remaining_distance_m <= 0.001:
            break

        # Determine the target speed and driver controls.
        target_speed = max_speed_mps

        if controls is None:
            if defect_config:
                potholes = (
                    [defect_config]
                    if isinstance(defect_config, dict)
                    else list(defect_config)
                )

                # Slow down for an approaching pothole.
                # Do not reset the target speed because another
                # pothole has already been passed.
                for pothole in potholes:
                    if pothole.get("defect_type") != "pothole":
                        continue

                    pothole_distance = float(
                        pothole["distance_along_route_m"]
                    )

                    distance_to_pothole = (
                        pothole_distance - distance_along_route_m
                    )

                    if distance_to_pothole > 0:
                        braking_distance = max(
                            0.0,
                            (
                                speed_mps**2 - 3.0**2
                            ) / (2.0 * max_braking_mps2),
                        )

                        if distance_to_pothole <= max(
                            20.0, braking_distance
                        ):
                            target_speed = min(target_speed, 3.0)

            throttle, brake = _automatic_controls(
                speed_mps,
                remaining_distance_m,
                target_speed_mps=target_speed,
                max_acceleration_mps2=max_acceleration_mps2,
                max_braking_mps2=max_braking_mps2,
            )
        else:
            active_controls = controls or {}

            throttle = float(
                np.clip(
                    active_controls.get("throttle", 0.0),
                    0.0,
                    1.0,
                )
            )
            brake = float(
                np.clip(
                    active_controls.get("brake", 0.0),
                    0.0,
                    1.0,
                )
            )

        # Desired acceleration from the driver controls.
        raw_acceleration = (
            throttle * max_acceleration_mps2
            - brake * max_braking_mps2
            - DRAG_COEFFICIENT * speed_mps
        )

        # Limit speed only in automatic mode. Fixed controls remain
        # governed by the requested throttle and brake.
        if controls is None:
            speed_error = target_speed - speed_mps

            if speed_error <= 0:
                raw_acceleration = min(raw_acceleration, 0.0)
            elif speed_error < 2.0:
                raw_acceleration = min(
                    raw_acceleration,
                    2.0 * speed_error,
                )

        # Prevent negative speed and acceleration from rest
        # in the wrong direction.
        if speed_mps <= 0.0:
            raw_acceleration = max(raw_acceleration, 0.0)

        # Ease braking close to an upcoming pothole.
        if controls is None and defect_config:
            potholes = (
                [defect_config]
                if isinstance(defect_config, dict)
                else list(defect_config)
            )

            for pothole in potholes:
                if pothole.get("defect_type") != "pothole":
                    continue

                distance_to_pothole = (
                    float(pothole["distance_along_route_m"])
                    - distance_along_route_m
                )

                if 0.0 < distance_to_pothole <= 8.0:
                    acceleration_floor = (
                        -0.5 * distance_to_pothole / 8.0
                    )
                    raw_acceleration = max(
                        raw_acceleration,
                        acceleration_floor,
                    )

        # Initialize the jerk-limit bounds on every simulation step.
        max_change = max_jerk_mps3 * dt
        lower_acceleration = (
            previous_acceleration_mps2 - max_change
        )
        upper_acceleration = (
            previous_acceleration_mps2 + max_change
        )

        # Calculate acceleration BEFORE using it in the speed update.
        acceleration_mps2 = float(
            np.clip(
                raw_acceleration,
                lower_acceleration,
                upper_acceleration,
            )
        )

        # Calculate the next speed.
        next_speed_mps = max(
            0.0,
            speed_mps + acceleration_mps2 * dt,
        )

        # Calculate actual acceleration after enforcing non-negative speed.
        actual_acceleration_mps2 = (
            next_speed_mps - speed_mps
        ) / dt

        jerk_mps3 = (
            actual_acceleration_mps2
            - previous_acceleration_mps2
        ) / dt

        distance_step_m = (
            (speed_mps + next_speed_mps) / 2.0
        ) * dt

        next_distance_m = min(
            distance_along_route_m + distance_step_m,
            total_distance,
        )

        next_elapsed_seconds = elapsed_seconds + dt

        next_position = _get_position_at_distance(
            coordinates,
            cumulative_distances,
            next_distance_m,
        )

        (
            sensor_response,
            ground_truth_defect,
            defect_id,
            ground_truth_defect_type,
        ) = _simulate_pothole_response(
            next_distance_m,
            defect_config,
        )

        records.append({
            "vehicle_id": vehicle_id,
            "timestamp": start_time + timedelta(
                seconds=(
                    vehicle_start_offset_seconds
                    + next_elapsed_seconds
                )
            ),
            "latitude": next_position["latitude"],
            "longitude": next_position["longitude"],
            "distance_along_route_m": next_distance_m,
            "speed_mps": next_speed_mps,
            "acceleration_mps2": actual_acceleration_mps2,
            "jerk_mps3": jerk_mps3,
            "ground_truth_jerk": False,
            "ground_truth_defect": ground_truth_defect,
            "defect_id": defect_id,
            "ground_truth_defect_type": ground_truth_defect_type,
            "sensor_vertical_acceleration_mps2": sensor_response,
            "throttle": throttle,
            "brake": brake,
        })

        # Update state for the next step.
        previous_acceleration_mps2 = actual_acceleration_mps2
        speed_mps = next_speed_mps
        distance_along_route_m = next_distance_m
        elapsed_seconds = next_elapsed_seconds

        if distance_along_route_m >= total_distance:
            break

        if speed_mps <= 0 and throttle <= 0:
            break

    return records


def simulate_vehicles(
    vehicle_count,
    seed=None,
    route=None,
    controls=None,
    config=None,
    defect_config=None,
):
    """
    Simulate vehicles using a motion model.

    Existing calls remain compatible:
        samples, events = simulate_vehicles(5, route=route)

    Optional fixed controls:
        samples, events = simulate_vehicles(
            1,
            route=route,
            controls={"throttle": 0.5, "brake": 0.0},
        )

    Ground-truth defects are not generated in this phase.
    """

    if vehicle_count < 1:
        raise ValueError("vehicle_count must be at least 1.")

    # Keep seed in the function signature for compatibility with existing
    # callers. This motion model does not use random values.
    _ = seed

    if route is not None:
        route_data = _prepare_route(route)
    else:
        # Backward-compatible route for calls that don't supply a real route.
        # This is a straight geographic segment, not random coordinates.
        fallback_coordinates = [
            {
                "latitude": 13.0827,
                "longitude": 80.2707,
            },
            {
                "latitude": 13.0827,
                "longitude": 80.2760,
            },
        ]

        fallback_route = {"coordinates": fallback_coordinates}
        route_data = _prepare_route(fallback_route)

    start_time = datetime.now(ZoneInfo("Asia/Kolkata"))
    all_records = []

    
    for vehicle_index in range(vehicle_count):
        vehicle_id = f"VEH-{vehicle_index + 1:03d}"

        # Give each vehicle a different target speed.
        vehicle_config = {
            **DEFAULT_SIMULATION_CONFIG,
            **(config or {}),
        }
        
        vehicle_config["target_speed_kmh"] *= (
            1.16 - 0.04 * vehicle_index
        )



        vehicle_records = _simulate_one_vehicle(
            vehicle_id=vehicle_id,
            route_data=route_data,
            start_time=start_time,
            vehicle_start_offset_seconds=vehicle_index * 5,
            controls=controls,
            config=vehicle_config,
            defect_config=defect_config,
        )

        all_records.extend(vehicle_records)

    samples = pd.DataFrame(all_records)

    # This is a simple threshold flag for compatibility, not a confirmed
    # road-defect label. The detector pipeline will be refined separately.
    samples["jerk_detected"] = (
        samples["jerk_mps3"].abs() >= 2.5
    )

    events = samples[
        samples["jerk_detected"]
    ].copy()

    return samples, events
