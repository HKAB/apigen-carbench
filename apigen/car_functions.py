"""TEMP DATA: a self-contained simulated vehicle and its callable functions.

Each function mutates an in-memory ``VehicleState`` and returns a
JSON-serializable result reflecting the new state. ``CAR_FUNCTIONS`` maps the
API name (matching ``data/apis/car_apis.json``) to the implementation, so the
execution backend can dispatch by name.

Replace this module (and car_apis.json) with your real car APIs later. The
rest of the pipeline does not depend on these specific functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VALID_ZONES = {"driver", "passenger", "rear", "all"}
VALID_WINDOWS = {"driver", "passenger", "rear_left", "rear_right", "all"}
VALID_WINDOW_ACTIONS = {"open", "close"}
VALID_MEDIA_SOURCES = {"radio", "bluetooth", "usb", "streaming"}
VALID_SEATS = {"driver", "passenger", "rear_left", "rear_right"}
VALID_DRIVE_MODES = {"eco", "comfort", "sport", "off_road"}


@dataclass
class VehicleState:
    """Mutable in-memory state of the simulated car."""

    climate: dict[str, float] = field(
        default_factory=lambda: {"driver": 21.0, "passenger": 21.0, "rear": 21.0}
    )
    fan_speed: int = 2
    windows: dict[str, str] = field(
        default_factory=lambda: {w: "closed" for w in VALID_WINDOWS if w != "all"}
    )
    volume: int = 15
    media: dict[str, Any] = field(default_factory=lambda: {"source": "radio", "title": None})
    doors_locked: bool = True
    seats: dict[str, int] = field(default_factory=lambda: {s: 0 for s in VALID_SEATS})
    cruise: dict[str, Any] = field(default_factory=lambda: {"enabled": False, "speed": 0})
    drive_mode: str = "comfort"
    defroster: dict[str, bool] = field(default_factory=lambda: {"front": False, "rear": False})
    destination: str | None = None


def _zones(zone: str) -> list[str]:
    return ["driver", "passenger", "rear"] if zone == "all" else [zone]


def set_climate_temperature(state: VehicleState, *, zone: str, temperature: float) -> dict[str, Any]:
    if zone not in VALID_ZONES:
        raise ValueError(f"unknown zone '{zone}'; expected one of {sorted(VALID_ZONES)}")
    if not 16.0 <= float(temperature) <= 30.0:
        raise ValueError("temperature must be between 16 and 30 degrees Celsius")
    for z in _zones(zone):
        state.climate[z] = float(temperature)
    return {"status": "ok", "climate": dict(state.climate)}


def set_fan_speed(state: VehicleState, *, level: int) -> dict[str, Any]:
    if not 0 <= int(level) <= 5:
        raise ValueError("fan level must be between 0 and 5")
    state.fan_speed = int(level)
    return {"status": "ok", "fan_speed": state.fan_speed}


def control_window(state: VehicleState, *, window: str, action: str) -> dict[str, Any]:
    if window not in VALID_WINDOWS:
        raise ValueError(f"unknown window '{window}'")
    if action not in VALID_WINDOW_ACTIONS:
        raise ValueError(f"action must be one of {sorted(VALID_WINDOW_ACTIONS)}")
    targets = [w for w in state.windows] if window == "all" else [window]
    for w in targets:
        state.windows[w] = "open" if action == "open" else "closed"
    return {"status": "ok", "windows": dict(state.windows)}


def set_volume(state: VehicleState, *, level: int) -> dict[str, Any]:
    if not 0 <= int(level) <= 40:
        raise ValueError("volume must be between 0 and 40")
    state.volume = int(level)
    return {"status": "ok", "volume": state.volume}


def play_media(state: VehicleState, *, source: str, title: str | None = None) -> dict[str, Any]:
    if source not in VALID_MEDIA_SOURCES:
        raise ValueError(f"unknown media source '{source}'")
    state.media = {"source": source, "title": title}
    return {"status": "playing", "media": dict(state.media)}


def lock_doors(state: VehicleState, *, lock: bool) -> dict[str, Any]:
    state.doors_locked = bool(lock)
    return {"status": "ok", "doors_locked": state.doors_locked}


def adjust_seat(state: VehicleState, *, seat: str, recline: int) -> dict[str, Any]:
    if seat not in VALID_SEATS:
        raise ValueError(f"unknown seat '{seat}'")
    if not 0 <= int(recline) <= 60:
        raise ValueError("recline must be between 0 and 60 degrees")
    state.seats[seat] = int(recline)
    return {"status": "ok", "seat": seat, "recline": state.seats[seat]}


def set_cruise_control(state: VehicleState, *, enabled: bool, speed: int = 0) -> dict[str, Any]:
    if enabled and not 30 <= int(speed) <= 180:
        raise ValueError("cruise speed must be between 30 and 180 km/h when enabled")
    state.cruise = {"enabled": bool(enabled), "speed": int(speed) if enabled else 0}
    return {"status": "ok", "cruise": dict(state.cruise)}


def set_navigation_destination(
    state: VehicleState, *, address: str, avoid_tolls: bool = False
) -> dict[str, Any]:
    if not address or not address.strip():
        raise ValueError("address must not be empty")
    state.destination = address
    return {
        "status": "route_set",
        "destination": address,
        "avoid_tolls": bool(avoid_tolls),
        "eta_minutes": 12 + len(address) % 30,
    }


def set_drive_mode(state: VehicleState, *, mode: str) -> dict[str, Any]:
    if mode not in VALID_DRIVE_MODES:
        raise ValueError(f"unknown drive mode '{mode}'")
    state.drive_mode = mode
    return {"status": "ok", "drive_mode": state.drive_mode}


def toggle_defroster(state: VehicleState, *, zone: str, enabled: bool) -> dict[str, Any]:
    if zone not in {"front", "rear"}:
        raise ValueError("defroster zone must be 'front' or 'rear'")
    state.defroster[zone] = bool(enabled)
    return {"status": "ok", "defroster": dict(state.defroster)}


def find_charging_station(state: VehicleState, *, radius_km: float = 10.0) -> dict[str, Any]:
    if not 0 < float(radius_km) <= 100:
        raise ValueError("radius_km must be between 0 and 100")
    count = max(1, int(radius_km) // 3)
    return {
        "status": "ok",
        "radius_km": float(radius_km),
        "stations": [{"name": f"Station {i + 1}", "distance_km": round((i + 1) * 1.5, 1)} for i in range(count)],
    }


def vinfast_knowledge_base(
    state: VehicleState,
    *,
    key_word: str | None = None,
    rewrite_message: str | None = None,
    car_model_name: str | None = None,
    queries: str | None = None,
) -> dict[str, Any]:
    """Stub – PURPOSE"""
    return {
        "status": "ok",
        "key_word": key_word,
        "rewrite_message": rewrite_message,
        "car_model_name": car_model_name,
        "queries": queries,
    }

def weather_tool(state: VehicleState, *, rewrite_message: str | None = None) -> dict[str, Any]:
    """Stub – This tool is useful when you need to answer questions about the weather, implied weather-related questions, or indicator"""
    return {
        "status": "ok",
        "rewrite_message": rewrite_message,
    }

def movie_tool(
    state: VehicleState,
    *,
    movie_name: str | None = None,
    cinema: str | None = None,
    movie_time: str | None = None,
    movie_book_tickets: bool | None = None,
    movie_information: bool | None = None,
    movie_genres: str | None = None,
    rewrite_message: str | None = None,
) -> dict[str, Any]:
    """Stub – Use this function to retrieve or explore any kind of information related to movies. This includes but is not limited to:"""

    # movie_genres: one of 'Hành động', 'Kinh dị', 'Gia Đình', 'Hài', 'Tình cảm', 'Tâm lý', 'Viễn tưởng', 'Giả Tưởng', 'Ca nhạc', 'Tài Liệu', 'Khác'
    return {
        "status": "ok",
        "movie_name": movie_name,
        "cinema": cinema,
        "movie_time": movie_time,
        "movie_book_tickets": movie_book_tickets,
    }

def cooking_search(
    state: VehicleState,
    *,
    rewrite_message: str | None = None,
    location: str | None = None,
    queries: str | None = None,
) -> dict[str, Any]:
    """Stub – Useful for answering questions about general dishes and menus (EXCEPT local specialties of a country/region), including """
    return {
        "status": "ok",
        "rewrite_message": rewrite_message,
        "location": location,
        "queries": queries,
    }

def Vehicle_faults_and_operating_tips(
    state: VehicleState,
    *,
    action: str | None = None,
    object: str | None = None,
    number: float | None = None,
    key_word: str | None = None,
    rewrite_message: str | None = None,
    car_model_name: str | None = None,
    queries: str | None = None,
) -> dict[str, Any]:
    """Stub – Purpose - Diagnose real-world vehicle issues and provide troubleshooting / remedial advice."""

    # action: one of 'check_issues', 'check', 'kb_action', 'check_distance'
    # object: one of 'BATTERY', 'TIRE_PRESSURE'
    return {
        "status": "ok",
        "action": action,
        "object": object,
        "number": number,
        "key_word": key_word,
    }

def Frs_tool(
    state: VehicleState,
    *,
    rewrite_message: str | None = None,
    car_model_name: str | None = None,
    queries: str | None = None,
    new_frs: bool | None = None,
) -> dict[str, Any]:
    """Stub – Purpose - Provide software-version, firmware version, fota version information: release history, newly added functions, """
    return {
        "status": "ok",
        "rewrite_message": rewrite_message,
        "car_model_name": car_model_name,
        "queries": queries,
        "new_frs": new_frs,
    }

def tourism_search(
    state: VehicleState,
    *,
    rewrite_message: str | None = None,
    location: str | None = None,
    queries: str | None = None,
) -> dict[str, Any]:
    """Stub – useful for when you need to answer questions about travel: unique or popular restaurants & dishes, local specialties, si"""
    return {
        "status": "ok",
        "rewrite_message": rewrite_message,
        "location": location,
        "queries": queries,
    }

def zodiac_search(state: VehicleState, *, rewrite_message: str | None = None) -> dict[str, Any]:
    """Stub – Useful for answering questions related to zodiac signs, including general information (such as birth dates, personality """
    return {
        "status": "ok",
        "rewrite_message": rewrite_message,
    }

def vingroup_knowledge_base(state: VehicleState) -> dict[str, Any]:
    """Stub – Một công cụ chuyên về tập đoàn Vingroup, hữu ích khi đầu vào đề cập đến bất cứ điều gì liên quan đến Vingroup và các côn"""
    return {
        "status": "ok",
    }

# name -> callable. Keys must match data/apis/car_apis.json.
CAR_FUNCTIONS = {
    "set_climate_temperature": set_climate_temperature,
    "set_fan_speed": set_fan_speed,
    "control_window": control_window,
    "set_volume": set_volume,
    "play_media": play_media,
    "lock_doors": lock_doors,
    "adjust_seat": adjust_seat,
    "set_cruise_control": set_cruise_control,
    "set_navigation_destination": set_navigation_destination,
    "set_drive_mode": set_drive_mode,
    "toggle_defroster": toggle_defroster,
    "find_charging_station": find_charging_station,
    "vinfast_knowledge_base": vinfast_knowledge_base,
    "weather_tool": weather_tool,
    "movie_tool": movie_tool,
    "cooking_search": cooking_search,
    "Vehicle_faults_and_operating_tips": Vehicle_faults_and_operating_tips,
    "Frs_tool": Frs_tool,
    "tourism_search": tourism_search,
    "zodiac_search": zodiac_search,
    "vingroup_knowledge_base": vingroup_knowledge_base,
}
