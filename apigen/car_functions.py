"""TEMP DATA: a self-contained simulated vehicle and its callable functions.

Each function mutates an in-memory ``VehicleState`` and returns a
JSON-serializable result reflecting the new state. ``CAR_FUNCTIONS`` maps the
API name (matching ``data/apis/car_apis.json``) to the implementation, so the
execution backend can dispatch by name.

Replace this module (and car_apis.json) with your real car APIs later. The
rest of the pipeline does not depend on these specific functions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any
from urllib import error, request

VALID_ZONES = {"driver", "passenger", "rear", "all"}
VALID_WINDOWS = {"driver", "passenger", "rear_left", "rear_right", "all"}
VALID_WINDOW_ACTIONS = {"open", "close"}
VALID_MEDIA_SOURCES = {"radio", "bluetooth", "usb", "streaming"}
VALID_SEATS = {"driver", "passenger", "rear_left", "rear_right"}
VALID_DRIVE_MODES = {"eco", "comfort", "sport", "off_road"}


@lru_cache(maxsize=1)
def _llm_runtime() -> tuple[str, str, str]:
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    api_token = os.getenv("LLM_API_TOKEN", "EMPTY")
    model = os.getenv("GENERATOR_MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    return base_url, api_token, model


def _llm_tool_answer(*, tool_name: str, user_need: str, context: dict[str, Any]) -> str:
    """Generate a short, informative assistant answer for info/RAG-like tools.

    Uses the same serving config/model as the generator stage via env vars.
    Falls back to a compact deterministic response if LLM is unavailable.
    """
    base_url, api_token, model = _llm_runtime()
    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là trợ lý trên xe. Trả lời ngắn gọn, đúng trọng tâm, hữu ích, và giải quyết đầy đủ "
                "nhu cầu người dùng. Không dài dòng. Ưu tiên tiếng Việt tự nhiên."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Công cụ: {tool_name}\n"
                f"Nhu cầu người dùng: {user_need or 'Yêu cầu thông tin tổng quát'}\n"
                f"Ngữ cảnh: {json.dumps(context, ensure_ascii=False)}\n"
                "Hãy trả lời ngắn gọn (1-3 câu), rõ ràng, có thông tin hành động khi phù hợp."
            ),
        },
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 180,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_token}",
    }
    req = request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"].strip()
        if content:
            return content
    except (error.URLError, error.HTTPError, TimeoutError, KeyError, IndexError, json.JSONDecodeError):
        pass

    # Safe fallback to keep execution deterministic even when LLM endpoint fails.
    brief_need = (user_need or "Yêu cầu thông tin tổng quát").strip()
    return f"Đã ghi nhận yêu cầu: {brief_need}. Hiện chưa lấy được phản hồi chi tiết, vui lòng thử lại sau ít phút."


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
    """Simulated downstream KB response for VinFast-related queries."""
    topic = (queries or key_word or rewrite_message or "thông tin tổng quan").strip()
    model = (car_model_name or "VinFast").strip()
    response = _llm_tool_answer(
        tool_name="vinfast_knowledge_base",
        user_need=topic,
        context={"car_model_name": model, "topic": topic},
    )
    return {
        "status": "ok",
        "response": response
    }


def weather_tool(state: VehicleState, *, rewrite_message: str | None = None) -> dict[str, Any]:
    """Simulated weather service response."""
    msg = (rewrite_message or "").strip()
    signal = len(msg) if msg else 17
    temp_c = 24 + (signal % 8)
    rain_prob = (signal * 9) % 100
    condition = "nhiều mây" if rain_prob >= 40 else "trời quang"
    response = _llm_tool_answer(
        tool_name="weather_tool",
        user_need=msg or "thời tiết hiện tại",
        context={
            "temperature_c": temp_c,
            "condition": condition,
            "rain_probability_percent": rain_prob,
        },
    )
    return {
        "status": "ok",
        "response": response
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
    """Simulated movie search/booking response."""
    name = movie_name or "phim đang chiếu"
    cin = cinema or "rạp gần bạn"
    show_time = movie_time or "20:00"
    genre = movie_genres or "Khác"

    if movie_book_tickets:
        response = _llm_tool_answer(
            tool_name="movie_tool",
            user_need=rewrite_message or f"Đặt vé phim {name}",
            context={
                "movie_name": name,
                "cinema": cin,
                "movie_time": show_time,
                "movie_genres": genre,
                "intent": "booking",
            },
        )
        status = "booking_pending"
    elif movie_information:
        response = _llm_tool_answer(
            tool_name="movie_tool",
            user_need=rewrite_message or f"Thông tin phim {name}",
            context={
                "movie_name": name,
                "cinema": cin,
                "movie_time": show_time,
                "movie_genres": genre,
                "intent": "information",
            },
        )
        status = "ok"
    else:
        response = _llm_tool_answer(
            tool_name="movie_tool",
            user_need=rewrite_message or f"Lịch chiếu phim {name}",
            context={
                "movie_name": name,
                "cinema": cin,
                "movie_time": show_time,
                "movie_genres": genre,
                "intent": "showtimes",
            },
        )
        status = "ok"

    return {
        "status": status,
        "response": response,
    }


def cooking_search(
    state: VehicleState,
    *,
    rewrite_message: str | None = None,
    location: str | None = None,
    queries: str | None = None,
) -> dict[str, Any]:
    """Simulated cooking and dish suggestion response."""
    area = location or "khu vực của bạn"
    q = queries or rewrite_message or "món dễ nấu"
    dishes = ["Cơm chiên hải sản", "Canh nấm đậu hũ", "Gà áp chảo sốt tiêu đen"]
    response = _llm_tool_answer(
        tool_name="cooking_search",
        user_need=q,
        context={"location": area, "suggested_dishes": dishes},
    )
    return {
        "status": "ok",
        "response": response
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
    """Simulated diagnostics and operation-tips response."""
    model = car_model_name or "VinFast"
    target = object or "GENERAL"
    act = action or "check_issues"

    if target == "BATTERY":
        response = _llm_tool_answer(
            tool_name="Vehicle_faults_and_operating_tips",
            user_need=queries or rewrite_message or key_word or "kiểm tra pin",
            context={"car_model_name": model, "action": act, "object": target, "number": number},
        )
    elif target == "TIRE_PRESSURE":
        psi = number if number is not None else 32
        response = _llm_tool_answer(
            tool_name="Vehicle_faults_and_operating_tips",
            user_need=queries or rewrite_message or key_word or "kiểm tra áp suất lốp",
            context={
                "car_model_name": model,
                "action": act,
                "object": target,
                "tire_pressure_psi": psi,
            },
        )
    else:
        response = _llm_tool_answer(
            tool_name="Vehicle_faults_and_operating_tips",
            user_need=queries or rewrite_message or key_word or "chẩn đoán lỗi xe",
            context={"car_model_name": model, "action": act, "object": target, "number": number},
        )

    return {
        "status": "ok",
        "response": response
    }


def Frs_tool(
    state: VehicleState,
    *,
    rewrite_message: str | None = None,
    car_model_name: str | None = None,
    queries: str | None = None,
    new_frs: bool | None = None,
) -> dict[str, Any]:
    """Simulated FRS/firmware release response."""
    model = car_model_name or "VinFast"
    latest = {
        "software_version": "v3.2.1",
        "firmware_version": "fw-1.14.0",
        "fota_version": "fota-2026.04",
    }
    response = _llm_tool_answer(
        tool_name="Frs_tool",
        user_need=queries or rewrite_message or "cập nhật phần mềm xe",
        context={"car_model_name": model, "new_frs": bool(new_frs), "release_info": latest},
    )
    if new_frs:
        response += " Khuyến nghị cập nhật khi xe đang đỗ an toàn."

    return {
        "status": "ok",
        "response": response
    }


def tourism_search(
    state: VehicleState,
    *,
    rewrite_message: str | None = None,
    location: str | None = None,
    queries: str | None = None,
) -> dict[str, Any]:
    """Simulated tourism recommendation response."""
    area = location or "điểm đến bạn quan tâm"
    q = queries or rewrite_message or "du lịch cuối tuần"
    places = ["Bảo tàng địa phương", "Phố ẩm thực trung tâm", "Khu ngắm cảnh ven sông"]
    response = _llm_tool_answer(
        tool_name="tourism_search",
        user_need=q,
        context={"location": area, "highlights": places},
    )
    return {
        "status": "ok",
        "response": response
    }


def zodiac_search(state: VehicleState, *, rewrite_message: str | None = None) -> dict[str, Any]:
    """Simulated zodiac information response."""
    text = (rewrite_message or "").lower()
    sign = "cung của bạn"
    for s in [
        "bạch dương", "kim ngưu", "song tử", "cự giải", "sư tử", "xử nữ",
        "thiên bình", "bọ cạp", "nhân mã", "ma kết", "bảo bình", "song ngư",
    ]:
        if s in text:
            sign = s.title()
            break

    response = _llm_tool_answer(
        tool_name="zodiac_search",
        user_need=rewrite_message or sign,
        context={"zodiac": sign, "lucky_numbers": [3, 7, 21]},
    )
    return {
        "status": "ok",
        "response": response
    }


def vingroup_knowledge_base(state: VehicleState) -> dict[str, Any]:
    """Simulated Vingroup KB response."""
    response = _llm_tool_answer(
        tool_name="vingroup_knowledge_base",
        user_need="thông tin hệ sinh thái Vingroup",
        context={"highlights": ["Vingroup", "VinFast", "Vinpearl", "Vinhomes", "Vinmec", "Vinschool"]},
    )
    return {
        "status": "ok",
        "response": response
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
