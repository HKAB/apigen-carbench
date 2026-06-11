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
    """Simulated downstream KB response for VinFast-related queries."""
    topic = (queries or key_word or rewrite_message or "thông tin tổng quan").strip()
    model = (car_model_name or "VinFast").strip()
    response = (
        f"Đã tra cứu kiến thức VinFast cho '{topic}' ({model}). "
        "Gợi ý: kiểm tra hướng dẫn sử dụng, lịch bảo dưỡng định kỳ và bản cập nhật phần mềm mới nhất."
    )
    return {
        "status": "ok",
        "response": response,
        "topic": topic,
        "car_model_name": model,
        "insights": [
            "Bảo dưỡng định kỳ giúp tối ưu hiệu suất pin/động cơ.",
            "Nên cập nhật phần mềm khi xe ở trạng thái đỗ an toàn.",
            "Theo dõi cảnh báo trên cụm đồng hồ để xử lý sớm.",
        ],
    }


def weather_tool(state: VehicleState, *, rewrite_message: str | None = None) -> dict[str, Any]:
    """Simulated weather service response."""
    msg = (rewrite_message or "").strip()
    signal = len(msg) if msg else 17
    temp_c = 24 + (signal % 8)
    rain_prob = (signal * 9) % 100
    condition = "nhiều mây" if rain_prob >= 40 else "trời quang"
    response = (
        f"Nhiệt độ hiện tại khoảng {temp_c}°C, {condition}, "
        f"khả năng có mưa là {rain_prob}%."
    )
    return {
        "status": "ok",
        "response": response,
        "forecast": {
            "temperature_c": temp_c,
            "condition": condition,
            "rain_probability_percent": rain_prob,
        },
        # "rewrite_message": rewrite_message,
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
        response = f"Đã giữ chỗ tạm thời cho '{name}' tại {cin}, suất {show_time}. Vui lòng xác nhận thanh toán."
        status = "booking_pending"
    elif movie_information:
        response = (
            f"'{name}' ({genre}) hiện có lịch chiếu tại {cin} vào {show_time}. "
            "Đánh giá khán giả: 8.2/10, thời lượng dự kiến: 118 phút."
        )
        status = "ok"
    else:
        response = f"Tìm thấy lịch chiếu phù hợp cho '{name}' tại {cin} vào {show_time}."
        status = "ok"

    return {
        "status": status,
        "response": response,
        "movie_name": name,
        "cinema": cin,
        "movie_time": show_time,
        "movie_genres": genre,
        "movie_book_tickets": bool(movie_book_tickets),
        "movie_information": bool(movie_information),
        # "rewrite_message": rewrite_message,
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
    response = f"Dựa trên yêu cầu '{q}', gợi ý 3 món phù hợp tại {area}: {', '.join(dishes)}."
    return {
        "status": "ok",
        "response": response,
        "location": area,
        "queries": q,
        "suggested_dishes": dishes,
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
        response = (
            f"{model}: Hệ thống pin đang ở mức an toàn. "
            "Khuyến nghị giữ pin trong khoảng 20%–80% cho sử dụng hằng ngày."
        )
        recommendations = ["Hạn chế sạc nhanh liên tục", "Kiểm tra cổng sạc định kỳ"]
    elif target == "TIRE_PRESSURE":
        psi = number if number is not None else 32
        response = (
            f"{model}: Áp suất lốp hiện tại khoảng {psi} PSI. "
            "Nên duy trì theo khuyến nghị nhà sản xuất để tối ưu độ bám và tiết kiệm năng lượng."
        )
        recommendations = ["Kiểm tra lốp khi nguội", "Đảo lốp mỗi 8.000–10.000 km"]
    else:
        response = (
            f"{model}: Đã tiếp nhận yêu cầu chẩn đoán ({act}). "
            "Vui lòng theo dõi cảnh báo trên màn hình trung tâm để xử lý kịp thời."
        )
        recommendations = ["Đọc mã lỗi OBD nếu có", "Liên hệ xưởng dịch vụ khi lỗi lặp lại"]

    return {
        "status": "ok",
        "response": response,
        "action": act,
        "object": target,
        "number": number,
        "key_word": key_word,
        "queries": queries,
        # "rewrite_message": rewrite_message,
        "recommendations": recommendations,
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
    response = (
        f"{model}: phiên bản hiện hành {latest['software_version']} / {latest['firmware_version']}. "
        "Bản cập nhật tối ưu điều hòa và cải thiện độ mượt giao diện."
    )
    if new_frs:
        response += " Có bản phát hành mới, khuyến nghị cập nhật khi xe đang đỗ."

    return {
        "status": "ok",
        "response": response,
        "car_model_name": model,
        "queries": queries,
        "new_frs": bool(new_frs),
        "release_info": latest,
        # "rewrite_message": rewrite_message,
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
    response = f"Gợi ý lịch trình tại {area} cho '{q}': {', '.join(places)}."
    return {
        "status": "ok",
        "response": response,
        "location": area,
        "queries": q,
        "highlights": places,
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

    response = (
        f"Tổng quan {sign}: hôm nay phù hợp để hoàn thành việc tồn đọng, "
        "ưu tiên giao tiếp rõ ràng và giữ nhịp sinh hoạt ổn định."
    )
    return {
        "status": "ok",
        "response": response,
        # "rewrite_message": rewrite_message,
        "zodiac": sign,
        "lucky_numbers": [3, 7, 21],
    }


def vingroup_knowledge_base(state: VehicleState) -> dict[str, Any]:
    """Simulated Vingroup KB response."""
    return {
        "status": "ok",
        "response": (
            "Thông tin nhanh về hệ sinh thái Vingroup: công nghệ, công nghiệp, "
            "thương mại dịch vụ và các công ty thành viên liên quan."
        ),
        "highlights": ["Vingroup", "VinFast", "Vinpearl", "Vinhomes", "Vinmec", "Vinschool"],
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
