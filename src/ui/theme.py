"""界面配色与等级样式，主窗口和对比窗口共用。"""

from __future__ import annotations

# 类别颜色，按 cls_id 轮转
CLASS_COLORS = [
    "#ffd60a", "#ff3b30", "#30d158", "#0a84ff", "#ff5c8d", "#bf5af2",
    "#64d2ff", "#ff9f0a", "#5e5ce6", "#ac8e68",
]

# 等级 -> (背景色, 显示文字)
LEVEL_STYLE = {
    0: ("#30d158", "合格"),
    1: ("#30d158", "I级 轻微"),
    2: ("#ffd60a", "II级 中度"),
    3: ("#ff9f0a", "III级 严重"),
    4: ("#ff3b30", "IV级 致命"),
}


def style_for(level: int) -> tuple[str, str]:
    return LEVEL_STYLE.get(level, ("#888", "未知"))


def badge_style(level: int) -> str:
    color, _ = style_for(level)
    return (
        f"background:{color}; color:#111; border-radius:8px; padding:14px;"
        "font-size:24px; font-weight:bold;"
    )
