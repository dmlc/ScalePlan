# ANSI color codes
_BOLD = "\033[1m"
_END = "\033[0m"
_GREEN = "\033[92m"
_BLUE = "\033[94m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_RED = "\033[91m"
_MAGENTA = "\033[95m"
_GRAY = "\033[90m"

# Export color codes for external use
__all__ = ["_BOLD", "_END", "_GREEN", "_BLUE", "_YELLOW", "_CYAN", "_RED", "_MAGENTA", "_GRAY"]


def print_bold(text: str) -> None:
    print(_BOLD + text + _END)


def print_kv(key: str, value: str, key_width: int = 25, color: str = "") -> None:
    """Print a key-value pair with nice formatting."""
    formatted_key = f"{key}:".ljust(key_width)
    if color:
        print(f"  {_GRAY}{formatted_key}{_END} {color}{value}{_END}")
    else:
        print(f"  {_GRAY}{formatted_key}{_END} {value}")


def print_metric(name: str, value: str, unit: str = "", highlight: bool = False) -> None:
    """Print a metric with optional highlighting."""
    color = _GREEN if highlight else _YELLOW
    if unit:
        print(f"  {name}: {color}{value}{_END} {_GRAY}{unit}{_END}")
    else:
        print(f"  {name}: {color}{value}{_END}")


def print_section_separator() -> None:
    """Print a light section separator."""
    print(f"{_GRAY}{'·' * 80}{_END}")


def print_success(text: str) -> None:
    """Print success message in green."""
    print(f"{_GREEN}✓ {text}{_END}")


def print_warning(text: str) -> None:
    """Print warning message in yellow."""
    print(f"{_YELLOW}⚠ {text}{_END}")


def print_info(text: str) -> None:
    """Print info message in blue."""
    print(f"{_BLUE}ℹ {text}{_END}")


def format_number(num: float, decimals: int = 2) -> str:
    """Format large numbers with appropriate units."""
    if num >= 1e12:
        return f"{num / 1e12:.{decimals}f}T"
    elif num >= 1e9:
        return f"{num / 1e9:.{decimals}f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.{decimals}f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.{decimals}f}K"
    else:
        return f"{num:.{decimals}f}"


def print_h1_header(section_name: str) -> None:
    print()
    print(f"{_CYAN}{'=' * 80}{_END}")
    print(f"{_BOLD}{_CYAN} {section_name}{_END}")
    print(f"{_CYAN}{'=' * 80}{_END}")


def print_h2_header(section_name: str) -> None:
    print()
    print(f"{_BOLD}{_BLUE}▶ {section_name}{_END}")
    print(f"{_BLUE}{'-' * 60}{_END}")


def get_color_by_intensity(value: float, high_threshold: float, medium_threshold: float) -> str:
    """Get color based on intensity value.

    Args:
        value: The value to evaluate
        high_threshold: Values >= this are considered high (red)
        medium_threshold: Values >= this are considered medium (yellow)

    Returns:
        Color code string (red for high, yellow for medium, green for low)
    """
    if value >= high_threshold:
        return _RED
    elif value >= medium_threshold:
        return _YELLOW
    else:
        return _GREEN


def get_color_by_percentage(percentage: float) -> str:
    """Get color based on percentage value.

    Common thresholds for percentages:
    - >= 30%: Red (dominant)
    - >= 15%: Yellow (significant)
    - < 15%: Green (small)

    Args:
        percentage: The percentage value (0-100)

    Returns:
        Color code string
    """
    return get_color_by_intensity(percentage, 30, 15)


def get_color_by_time_ms(time_ms: float) -> str:
    """Get color based on time in milliseconds.

    Common thresholds for compute time:
    - >= 0.5ms: Red (compute-heavy)
    - >= 0.2ms: Yellow (medium)
    - < 0.2ms: Green (light)

    Args:
        time_ms: Time in milliseconds

    Returns:
        Color code string
    """
    return get_color_by_intensity(time_ms, 0.5, 0.2)


def get_color_for_component_percentage(percentage: float) -> str:
    """Get color for transformer block component based on percentage.

    Uses specialized thresholds for block-level components:
    - >= 20%: Red (dominant within block)
    - >= 10%: Yellow (significant within block)
    - < 10%: Green (small within block)

    Args:
        percentage: The percentage value (0-100)

    Returns:
        Color code string
    """
    return get_color_by_intensity(percentage, 20, 10)
