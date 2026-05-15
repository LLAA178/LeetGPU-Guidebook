from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ASSET_DIR = Path(__file__).resolve().parent / "assets"
DATA_PATH = ASSET_DIR / "occupancy-ilp-matrix-data.txt"

VARIANTS = ["dep_chain", "ilp2", "ilp4"]
VARIANT_LABEL = {
    "dep_chain": "dep_chain (1 chain x 4 FMA)",
    "ilp2": "ilp2 (2 chains x 2 FMA)",
    "ilp4": "ilp4 (4 chains x 1 FMA)",
}
EXTRA_SMEMS = [0, 8192, 16384, 32768]
VARIANT_COLORS = {"dep_chain": "#dc2626", "ilp2": "#d97706", "ilp4": "#0f766e"}
SMEM_COLORS = {
    "0KB": "#0f766e",
    "8KB": "#2563eb",
    "16KB": "#7c3aed",
    "32KB": "#d97706",
    "32KB": "#dc2626",
}


def load_fonts():
    try:
        return {
            "title": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30),
            "subtitle": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20),
            "body": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18),
            "small": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14),
        }
    except Exception:
        fallback = ImageFont.load_default()
        return {
            "title": fallback,
            "subtitle": fallback,
            "body": fallback,
            "small": fallback,
        }


FONTS = load_fonts()


def parse_rows():
    rows = []
    for line in DATA_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(
            (
                "device:",
                "num_elems:",
                "iters:",
                "repeats:",
                "block_size:",
                "sm_count:",
                "max_threads_per_sm:",
                "warp_size:",
                "variant ",
                "experiment:",
            )
        ):
            continue
        parts = line.split()
        if len(parts) != 12:
            continue
        rows.append(
            {
                "variant": parts[0],
                "extra_smem_bytes": int(parts[1]),
                "regs_per_thread": int(parts[2]),
                "active_blocks_per_sm": float(parts[4]),
                "active_warps_per_sm": float(parts[5]),
                "max_warps_per_sm": float(parts[6]),
                "theoretical_occupancy": float(parts[7]),
                "kernel_ms": float(parts[8]),
                "speedup_vs_dep_0KB": float(parts[9]),
            }
        )
    return rows


ROWS = parse_rows()


def text_size(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_vertical_text(draw, x, y0, y1, text):
    step = 14
    total = len(text) * step
    y = (y0 + y1 - total) / 2
    for ch in text:
        draw.text((x, y), ch, font=FONTS["small"], fill="#374151")
        y += step


def metric_title(metric):
    if metric == "kernel_ms":
        return "kernel time (ms)"
    if metric == "theoretical_occupancy":
        return "theoretical occupancy"
    return "speedup vs dep_chain 0KB"


def value_range(series):
    vals = [v for values in series.values() for v in values]
    vmin, vmax = min(vals), max(vals)
    if vmax - vmin < 1e-9:
        vmax = vmin + 1.0
    return vmin, vmax


def point_label(value, metric):
    return f"{value:.3f}"


def draw_panel(draw, rect, panel_title, x_labels, series, colors, metric, x_title, legend_title):
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle(rect, radius=18, fill="#ffffff", outline="#d1d5db", width=2)
    draw.text((x0 + 18, y0 + 14), panel_title, font=FONTS["subtitle"], fill="#111827")
    draw.text((x0 + 18, y0 + 42), metric_title(metric), font=FONTS["small"], fill="#6b7280")

    px0, py0, px1, py1 = x0 + 88, y0 + 76, x1 - 24, y1 - 70
    y_min, y_max = value_range(series)
    span = y_max - y_min

    for i in range(5):
        frac = i / 4
        yy = py1 - (py1 - py0) * frac
        draw.line((px0, yy, px1, yy), fill="#eef2f7", width=1)
        value = y_min + span * frac
        label = f"{value:.2f}" if metric == "kernel_ms" else f"{value:.3f}"
        tw, th = text_size(draw, label, FONTS["small"])
        draw.text((px0 - 12 - tw, yy - th / 2), label, font=FONTS["small"], fill="#6b7280")

    xs = []
    for i, label in enumerate(x_labels):
        x = px0 + (px1 - px0) * (i / (len(x_labels) - 1) if len(x_labels) > 1 else 0.5)
        xs.append(x)
        tw, _ = text_size(draw, label, FONTS["small"])
        draw.text((x - tw / 2, py1 + 10), label, font=FONTS["small"], fill="#6b7280")

    for key, values in series.items():
        points = []
        for x, value in zip(xs, values):
            y = py1 - (py1 - py0) * ((value - y_min) / span)
            points.append((x, y, value))
        draw.line([(x, y) for x, y, _ in points], fill=colors[key], width=3)
        for x, y, value in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colors[key], outline="white", width=1)
            label = point_label(value, metric)
            tw, th = text_size(draw, label, FONTS["small"])
            draw.text((x - tw / 2, y - th - 6), label, font=FONTS["small"], fill=colors[key])

    tw, _ = text_size(draw, x_title, FONTS["body"])
    draw.text(((px0 + px1 - tw) / 2, y1 - 34), x_title, font=FONTS["body"], fill="#374151")
    draw_vertical_text(draw, x0 + 18, py0, py1, metric_title(metric))

    lx, ly = px1 - 170, py0 + 6
    draw.text((lx, ly - 22), legend_title, font=FONTS["small"], fill="#6b7280")
    for key in series.keys():
        draw.rounded_rectangle((lx, ly, lx + 14, ly + 14), radius=3, fill=colors[key])
        draw.text((lx + 22, ly - 2), str(key), font=FONTS["small"], fill="#374151")
        ly += 22


def render_metric_vs_smem(metric, out_name, title, subtitle):
    width, height = 1700, 760
    image = Image.new("RGB", (width, height), "#fafaf9")
    draw = ImageDraw.Draw(image)
    draw.text((50, 30), title, font=FONTS["title"], fill="#111827")
    draw.text((50, 72), subtitle, font=FONTS["body"], fill="#4b5563")

    rect = (50, 120, 1650, 700)
    x_labels = ["0KB", "8KB", "16KB", "32KB"]
    series = {}
    for variant in VARIANTS:
        ordered = [
            next(row for row in ROWS if row["variant"] == variant and row["extra_smem_bytes"] == smem)
            for smem in EXTRA_SMEMS
        ]
        series[VARIANT_LABEL[variant]] = [row[metric] for row in ordered]
    draw_panel(draw, rect, "All ILP variants", x_labels, series, {VARIANT_LABEL[k]: VARIANT_COLORS[k] for k in VARIANTS}, metric, "extra_smem_bytes", "kernel shape")
    image.save(ASSET_DIR / out_name)


def render_metric_vs_variant(metric, out_name, title, subtitle):
    width, height = 1700, 760
    image = Image.new("RGB", (width, height), "#fafaf9")
    draw = ImageDraw.Draw(image)
    draw.text((50, 30), title, font=FONTS["title"], fill="#111827")
    draw.text((50, 72), subtitle, font=FONTS["body"], fill="#4b5563")

    rect = (50, 120, 1650, 700)
    x_labels = ["dep_chain", "ilp2", "ilp4"]
    series = {}
    for smem in EXTRA_SMEMS:
        key = f"{smem // 1024 if smem else 0}KB"
        ordered = [next(row for row in ROWS if row["variant"] == variant and row["extra_smem_bytes"] == smem) for variant in VARIANTS]
        series[key] = [row[metric] for row in ordered]
    draw_panel(draw, rect, "All occupancy points", x_labels, series, SMEM_COLORS, metric, "ILP shape", "extra_smem")
    image.save(ASSET_DIR / out_name)


def main():
    render_metric_vs_smem(
        "kernel_ms",
        "occupancy-ilp-matrix-kernel-vs-shared-mem.png",
        "Occupancy x ILP Matrix: Shared Memory vs Kernel Time",
        "Each line fixes ILP shape. Y-axis = kernel time (ms).",
    )
    render_metric_vs_smem(
        "theoretical_occupancy",
        "occupancy-ilp-matrix-occupancy-vs-shared-mem.png",
        "Occupancy x ILP Matrix: Shared Memory vs Theoretical Occupancy",
        "Each line fixes ILP shape. Y-axis = theoretical occupancy per SM.",
    )
    render_metric_vs_variant(
        "kernel_ms",
        "occupancy-ilp-matrix-kernel-vs-ilp.png",
        "Occupancy x ILP Matrix: ILP Shape vs Kernel Time",
        "Each line fixes extra shared memory. Y-axis = kernel time (ms).",
    )
    render_metric_vs_variant(
        "speedup_vs_dep_0KB",
        "occupancy-ilp-matrix-speedup-vs-ilp.png",
        "Occupancy x ILP Matrix: ILP Shape vs Speedup",
        "Each line fixes extra shared memory. Y-axis = speedup relative to dep_chain at 0KB.",
    )


if __name__ == "__main__":
    main()
