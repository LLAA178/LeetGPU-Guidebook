from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ASSET_DIR = Path(__file__).resolve().parent / "assets"
DATA_PATH = ASSET_DIR / "occupancy-sweep-data.txt"

VARIANTS = ["low_regs", "mid_regs"]
VARIANT_LABEL = {
    "low_regs": "low_regs (16 regs/thread)",
    "mid_regs": "mid_regs (44 regs/thread)",
}
BLOCK_SIZES = [64, 128, 256, 512]
EXTRA_SMEMS = [0, 4096, 8192, 16384, 32768]
BLOCK_COLORS = {64: "#0f766e", 128: "#2563eb", 256: "#d97706", 512: "#dc2626"}
SMEM_COLORS = {
    "0KB": "#0f766e",
    "4KB": "#2563eb",
    "8KB": "#7c3aed",
    "16KB": "#d97706",
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
                "sm_count:",
                "max_threads_per_sm:",
                "warp_size:",
                "variant ",
            )
        ):
            continue
        parts = line.split()
        if len(parts) != 13:
            continue
        rows.append(
            {
                "variant": parts[0],
                "block_size": int(parts[1]),
                "regs_per_thread": int(parts[2]),
                "extra_smem_bytes": int(parts[4]),
                "active_blocks_per_sm": float(parts[6]),
                "active_warps_per_sm": float(parts[7]),
                "max_warps_per_sm": float(parts[8]),
                "theoretical_occupancy": float(parts[9]),
                "kernel_ms": float(parts[10]),
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


def point_label(value, metric):
    if metric == "kernel_ms":
        return f"{value:.3f}"
    return f"{value:.3f}"


def metric_title(metric):
    return "kernel time (ms)" if metric == "kernel_ms" else "theoretical occupancy"


def value_range(series):
    vals = [v for values in series.values() for v in values]
    vmin, vmax = min(vals), max(vals)
    if metric_title:
        pass
    if vmax - vmin < 1e-9:
        vmax = vmin + 1.0
    return vmin, vmax


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

    lx, ly = px1 - 110, py0 + 6
    draw.text((lx, ly - 22), legend_title, font=FONTS["small"], fill="#6b7280")
    for key in series.keys():
        draw.rounded_rectangle((lx, ly, lx + 14, ly + 14), radius=3, fill=colors[key])
        draw.text((lx + 22, ly - 2), str(key), font=FONTS["small"], fill="#374151")
        ly += 22


def render_page(out_name, title, subtitle, metric, mode):
    width, height = 1700, 760
    image = Image.new("RGB", (width, height), "#fafaf9")
    draw = ImageDraw.Draw(image)
    draw.text((50, 30), title, font=FONTS["title"], fill="#111827")
    draw.text((50, 72), subtitle, font=FONTS["body"], fill="#4b5563")

    rects = [(50, 120, 820, 700), (880, 120, 1650, 700)]
    for variant, rect in zip(VARIANTS, rects):
        variant_rows = [row for row in ROWS if row["variant"] == variant]
        if mode == "smem":
            x_labels = ["0KB", "4KB", "8KB", "16KB", "32KB"]
            series = {}
            for block in BLOCK_SIZES:
                ordered = [
                    next(
                        row
                        for row in variant_rows
                        if row["block_size"] == block and row["extra_smem_bytes"] == smem
                    )
                    for smem in EXTRA_SMEMS
                ]
                series[block] = [row[metric] for row in ordered]
            draw_panel(
                draw,
                rect,
                VARIANT_LABEL[variant],
                x_labels,
                series,
                BLOCK_COLORS,
                metric,
                "extra_smem_bytes",
                "block_size",
            )
        else:
            x_labels = [str(block) for block in BLOCK_SIZES]
            series = {}
            for smem in EXTRA_SMEMS:
                key = f"{smem // 1024 if smem else 0}KB"
                ordered = [
                    next(
                        row
                        for row in variant_rows
                        if row["block_size"] == block and row["extra_smem_bytes"] == smem
                    )
                    for block in BLOCK_SIZES
                ]
                series[key] = [row[metric] for row in ordered]
            draw_panel(
                draw,
                rect,
                VARIANT_LABEL[variant],
                x_labels,
                series,
                SMEM_COLORS,
                metric,
                "block_size",
                "extra_smem",
            )

    image.save(ASSET_DIR / out_name)


def main():
    render_page(
        "occupancy-sweep-kernel-vs-shared-mem.png",
        "Occupancy Sweep: Shared Memory vs Kernel Time",
        "Y-axis = kernel time (ms). Each point is labeled. Each panel fixes register pressure.",
        "kernel_ms",
        "smem",
    )
    render_page(
        "occupancy-sweep-occupancy-vs-shared-mem.png",
        "Occupancy Sweep: Shared Memory vs Theoretical Occupancy",
        "Y-axis = theoretical occupancy per SM. Each point is labeled. Each panel fixes register pressure.",
        "theoretical_occupancy",
        "smem",
    )
    render_page(
        "occupancy-sweep-kernel-vs-block-size.png",
        "Occupancy Sweep: Block Size vs Kernel Time",
        "Y-axis = kernel time (ms). Each point is labeled. Each panel fixes register pressure.",
        "kernel_ms",
        "block",
    )
    render_page(
        "occupancy-sweep-occupancy-vs-block-size.png",
        "Occupancy Sweep: Block Size vs Theoretical Occupancy",
        "Y-axis = theoretical occupancy per SM. Each point is labeled. Each panel fixes register pressure.",
        "theoretical_occupancy",
        "block",
    )


if __name__ == "__main__":
    main()
