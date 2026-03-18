"""Optimize edilmiş yerleştirmeyi DXF ve PLT (HPGL) formatında dışa aktar — cutter'a göndermeye hazır."""
import numpy as np


def export_dxf(placements: list, pieces: list, bin_width: float,
               used_length: float, output_path: str):
    """Yerleştirmeyi DXF formatında kaydet.

    Args:
        placements: decoder çıktısı, her biri {piece_id, polygon, x, y, rotation}
        pieces: orijinal parça listesi
        bin_width: kumaş eni (mm)
        used_length: kullanılan uzunluk (mm)
        output_path: çıktı dosya yolu
    """
    lines = []
    # Header
    lines.append("0\nSECTION\n2\nHEADER\n9\n$INSUNITS\n70\n4\n0\nENDSEC")

    # Entities section
    lines.append("0\nSECTION\n2\nENTITIES")

    # Kumaş sınırı (dikdörtgen)
    lines.append(_dxf_polyline([
        (0, 0), (bin_width, 0), (bin_width, used_length), (0, used_length), (0, 0)
    ], layer="BIN", closed=True))

    # Her parça
    for pl in placements:
        coords = list(pl["polygon"].exterior.coords)
        pid = pl["piece_id"]
        layer = f"PIECE_{pid}"
        lines.append(_dxf_polyline(coords, layer=layer, closed=True))

    lines.append("0\nENDSEC\n0\nEOF")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _dxf_polyline(coords: list, layer: str = "0", closed: bool = False) -> str:
    """DXF LWPOLYLINE entity oluştur."""
    n = len(coords)
    if closed and coords[0] != coords[-1]:
        coords = list(coords) + [coords[0]]
        n = len(coords)

    parts = [
        "0", "LWPOLYLINE",
        "8", layer,
        "90", str(n),
        "70", "1" if closed else "0",
    ]
    for x, y in coords:
        parts.extend(["10", f"{x:.3f}", "20", f"{y:.3f}"])

    return "\n".join(parts)


def export_plt(placements: list, pieces: list, bin_width: float,
               used_length: float, output_path: str, scale: float = 40.0):
    """Yerleştirmeyi PLT (HPGL) formatında kaydet.

    Args:
        placements: decoder çıktısı
        pieces: orijinal parça listesi
        bin_width: kumaş eni (mm)
        used_length: kullanılan uzunluk (mm)
        output_path: çıktı dosya yolu
        scale: mm → HPGL unit çarpanı (1mm = 40 units default)
    """
    commands = ["IN;", "SP1;"]  # Initialize, select pen 1

    def mm_to_hpgl(x, y):
        return int(x * scale), int(y * scale)

    # Kumaş sınırı
    bx0, by0 = mm_to_hpgl(0, 0)
    bx1, by1 = mm_to_hpgl(bin_width, used_length)
    commands.append(f"PU{bx0},{by0};")
    commands.append(f"PD{bx1},{by0},{bx1},{by1},{bx0},{by1},{bx0},{by0};")
    commands.append("PU;")

    # Her parça
    commands.append("SP2;")  # Pen 2 for pieces
    for pl in placements:
        coords = list(pl["polygon"].exterior.coords)
        if not coords:
            continue

        # İlk noktaya git
        hx, hy = mm_to_hpgl(coords[0][0], coords[0][1])
        commands.append(f"PU{hx},{hy};")

        # Çiz
        pd_points = []
        for x, y in coords[1:]:
            hx, hy = mm_to_hpgl(x, y)
            pd_points.append(f"{hx},{hy}")
        commands.append(f"PD{','.join(pd_points)};")
        commands.append("PU;")

    commands.append("SP0;")  # pen up
    commands.append("IN;")   # reset

    with open(output_path, "w", encoding="ascii") as f:
        f.write("\n".join(commands))


def export_svg(placements: list, pieces: list, bin_width: float,
               used_length: float, output_path: str):
    """Yerleştirmeyi SVG olarak kaydet (görselleştirme + kontrol için)."""
    margin = 20
    w = bin_width + 2 * margin
    h = used_length + 2 * margin

    colors = [
        "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
        "#264653", "#A8DADC", "#6D6875", "#B5838D", "#FFB4A2",
        "#95D5B2", "#D4A373", "#CDB4DB",
    ]

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}">']
    svg.append(f'<rect x="{margin}" y="{margin}" width="{bin_width:.0f}" height="{used_length:.0f}" fill="none" stroke="#333" stroke-width="2"/>')

    for pl in placements:
        coords = list(pl["polygon"].exterior.coords)
        points = " ".join(f"{x + margin:.1f},{y + margin:.1f}" for x, y in coords)
        color = colors[pl["piece_id"] % len(colors)]
        svg.append(f'<polygon points="{points}" fill="{color}" fill-opacity="0.6" stroke="{color}" stroke-width="1"/>')

        # Parça ID etiketi
        cx = np.mean([c[0] for c in coords]) + margin
        cy = np.mean([c[1] for c in coords]) + margin
        svg.append(f'<text x="{cx:.0f}" y="{cy:.0f}" text-anchor="middle" font-size="12" fill="#000">#{pl["piece_id"]}</text>')

    svg.append("</svg>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
