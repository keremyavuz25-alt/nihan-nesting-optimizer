"""DXF dosyasından parça geometrilerini çıkarır."""
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid


def parse_dxf_raw(filepath: str) -> list[dict]:
    """DXF'i satır satır parse eder (ezdxf fallback)."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    pieces = []
    i = 0
    in_block = False
    in_poly = False
    vertices = []
    block_idx = -1

    while i < len(lines):
        code = lines[i].strip()
        value = lines[i + 1].strip() if i + 1 < len(lines) else ""

        if code == "0" and value == "BLOCK":
            in_block = True
            block_idx += 1
            vertices = []

        if code == "0" and value in ("POLYLINE", "LWPOLYLINE"):
            in_poly = True
            vertices = []

        if in_poly and code == "10":
            x = float(value)
            # look for Y (code 20)
            j = i + 2
            while j < len(lines) - 1:
                if lines[j].strip() == "20":
                    y = float(lines[j + 1].strip())
                    vertices.append((x, y))
                    break
                j += 2

        if code == "0" and value == "SEQEND":
            if len(vertices) > 2:
                pieces.append({
                    "id": len(pieces),
                    "vertices": vertices.copy(),
                })
            in_poly = False
            vertices = []

        if code == "0" and value == "ENDBLK":
            in_block = False

        i += 1

    return pieces


def pieces_to_polygons(raw_pieces: list[dict]) -> list[dict]:
    """Raw vertex listelerini Shapely Polygon'lara çevirir."""
    polygons = []
    for p in raw_pieces:
        coords = p["vertices"]
        if len(coords) < 3:
            continue
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty or poly.area < 1.0:
            continue

        bounds = poly.bounds  # minx, miny, maxx, maxy
        w = bounds[2] - bounds[0]
        h = bounds[3] - bounds[1]

        polygons.append({
            "id": p["id"],
            "polygon": poly,
            "area": poly.area,
            "width": w,
            "height": h,
            "centroid": (poly.centroid.x, poly.centroid.y),
            "vertices": np.array(coords),
        })

    return polygons


def load_dxf(filepath: str) -> list[dict]:
    """DXF dosyasını yükle ve polygon listesi döndür."""
    raw = parse_dxf_raw(filepath)
    polys = pieces_to_polygons(raw)
    # Orijini sıfırla — tüm parçaları (0,0) köşesine taşı
    for p in polys:
        bounds = p["polygon"].bounds
        from shapely.affinity import translate
        p["polygon"] = translate(p["polygon"], -bounds[0], -bounds[1])
        p["vertices"] = np.array(p["polygon"].exterior.coords)
        b = p["polygon"].bounds
        p["width"] = b[2] - b[0]
        p["height"] = b[3] - b[1]
    return polys


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test.dxf"
    pieces = load_dxf(path)
    total_area = sum(p["area"] for p in pieces)
    print(f"{len(pieces)} parça, toplam alan: {total_area / 100:.0f} cm²")
    for p in pieces:
        print(f"  #{p['id']}: {p['width']:.0f}x{p['height']:.0f}mm, {p['area']/100:.0f}cm², {len(p['vertices'])} vertex")
