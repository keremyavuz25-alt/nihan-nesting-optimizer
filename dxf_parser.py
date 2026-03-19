"""DXF dosyasından parça geometrilerini çıkarır — MATERIAL filtreli."""
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid


def parse_dxf_raw(filepath: str, material_filter: str = "KUMAS") -> list[dict]:
    """DXF'i parse et. Sadece belirtilen MATERIAL'e ait parçaları döndür.

    Args:
        filepath: DXF dosya yolu
        material_filter: Sadece bu malzeme türündeki parçaları al.
                        "KUMAS" = ana kumaş (pastal parçaları)
                        "ASTAR" = astar
                        None = filtre yok, tüm parçalar
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    # Faz 1: BLOCK'ları parse et — her block'un POLYLINE'ları, TEXT'leri, LINE'ları
    blocks = []
    i = 0

    while i < len(lines):
        code = lines[i].strip()
        value = lines[i + 1].strip() if i + 1 < len(lines) else ""

        if code == "0" and value == "BLOCK":
            block = {
                "cutlines": [],      # Layer 1 POLYLINE'lar
                "grain_lines": [],   # Layer 7 LINE'lar
                "internal": [],      # Layer 8 POLYLINE'lar
                "texts": [],         # Tüm TEXT'ler
                "material": None,
            }

            # Block içini parse et
            i += 1
            in_poly = False
            poly_layer = ""
            vertices = []

            while i < len(lines):
                code = lines[i].strip()
                value = lines[i + 1].strip() if i + 1 < len(lines) else ""

                # POLYLINE başlangıcı
                if code == "0" and value in ("POLYLINE", "LWPOLYLINE"):
                    in_poly = True
                    poly_layer = ""
                    vertices = []

                # Layer kodu
                if in_poly and code == "8" and not poly_layer:
                    poly_layer = value

                # Vertex
                if in_poly and code == "10":
                    x = float(value)
                    j = i + 2
                    while j < len(lines) - 1:
                        if lines[j].strip() == "20":
                            y = float(lines[j + 1].strip())
                            vertices.append((x, y))
                            break
                        j += 2

                # POLYLINE sonu
                if code == "0" and value == "SEQEND":
                    if len(vertices) > 2:
                        if poly_layer == "1":
                            block["cutlines"].append(vertices.copy())
                        elif poly_layer == "8":
                            block["internal"].append(vertices.copy())
                    in_poly = False
                    vertices = []
                    poly_layer = ""

                # TEXT — MATERIAL bilgisi
                if code == "1" and not in_poly:
                    block["texts"].append(value)
                    if value.upper().startswith("MATERIAL"):
                        mat = value.split(":")[-1].strip().upper()
                        # Encoding fix: KUMAŞ → KUMAS, ÇÖZÜM → COZUM
                        mat = mat.replace("\ufffd", "").strip()
                        if "KUMA" in mat:
                            mat = "KUMAS"
                        elif "ASTAR" in mat:
                            mat = "ASTAR"
                        elif "COZUM" in mat or "ZUM" in mat or "Z" in mat and len(mat) < 8:
                            mat = "COZUM"
                        elif "KOMB" in mat:
                            mat = "KOMBIN"
                        elif "KURK" in mat or "RK" in mat and len(mat) < 6:
                            mat = "KURK"
                        elif "ELYAF" in mat:
                            mat = "ELYAF"
                        block["material"] = mat

                # LINE (grain line, Layer 7)
                # Basit kayıt — koordinatları şimdilik almıyoruz

                # Block sonu
                if code == "0" and value == "ENDBLK":
                    break

                i += 1

            if block["cutlines"]:
                blocks.append(block)

        i += 1

    # Faz 2: Material filtresi uygula
    pieces = []
    for block in blocks:
        if material_filter is not None:
            if block["material"] != material_filter:
                continue

        for cutline in block["cutlines"]:
            pieces.append({
                "id": len(pieces),
                "vertices": cutline,
                "material": block["material"],
            })

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

        bounds = poly.bounds
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
            "material": p.get("material"),
        })

    return polygons


def load_dxf(filepath: str, material: str = "KUMAS") -> list[dict]:
    """DXF dosyasını yükle — sadece belirtilen malzeme parçalarını döndür.

    Args:
        filepath: DXF dosya yolu
        material: "KUMAS" (default), "ASTAR", None (hepsi)
    """
    raw = parse_dxf_raw(filepath, material_filter=material)
    polys = pieces_to_polygons(raw)
    # Orijini sıfırla
    from shapely.affinity import translate
    for p in polys:
        bounds = p["polygon"].bounds
        p["polygon"] = translate(p["polygon"], -bounds[0], -bounds[1])
        p["vertices"] = np.array(p["polygon"].exterior.coords)
        b = p["polygon"].bounds
        p["width"] = b[2] - b[0]
        p["height"] = b[3] - b[1]
    return polys


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    path = sys.argv[1] if len(sys.argv) > 1 else "test.dxf"
    mat = sys.argv[2] if len(sys.argv) > 2 else "KUMAS"

    # Önce tüm malzemeleri göster
    all_pieces = parse_dxf_raw(path, material_filter=None)
    materials = {}
    for p in all_pieces:
        m = p.get("material", "UNKNOWN")
        materials[m] = materials.get(m, 0) + 1
    print(f"Tüm parçalar: {len(all_pieces)}")
    for m, c in sorted(materials.items()):
        marker = " <-- PASTAL" if m == "KUMAS" else ""
        print(f"  {m}: {c}{marker}")

    # Filtrelenmiş
    pieces = load_dxf(path, material=mat)
    total_area = sum(p["area"] for p in pieces)
    print(f"\nFiltre: MATERIAL={mat}")
    print(f"{len(pieces)} parça, toplam alan: {total_area / 100:.0f} cm2")
    for p in pieces:
        print(f"  #{p['id']}: {p['width']:.0f}x{p['height']:.0f}mm, {p['area']/100:.0f}cm2")
