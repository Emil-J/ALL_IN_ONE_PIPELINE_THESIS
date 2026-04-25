"""
Stitch and visualize a TMS tile range as one global map mosaic.

What this script does:
- Reads TMS tiles stored as: <root>/<zoom>/<x>/<y>.png
- Reconstructs the full mosaic
- Places highest TMS y at the top row
- Draws a border around every tile
- Writes the tile ID in the top-right corner of each tile
- Saves the final stitched image
- Optionally displays the result

Tested assumptions for your case:
- Tiles are PNG
- Folder structure is 16/x/y.png
- Tile size is 512x512
- TMS convention is used
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================

ROOT_DIR = Path(r"C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\REFERENCE_MAP_CPH\aerial")
OUTPUT_DIR = Path(r"C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\TMS_Map_Reconstruction_Check")

ZOOM = 16

TILE_X_MIN = 34994
TILE_X_MAX = 35090

TILE_Y_MIN = 44976
TILE_Y_MAX = 45063

TILE_SIZE = 512
TILE_EXT = ".png"

# Output file
OUTPUT_PATH = OUTPUT_DIR / f"stitched_tms_z{ZOOM}_{TILE_X_MIN}_{TILE_X_MAX}_{TILE_Y_MIN}_{TILE_Y_MAX}.png"

# Drawing options
BORDER_COLOR = "black"
BORDER_WIDTH = 2

LABEL_TEXT_COLOR = "black"
LABEL_BG_COLOR = (255, 255, 255, 200)   # semi-transparent white
LABEL_PADDING_X = 8
LABEL_PADDING_Y = 6
LABEL_MARGIN = 8

# Set to True if you want missing tiles replaced by a placeholder tile
FILL_MISSING_TILES = True

# Placeholder tile appearance
MISSING_TILE_BG = (235, 235, 235)
MISSING_TILE_TEXT = "MISSING"
MISSING_TILE_TEXT_COLOR = "red"

# Display final image
SHOW_RESULT = False

# ============================================================
# HELPERS
# ============================================================

def load_font(preferred_size=20):
    """
    Try to load a decent TrueType font.
    Fall back to PIL default if unavailable.
    """
    candidate_fonts = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "Tahoma.ttf",
    ]

    for font_name in candidate_fonts:
        try:
            return ImageFont.truetype(font_name, preferred_size)
        except Exception:
            continue

    return ImageFont.load_default()


def create_missing_tile(tile_size, tile_id_text, font):
    """
    Create a visible placeholder tile for missing data.
    """
    tile = Image.new("RGBA", (tile_size, tile_size), MISSING_TILE_BG)
    draw = ImageDraw.Draw(tile)

    # Red cross
    draw.line((0, 0, tile_size, tile_size), fill="red", width=3)
    draw.line((0, tile_size, tile_size, 0), fill="red", width=3)

    # Border
    draw.rectangle(
        [0, 0, tile_size - 1, tile_size - 1],
        outline=BORDER_COLOR,
        width=BORDER_WIDTH
    )

    # Center "MISSING"
    missing_bbox = draw.textbbox((0, 0), MISSING_TILE_TEXT, font=font)
    missing_w = missing_bbox[2] - missing_bbox[0]
    missing_h = missing_bbox[3] - missing_bbox[1]
    missing_x = (tile_size - missing_w) // 2
    missing_y = (tile_size - missing_h) // 2
    draw.text((missing_x, missing_y), MISSING_TILE_TEXT, fill=MISSING_TILE_TEXT_COLOR, font=font)

    # Add tile ID in top-right
    draw_tile_label(draw, tile_size, tile_id_text, font)

    return tile


def draw_tile_label(draw, tile_size, tile_id_text, font):
    """
    Draw tile ID in the top-right corner with a background box.
    """
    bbox = draw.textbbox((0, 0), tile_id_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x_text = tile_size - LABEL_MARGIN - text_w - LABEL_PADDING_X
    y_text = LABEL_MARGIN + LABEL_PADDING_Y

    x0 = x_text - LABEL_PADDING_X
    y0 = y_text - LABEL_PADDING_Y
    x1 = x_text + text_w + LABEL_PADDING_X
    y1 = y_text + text_h + LABEL_PADDING_Y

    draw.rounded_rectangle(
        [x0, y0, x1, y1],
        radius=6,
        fill=LABEL_BG_COLOR,
        outline="black",
        width=1
    )
    draw.text((x_text, y_text), tile_id_text, fill=LABEL_TEXT_COLOR, font=font)


def add_border_and_label(tile_img, tile_id_text, font):
    """
    Draw border and top-right label on a tile image.
    """
    tile = tile_img.convert("RGBA")
    draw = ImageDraw.Draw(tile)

    draw.rectangle(
        [0, 0, tile.width - 1, tile.height - 1],
        outline=BORDER_COLOR,
        width=BORDER_WIDTH
    )

    draw_tile_label(draw, tile.width, tile_id_text, font)

    return tile


# ============================================================
# MAIN
# ============================================================

def main():
    font = load_font(preferred_size=20)

    x_values = list(range(TILE_X_MIN, TILE_X_MAX + 1))
    y_values = list(range(TILE_Y_MIN, TILE_Y_MAX + 1))

    # IMPORTANT:
    # TMS y increases upward, but image row index increases downward.
    # Therefore, highest TMS y goes at the top.
    y_values_top_to_bottom = sorted(y_values, reverse=True)

    num_cols = len(x_values)
    num_rows = len(y_values_top_to_bottom)

    mosaic_width = num_cols * TILE_SIZE
    mosaic_height = num_rows * TILE_SIZE

    mosaic = Image.new("RGBA", (mosaic_width, mosaic_height), (255, 255, 255, 255))

    missing_count = 0
    loaded_count = 0

    for row_idx, y in enumerate(y_values_top_to_bottom):
        for col_idx, x in enumerate(x_values):
            tile_path = ROOT_DIR / str(ZOOM) / str(x) / f"{y}{TILE_EXT}"
            tile_id_text = f"{ZOOM}/{x}/{y}"

            if tile_path.exists():
                try:
                    tile = Image.open(tile_path).convert("RGBA")

                    # Safety check
                    if tile.size != (TILE_SIZE, TILE_SIZE):
                        tile = tile.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS)

                    tile = add_border_and_label(tile, tile_id_text, font)
                    loaded_count += 1

                except Exception as e:
                    print(f"[ERROR] Failed to read tile: {tile_path}")
                    print(f"        Reason: {e}")
                    if FILL_MISSING_TILES:
                        tile = create_missing_tile(TILE_SIZE, tile_id_text, font)
                        missing_count += 1
                    else:
                        continue
            else:
                print(f"[WARNING] Missing tile: {tile_path}")
                if FILL_MISSING_TILES:
                    tile = create_missing_tile(TILE_SIZE, tile_id_text, font)
                    missing_count += 1
                else:
                    continue

            paste_x = col_idx * TILE_SIZE
            paste_y = row_idx * TILE_SIZE
            mosaic.paste(tile, (paste_x, paste_y), tile)

    mosaic.save(OUTPUT_PATH)
    print("=" * 70)
    print("Done.")
    print(f"Saved stitched mosaic to:\n{OUTPUT_PATH}")
    print(f"Loaded tiles : {loaded_count}")
    print(f"Missing tiles: {missing_count}")
    print(f"Mosaic size  : {mosaic_width} x {mosaic_height} pixels")
    print(f"Grid size    : {num_cols} columns x {num_rows} rows")
    print("=" * 70)

    if SHOW_RESULT:
        plt.figure(figsize=(16, 16))
        plt.imshow(mosaic)
        plt.axis("off")
        plt.title(
            f"TMS Mosaic - Zoom {ZOOM}\n"
            f"X: {TILE_X_MIN} to {TILE_X_MAX}, Y: {TILE_Y_MIN} to {TILE_Y_MAX}"
        )
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()