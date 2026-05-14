"""Compressed terrain vocabulary for Minecraft block names."""

from __future__ import annotations

import json
from pathlib import Path

NUM_CLASSES = 17
UNKNOWN_INDEX = 16
UNKNOWN_NAME = "unknown"
AIR_INDEX = 0

CLASS_NAMES = [
    "air",
    "grass_block",
    "dirt",
    "coarse_dirt",
    "stone",
    "gravel",
    "sand",
    "red_sand",
    "sandstone",
    "snow_block",
    "ice",
    "water",
    "clay",
    "podzol",
    "ore",
    "mycelium",
    UNKNOWN_NAME,
]

# Only preserve "surface" blocks
VOCAB: dict[str, int] = {
    "minecraft:air": AIR_INDEX,
    "minecraft:cave_air": AIR_INDEX,
    "minecraft:void_air": AIR_INDEX,
    "minecraft:grass_block": 1,
    "minecraft:dirt": 2,
    "minecraft:rooted_dirt": 2,
    "minecraft:coarse_dirt": 3,
    "minecraft:stone": 4,
    "minecraft:granite": 4,
    "minecraft:diorite": 4,
    "minecraft:andesite": 4,
    "minecraft:deepslate": 4,
    "minecraft:tuff": 4,
    "minecraft:gravel": 5,
    "minecraft:sand": 6,
    "minecraft:red_sand": 7,
    "minecraft:sandstone": 8,
    "minecraft:red_sandstone": 8,
    "minecraft:snow_block": 9,
    "minecraft:powder_snow": 9,
    "minecraft:ice": 10,
    "minecraft:packed_ice": 10,
    "minecraft:blue_ice": 10,
    "minecraft:water": 11,
    "minecraft:clay": 12,
    "minecraft:podzol": 13,
    "minecraft:coal_ore": 14,
    "minecraft:iron_ore": 14,
    "minecraft:gold_ore": 14,
    "minecraft:copper_ore": 14,
    "minecraft:deepslate_coal_ore": 14,
    "minecraft:deepslate_iron_ore": 14,
    "minecraft:mycelium": 15,
}


def encode(block_name: str) -> int:
    """Encode a raw Minecraft block name into a compact terrain index."""
    return VOCAB.get(block_name, AIR_INDEX)


def canonical_vocab_payload() -> dict[str, object]:
    """Return the JSON payload committed to configs/vocab.json."""
    return {
        "aliases": dict(sorted(VOCAB.items())),
        "classes": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "unknown_index": UNKNOWN_INDEX,
    }


def vocab_config_path() -> Path:
    """Return the committed vocabulary JSON path."""
    return Path(__file__).resolve().parents[4] / "configs" / "vocab.json"


def write_vocab_json() -> Path:
    """Write the canonical vocabulary JSON deterministically."""
    out_path = vocab_config_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(canonical_vocab_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


if __name__ == "__main__":
    write_vocab_json()
