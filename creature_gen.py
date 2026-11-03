#!/usr/bin/env python3
"""
creature_gen.py - Autonomous Voxel Creature Factory for Three.js Voxel Game.

Features:
- Connects to local Ollama instance (default: http://localhost:11434)
- Prompts LLM for unique voxel creatures (wildlife, monsters, bosses)
- Enforces strict JSON output with no markdown code blocks
- Suffix Protection: Appends random 4-character hex suffix (_a7f2) to id and filename
- Centering & Grounding Safety Guard: Forces all coordinates centered at x=0, z=0, base at y=0
- Performance Lag Safety Guard: Limits voxel density and optimizes geometry schema
- Manifest Builder: Maintains assets/creatures/manifest.json for fast client-side loading
- Robust Error Handling: Safe procedural fallback if Ollama is offline or returns invalid JSON
"""

import os
import sys
import json
import time
import random
import secrets
import argparse
from datetime import datetime, timezone
import requests

# --- CONFIGURATION ---
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
CREATURES_DIR = os.path.join("assets", "creatures")
MANIFEST_PATH = os.path.join(CREATURES_DIR, "manifest.json")
MAX_VOXELS = 180  # Performance safeguard against high-poly lag

# Ensure output directory exists
os.makedirs(CREATURES_DIR, exist_ok=True)


CREATURE_PROMPT_TEMPLATE = """You are an expert voxel 3D artist creating retro low-poly voxel characters.
Generate a creative, detailed voxel creature in STRICT JSON format with NO markdown wrapping (no ```json code blocks, just raw JSON).

Creature category to invent: {category_theme}

JSON Output Schema:
{{
  "id": "lowercase_creature_name_without_suffix",
  "name": "Display Name",
  "category": "wildlife" | "monster" | "boss",
  "description": "Short 1-sentence description of behavior or lore",
  "movementStyle": "walking" | "hover" | "stationary",
  "scale": 0.8,
  "voxels": [
    {{"x": 0, "y": 0, "z": 0, "color": "#HEXVAL", "part": "leg_fl"}},
    ...
  ]
}}

CRITICAL DESIGN RULES:
1. CENTERING RULE: All voxel coordinates MUST be horizontally centered around x=0 and z=0.
2. GROUNDING RULE: The lowest voxels (feet, belly, or base) MUST rest strictly at y=0. Do NOT use negative y values.
3. Keep the total number of voxels between 35 and 90 for optimal real-time browser performance.
4. Colors MUST be valid 6-character hex strings (e.g. #3b82f6, #10b981, #f59e0b).
5. Output ONLY the raw JSON object. Do not include introductory text, explanations, or markdown code fences."""


def generate_suffix() -> str:
    """Generates a secure 4-character hex suffix to prevent file overwrites."""
    return secrets.token_hex(2)


def enforce_centering_and_grounding(voxels: list) -> list:
    """
    Mathematical Safety Guard:
    Enforces strict centering at x=0, z=0 and sets lowest point to y=0.
    Eliminates any off-center or floating render bugs caused by LLM coordinate drifting.
    """
    if not voxels:
        return []

    # Filter out malformed entries
    valid_voxels = []
    for v in voxels:
        if isinstance(v, dict) and "x" in v and "y" in v and "z" in v:
            color = str(v.get("color", "#888888")).strip()
            if not color.startswith("#"):
                color = f"#{color}"
            if len(color) != 7:
                color = "#64748b"

            valid_voxels.append({
                "x": int(round(float(v["x"]))),
                "y": int(round(float(v["y"]))),
                "z": int(round(float(v["z"]))),
                "color": color.lower(),
                "part": str(v.get("part", "body"))
            })

    if not valid_voxels:
        return []

    # Calculate bounding box
    min_x = min(v["x"] for v in valid_voxels)
    max_x = max(v["x"] for v in valid_voxels)
    min_y = min(v["y"] for v in valid_voxels)
    min_z = min(v["z"] for v in valid_voxels)
    max_z = max(v["z"] for v in valid_voxels)

    offset_x = int(round((min_x + max_x) / 2.0))
    offset_y = min_y  # Shift lowest block to y = 0
    offset_z = int(round((min_z + max_z) / 2.0))

    # Apply offsets and deduplicate voxels at same coordinate
    seen_coords = set()
    sanitized = []
    for v in valid_voxels:
        cx = v["x"] - offset_x
        cy = v["y"] - offset_y
        cz = v["z"] - offset_z
        coord_key = (cx, cy, cz)
        if coord_key not in seen_coords:
            seen_coords.add(coord_key)
            sanitized.append({
                "x": cx,
                "y": cy,
                "z": cz,
                "color": v["color"],
                "part": v["part"]
            })

    # Performance safeguard: clamp to max allowed voxels
    if len(sanitized) > MAX_VOXELS:
        sanitized = sanitized[:MAX_VOXELS]

    return sanitized


def calculate_dimensions(voxels: list) -> dict:
    """Calculates width, height, and depth for bounding volume."""
    if not voxels:
        return {"width": 1, "height": 1, "depth": 1}
    xs = [v["x"] for v in voxels]
    ys = [v["y"] for v in voxels]
    zs = [v["z"] for v in voxels]
    return {
        "width": max(xs) - min(xs) + 1,
        "height": max(ys) - min(ys) + 1,
        "depth": max(zs) - min(zs) + 1
    }


def update_manifest(creature_meta: dict, file_rel_path: str) -> dict:
    """
    Manifest Builder:
    Maintains assets/creatures/manifest.json listing all creature files.
    Enables fast, zero-latency client-side creature loading.
    """
    manifest_data = {"version": "1.0", "updated_at": "", "count": 0, "creatures": []}

    # Load existing manifest if present
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
                if not isinstance(manifest_data.get("creatures"), list):
                    manifest_data["creatures"] = []
        except Exception as e:
            print(f"[Manifest] Warning: parsing existing manifest failed: {e}. Rebuilding.")
            manifest_data["creatures"] = []

    # Remove existing entry with same id if any, to allow clean updates
    creatures_list = [c for c in manifest_data["creatures"] if c.get("id") != creature_meta["id"]]

    entry = {
        "id": creature_meta["id"],
        "name": creature_meta["name"],
        "category": creature_meta.get("category", "wildlife"),
        "movementStyle": creature_meta.get("movementStyle", "walking"),
        "scale": creature_meta.get("scale", 1.0),
        "voxel_count": len(creature_meta.get("voxels", [])),
        "dimensions": creature_meta.get("dimensions", {"width": 1, "height": 1, "depth": 1}),
        "file": file_rel_path.replace("\\", "/"),
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    creatures_list.append(entry)
    manifest_data["creatures"] = creatures_list
    manifest_data["count"] = len(creatures_list)
    manifest_data["updated_at"] = datetime.now(timezone.utc).isoformat()

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_data


def generate_fallback_creature(category: str = None) -> dict:
    """
    Procedural Fallback Generator:
    Guarantees the script never crashes even if Ollama is unreachable.
    Creates well-proportioned, beautifully textured retro voxel models.
    """
    themes = ["wildlife_dog", "wildlife_pig", "monster_golem", "boss_drake"]
    selected_theme = None
    if category:
        cat_lower = str(category).lower()
        if any(w in cat_lower for w in ["boss", "drake", "dragon"]):
            selected_theme = "boss_drake"
        elif any(w in cat_lower for w in ["monster", "golem", "skeleton"]):
            selected_theme = "monster_golem"
        elif any(w in cat_lower for w in ["pig", "boar", "swine"]):
            selected_theme = "wildlife_pig"
        elif any(w in cat_lower for w in ["wildlife", "dog", "pup", "wolf", "fox"]):
            selected_theme = "wildlife_dog"
    if not selected_theme:
        selected_theme = random.choice(themes)

    suffix = generate_suffix()
    voxels = []

    if selected_theme == "wildlife_dog":
        creature_id = f"voxel_pup_{suffix}"
        name = "Scruffy Voxel Pup"
        cat = "wildlife"
        desc = "A loyal blocky canine always eager to accompany Little Doug on explorations."
        move_style = "walking"
        scale = 0.8
        fur = "#a16207"
        snout = "#fef08a"
        nose = "#0f172a"
        collar = "#ef4444"

        # 4 Legs (y=0 to y=1)
        for lx in [-1, 1]:
            for lz in [-1, 2]:
                voxels.append({"x": lx, "y": 0, "z": lz, "color": fur, "part": "leg"})
                voxels.append({"x": lx, "y": 1, "z": lz, "color": fur, "part": "leg"})
        # Body (y=2 to y=3, z=-1 to 2)
        for by in [2, 3]:
            for bz in range(-1, 3):
                voxels.append({"x": -1, "y": by, "z": bz, "color": fur, "part": "body"})
                voxels.append({"x": 0, "y": by, "z": bz, "color": snout if by == 2 else fur, "part": "body"})
                voxels.append({"x": 1, "y": by, "z": bz, "color": fur, "part": "body"})
        # Collar
        voxels.append({"x": 0, "y": 3, "z": -1, "color": collar, "part": "collar"})
        # Head (y=3 to y=4, z=-2 to -1)
        for hy in [3, 4]:
            for hx in [-1, 0, 1]:
                voxels.append({"x": hx, "y": hy, "z": -2, "color": fur, "part": "head"})
        # Snout & Nose
        voxels.append({"x": 0, "y": 3, "z": -3, "color": snout, "part": "snout"})
        voxels.append({"x": 0, "y": 4, "z": -3, "color": nose, "part": "nose"})
        # Ears (y=5)
        voxels.append({"x": -1, "y": 5, "z": -2, "color": fur, "part": "ear"})
        voxels.append({"x": 1, "y": 5, "z": -2, "color": fur, "part": "ear"})
        # Tail (y=3 to 4, z=3)
        voxels.append({"x": 0, "y": 3, "z": 3, "color": fur, "part": "tail"})
        voxels.append({"x": 0, "y": 4, "z": 3, "color": snout, "part": "tail"})

    elif selected_theme == "boss_drake":
        creature_id = f"sky_drake_{suffix}"
        name = "Verdant Sky Drake"
        cat = "boss"
        desc = "A rare miniature celestial dragon hovering over ancient ridges."
        move_style = "hover"
        scale = 1.2
        scale_color = "#15803d"
        belly = "#86efac"
        eye = "#facc15"
        wing_color = "#22c55e"

        # Feet / Base (hovering posture, base touches y=0)
        for fx in [-1, 1]:
            voxels.append({"x": fx, "y": 0, "z": 0, "color": scale_color, "part": "claw"})
            voxels.append({"x": fx, "y": 1, "z": 0, "color": scale_color, "part": "claw"})
        # Torso
        for ty in [2, 3, 4]:
            for tz in [-1, 0, 1]:
                voxels.append({"x": 0, "y": ty, "z": tz, "color": scale_color, "part": "body"})
                voxels.append({"x": 0, "y": ty, "z": tz - 1, "color": belly, "part": "belly"})
        # Wings spread horizontally
        for wx in [-3, -2, 2, 3]:
            for wz in [0, 1]:
                voxels.append({"x": wx, "y": 3, "z": wz, "color": wing_color, "part": "wing"})
                voxels.append({"x": wx, "y": 4, "z": wz, "color": wing_color, "part": "wing"})
        # Head & Horns
        voxels.append({"x": 0, "y": 5, "z": -2, "color": scale_color, "part": "head"})
        voxels.append({"x": -1, "y": 5, "z": -2, "color": eye, "part": "eye"})
        voxels.append({"x": 1, "y": 5, "z": -2, "color": eye, "part": "eye"})
        voxels.append({"x": -1, "y": 6, "z": -1, "color": "#fef08a", "part": "horn"})
        voxels.append({"x": 1, "y": 6, "z": -1, "color": "#fef08a", "part": "horn"})

    elif selected_theme == "wildlife_pig":
        creature_id = f"mud_pig_{suffix}"
        name = "Pink Truffle Pig"
        cat = "wildlife"
        desc = "A cheerful pink pig searching the damp forest soil for delicious truffles."
        move_style = "walking"
        scale = 0.85
        pink = "#f472b6"
        snout = "#fbcfe8"
        hoof = "#475569"
        eye = "#0f172a"

        # 4 Stumpy Legs (y=0 to y=1)
        for lx in [-1, 1]:
            for lz in [-1, 1]:
                voxels.append({"x": lx, "y": 0, "z": lz, "color": hoof, "part": "hoof"})
                voxels.append({"x": lx, "y": 1, "z": lz, "color": pink, "part": "leg"})
        # Chubby Body (y=2 to y=3, z=-1 to 1)
        for by in [2, 3]:
            for bz in [-1, 0, 1]:
                for bx in [-1, 0, 1]:
                    voxels.append({"x": bx, "y": by, "z": bz, "color": pink, "part": "body"})
        # Head (y=3, z=-2)
        for hx in [-1, 0, 1]:
            voxels.append({"x": hx, "y": 3, "z": -2, "color": pink, "part": "head"})
        voxels.append({"x": -1, "y": 4, "z": -2, "color": eye, "part": "eye"})
        voxels.append({"x": 1, "y": 4, "z": -2, "color": eye, "part": "eye"})
        # Cute Snout (y=2, z=-3)
        voxels.append({"x": 0, "y": 2, "z": -3, "color": snout, "part": "snout"})
        # Curly Tail (y=3, z=2)
        voxels.append({"x": 0, "y": 3, "z": 2, "color": pink, "part": "tail"})

    else:  # monster_golem
        creature_id = f"stone_golem_{suffix}"
        name = "Ancient Moss Golem"
        cat = "monster"
        desc = "A lumbering sentinel carved of ancient granite and awakened by mystical ores."
        move_style = "walking"
        scale = 1.0
        stone = "#64748b"
        moss = "#16a34a"
        core = "#38bdf8"

        # Legs (y=0 to y=1)
        for lx in [-1, 1]:
            voxels.append({"x": lx, "y": 0, "z": 0, "color": stone, "part": "leg"})
            voxels.append({"x": lx, "y": 1, "z": 0, "color": stone, "part": "leg"})
        # Massive Torso (y=2 to 4)
        for ty in [2, 3, 4]:
            for tx in [-1, 0, 1]:
                voxels.append({"x": tx, "y": ty, "z": 0, "color": stone if tx != 0 else moss, "part": "torso"})
        # Glowing Rune Core
        voxels.append({"x": 0, "y": 3, "z": 1, "color": core, "part": "core"})
        # Arms
        voxels.append({"x": -2, "y": 3, "z": 0, "color": stone, "part": "arm"})
        voxels.append({"x": -2, "y": 2, "z": 0, "color": stone, "part": "arm"})
        voxels.append({"x": 2, "y": 3, "z": 0, "color": stone, "part": "arm"})
        voxels.append({"x": 2, "y": 2, "z": 0, "color": stone, "part": "arm"})
        # Head (y=5)
        voxels.append({"x": 0, "y": 5, "z": 0, "color": stone, "part": "head"})
        voxels.append({"x": 0, "y": 5, "z": 1, "color": core, "part": "eye"})

    sanitized_voxels = enforce_centering_and_grounding(voxels)

    return {
        "id": creature_id,
        "name": name,
        "category": cat,
        "description": desc,
        "movementStyle": move_style,
        "scale": scale,
        "dimensions": calculate_dimensions(sanitized_voxels),
        "voxels": sanitized_voxels
    }


def query_ollama(prompt: str) -> dict:
    """
    Connects to Ollama instance at OLLAMA_HOST.
    Enforces format: 'json' with strict JSON extraction.
    """
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.75,
            "top_p": 0.9
        }
    }

    print(f"[CreatureGen] Contacting Ollama ({OLLAMA_HOST}) with model '{OLLAMA_MODEL}'...")
    response = requests.post(url, json=payload, timeout=20)
    response.raise_for_status()

    raw_response = response.json().get("response", "")
    content = raw_response.strip()

    # Safety: Strip any markdown code backticks if the model ignored system instructions
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    parsed = json.loads(content)
    if not isinstance(parsed, dict) or "voxels" not in parsed:
        raise ValueError("Model output did not conform to creature voxel schema")

    return parsed


def generate_creature(theme: str = None) -> tuple:
    """
    Main Autonomous Creature Generation Pipeline.
    Returns: (creature_data: dict, file_path: str)
    """
    categories = [
        "Small friendly wildlife (e.g., foxes, badgers, mountain goats, wild boars, river otters)",
        "Spooky outdoor monsters (e.g., moss skeletons, rock stalkers, swamp wraiths, mushroom creepers)",
        "Epic world boss or rare guardian (e.g., elder storm drake, crystal colossus, ancient treant)"
    ]
    chosen_theme = theme or random.choice(categories)
    creature_data = None

    try:
        prompt = CREATURE_PROMPT_TEMPLATE.format(category_theme=chosen_theme)
        raw_creature = query_ollama(prompt)

        # Generate random 4-character hex suffix to prevent file collisions
        suffix = generate_suffix()
        base_id = str(raw_creature.get("id", "creature")).lower().replace(" ", "_")
        # Strip existing suffix if model hallucinated one
        clean_id = base_id.split("_")[0] if len(base_id) > 12 else base_id
        final_id = f"{clean_id}_{suffix}"

        # Apply strict centering and grounding guard
        sanitized_voxels = enforce_centering_and_grounding(raw_creature.get("voxels", []))
        if len(sanitized_voxels) < 8:
            raise ValueError("Generated creature has insufficient voxels, using procedural generator")

        creature_data = {
            "id": final_id,
            "name": str(raw_creature.get("name", "Voxel Wonder")).strip(),
            "category": str(raw_creature.get("category", "wildlife")).lower(),
            "description": str(raw_creature.get("description", "A mysterious inhabitant of the voxel wilderness.")).strip(),
            "movementStyle": str(raw_creature.get("movementStyle", "walking")).lower(),
            "scale": float(raw_creature.get("scale", 1.0)),
            "dimensions": calculate_dimensions(sanitized_voxels),
            "voxels": sanitized_voxels
        }
        print(f"[CreatureGen] Successfully generated LLM creature: '{creature_data['name']}' ({final_id})")

    except Exception as e:
        print(f"[CreatureGen] Notice: Ollama unreachable or invalid response ({e}). Activating safe fallback...")
        creature_data = generate_fallback_creature(chosen_theme)
        print(f"[CreatureGen] Created procedural fallback creature: '{creature_data['name']}' ({creature_data['id']})")

    # Save to assets/creatures/[id].json with suffix protection
    filename = f"{creature_data['id']}.json"
    file_path = os.path.join(CREATURES_DIR, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(creature_data, f, indent=2)

    # Update master manifest.json
    rel_path = os.path.join("assets", "creatures", filename).replace("\\", "/")
    update_manifest(creature_data, rel_path)

    print(f"[CreatureGen] Saved: {file_path}")
    print(f"[CreatureGen] Manifest updated at: {MANIFEST_PATH}")
    return creature_data, file_path


def main():
    parser = argparse.ArgumentParser(description="Autonomous Voxel Creature Factory")
    parser.add_argument("--count", type=int, default=1, help="Number of creatures to generate")
    parser.add_argument("--theme", type=str, default=None, help="Theme or category prompt")
    args = parser.parse_args()

    print("==================================================")
    print(" Voxel Creature Factory - Autonomous Pipeline     ")
    print("==================================================")
    for i in range(args.count):
        print(f"\n--- Generating Creature {i+1}/{args.count} ---")
        generate_creature(args.theme)


if __name__ == "__main__":
    main()
