#!/usr/bin/env python3
"""
creature_gen.py - Autonomous Voxel Creature Factory

Features:
- Connects directly to local AI workstation (http://192.168.12.253:11434/v1/chat/completions) using qwen3:14b
- Structured JSON schema enforcement with strict response_format
- Suffix Protection: Automatically appends 4-character hex suffix (_a7f2) to prevent file overwriting
- Centering & Grounding Safety Guard: Forces all coordinates centered at x=0, z=0, base at y=0
- Performance Lag Safeguard: Clamps voxel counts and enforces low-draw-call geometry
- Manifest Builder: Automatically creates and updates assets/creatures/manifest.json
- Autonomous Interval Loop: Continuously cycles at configurable intervals (matching agent.py)
- Robust Error Handling: Safe procedural fallback if workstation is offline or unreachable
- Git Synchronization: Commits and pushes new creatures to GitHub automatically
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

# --- CONFIGURATION MATCHING AGENT.PY ---
WORKSTATION_URL = os.environ.get("WORKSTATION_URL", "http://192.168.12.253:11434/v1/chat/completions")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3:14b")
CREATURES_DIR = os.path.join("assets", "creatures")
MANIFEST_PATH = os.path.join(CREATURES_DIR, "manifest.json")
INTERVAL_SECONDS = 3600               # 1 hour between creature creation cycles (matching agent.py)
MAX_VOXELS = 180                      # Performance safeguard against high-poly lag

# Ensure output directory exists
os.makedirs(CREATURES_DIR, exist_ok=True)

SYSTEM_PROMPT = """You are a procedural 3D voxel character artist and DUNGEON MASTER for a retro voxel adventure game featuring 'Little Doug'.
You design unique voxel creatures (wildlife, monsters, or bosses) in STRICT JSON format matching the schema.

Rules:
1. CENTERING RULE: All voxel coordinates MUST be horizontally centered around x=0 and z=0.
2. GROUNDING RULE: The lowest voxels (feet, belly, or base) MUST rest strictly at y=0 (minimum y must be 0, never negative).
3. PERFORMANCE: Keep total number of voxels between 35 and 90 for 60 FPS real-time browser performance.
4. COLORS: All voxel colors MUST be valid 6-character hex color codes (e.g., #3b82f6, #10b981, #f59e0b).
5. Output strictly raw JSON conforming to the schema."""

CREATURE_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {
            "type": "string",
            "description": "Short lowercase creature name identifier (without suffix), e.g., 'meadow_fox' or 'stone_golem'"
        },
        "name": {
            "type": "string",
            "description": "Readable display name of the creature"
        },
        "category": {
            "type": "string",
            "description": "Creature classification: wildlife, monster, or boss"
        },
        "description": {
            "type": "string",
            "description": "Short 1-sentence Dungeon Master lore description of behavior or habitat"
        },
        "movementStyle": {
            "type": "string",
            "description": "Animation movement style: 'walking' for ground wildlife/monsters, 'hover' for flying/aerial mobs, or 'stationary'"
        },
        "scale": {
            "type": "number",
            "description": "Rendering scale factor (typically 0.7 to 1.3)"
        },
        "voxels": {
            "type": "array",
            "description": "List of voxel blocks defining the creature model (35 to 90 voxels)",
            "items": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "Horizontal X coordinate centered at 0"},
                    "y": {"type": "integer", "description": "Vertical Y coordinate starting at 0 (feet/base)"},
                    "z": {"type": "integer", "description": "Horizontal Z coordinate centered at 0"},
                    "color": {"type": "string", "description": "Hex color string (e.g. #3b82f6)"},
                    "part": {"type": "string", "description": "Body part name like head, leg, body, wing"}
                },
                "required": ["x", "y", "z", "color"],
                "additionalProperties": False
            }
        }
    },
    "required": ["id", "name", "category", "description", "movementStyle", "voxels"],
    "additionalProperties": False
}


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
    Guarantees the script never crashes even if workstation is unreachable.
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


def ask_workstation_for_creature(theme_prompt: str):
    """
    Connects to the local AI workstation matching agent.py:
    URL: http://192.168.12.253:11434/v1/chat/completions
    Model: qwen3:14b
    """
    user_prompt = (
        f"Design a unique voxel creature for an outdoor wilderness world.\n"
        f"Category theme: {theme_prompt}\n\n"
        f"CRITICAL DESIGN CONSTRAINTS:\n"
        f"1. CENTERING: All voxel coordinates MUST be horizontally centered around x=0 and z=0.\n"
        f"2. GROUNDING: The feet, belly, or base of the creature MUST sit strictly at y=0. Never use negative y.\n"
        f"3. PERFORMANCE: Provide between 35 and 90 voxels for smooth 60 FPS real-time rendering.\n"
        f"4. PALETTE: Use vibrant, tasteful retro hex colors matching the creature's lore.\n"
        f"5. OUTPUT: Return strictly the raw JSON matching the schema."
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.8,
        "format": CREATURE_SCHEMA,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "creature_data_schema",
                "strict": True,
                "schema": CREATURE_SCHEMA
            }
        }
    }

    for attempt in range(3):
        try:
            print(f"[Creature Factory] Contacting AI workstation at {WORKSTATION_URL} (Attempt {attempt+1}/3)...")
            response = requests.post(WORKSTATION_URL, json=payload, timeout=1800)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            creature_dict = json.loads(content)

            # Defensive validation
            if not isinstance(creature_dict, dict):
                raise ValueError("Payload is not a valid JSON object")
            if "voxels" not in creature_dict or not isinstance(creature_dict["voxels"], list):
                raise ValueError("Creature missing required 'voxels' array")
            if len(creature_dict["voxels"]) < 8:
                raise ValueError("Creature model has insufficient voxel density")

            return creature_dict

        except Exception as e:
            print(f"[Warning] Workstation generation attempt failed: {e}")
            time.sleep(5)

    return None


def run_generation_cycle(theme: str = None, push_git: bool = True) -> tuple:
    """
    Executes a single autonomous creature generation cycle:
    1. Selects or takes a theme
    2. Queries workstation (or triggers safe procedural fallback)
    3. Normalizes centering and grounding
    4. Applies random suffix protection
    5. Saves file to assets/creatures/[id]_[suffix].json
    6. Updates manifest.json
    7. Commits and pushes to GitHub
    """
    categories = [
        "Small friendly wildlife (e.g. woodland dog, pine fox, alpine badger, meadow pig, river beaver, horned hare)",
        "Mysterious wilderness monster (e.g. ancient moss golem, bone scavenger, stone crawler, lichen spider, root horror)",
        "Legendary outdoor boss or guardian (e.g. celestial sky drake, glacial frost dragon, basalt colossus, elder forest stag)"
    ]
    chosen_theme = theme or random.choice(categories)

    print(f"\n[Creature Factory] Starting creature generation cycle...")
    print(f"[Creature Factory] Target Theme: {chosen_theme}")

    raw_data = ask_workstation_for_creature(chosen_theme)
    suffix = generate_suffix()

    if raw_data:
        base_id = str(raw_data.get("id", "creature")).lower().replace(" ", "_")
        clean_id = base_id.split("_")[0] if len(base_id) > 12 else base_id
        final_id = f"{clean_id}_{suffix}"

        sanitized_voxels = enforce_centering_and_grounding(raw_data.get("voxels", []))
        creature_data = {
            "id": final_id,
            "name": str(raw_data.get("name", "Voxel Wonder")).strip(),
            "category": str(raw_data.get("category", "wildlife")).lower(),
            "description": str(raw_data.get("description", "A mysterious inhabitant of the voxel wilderness.")).strip(),
            "movementStyle": str(raw_data.get("movementStyle", "walking")).lower(),
            "scale": float(raw_data.get("scale", 1.0)),
            "dimensions": calculate_dimensions(sanitized_voxels),
            "voxels": sanitized_voxels
        }
        print(f"[Creature Factory] Successfully generated from workstation: '{creature_data['name']}' ({final_id})")
    else:
        print("[Creature Factory] Notice: Workstation unavailable after retries. Generating safe procedural fallback...")
        creature_data = generate_fallback_creature(chosen_theme)
        print(f"[Creature Factory] Created procedural fallback creature: '{creature_data['name']}' ({creature_data['id']})")

    # Save to assets/creatures/[id].json with suffix protection
    filename = f"{creature_data['id']}.json"
    file_path = os.path.join(CREATURES_DIR, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(creature_data, f, indent=2)

    # Update master manifest.json
    rel_path = os.path.join("assets", "creatures", filename).replace("\\", "/")
    update_manifest(creature_data, rel_path)

    print(f"[Creature Factory] Saved: {file_path}")
    print(f"[Creature Factory] Manifest updated at: {MANIFEST_PATH}")

    # Synchronize to Git (matching agent.py)
    if push_git:
        try:
            c_name = creature_data.get("name", "Creature")
            print("[Creature Factory] Committing and pushing to GitHub...")
            os.system("git add assets/creatures/")
            os.system(f'git commit -m "Auto-generated autonomous creature [{c_name}] with manifest update"')
            os.system("git pull --rebase origin main")
            os.system("git push origin main")
            print("[Creature Factory] Creature successfully pushed to GitHub & Live Site!")
        except Exception as e:
            print(f"[Creature Factory] Git sync note: {e}")

    return creature_data, file_path


# Alias for backward compatibility with external imports
def generate_creature(theme: str = None) -> tuple:
    return run_generation_cycle(theme=theme, push_git=False)


def main():
    parser = argparse.ArgumentParser(description="Autonomous Voxel Creature Factory")
    parser.add_argument("--once", action="store_true", help="Run a single generation cycle and exit")
    parser.add_argument("--count", type=int, default=1, help="Number of creatures to generate (when using --once)")
    parser.add_argument("--theme", type=str, default=None, help="Specific creature theme or category prompt")
    parser.add_argument("--interval", type=int, default=INTERVAL_SECONDS, help="Seconds between autonomous generation cycles (default: 3600)")
    parser.add_argument("--no-git", action="store_true", help="Skip git add/commit/push during execution")
    args = parser.parse_args()

    interval = args.interval
    push_git = not args.no_git

    print("==================================================")
    print(" Voxel Creature Factory - Autonomous Agent Online ")
    print("==================================================")
    print(f"Target Workstation: {WORKSTATION_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Interval: {interval} seconds ({interval / 60:.1f} minutes)")
    print("==================================================")

    if args.once:
        for i in range(args.count):
            print(f"\n--- Generating Creature {i+1}/{args.count} ---")
            run_generation_cycle(theme=args.theme, push_git=push_git)
        return

    # Continuous autonomous interval loop matching agent.py
    while True:
        run_generation_cycle(theme=args.theme, push_git=push_git)
        print(f"\nSleeping for {interval / 60:.1f} minutes before next creature generation cycle...\n")
        time.sleep(interval)


if __name__ == "__main__":
    main()
