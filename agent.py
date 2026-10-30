import os
import json
import time
import requests

# --- CONFIGURATION ---
WORKSTATION_URL = "http://192.168.12.253:11434/v1/chat/completions"
MODEL_NAME = "qwen3:14b"
STATE_FILE = "world_state.json"
CHUNKS_DIR = "chunks"
MAX_BOUNDARY = 8                     # Max grid radius
INTERVAL_SECONDS = 3600              # 1 hour between expansion cycles

os.makedirs(CHUNKS_DIR, exist_ok=True)

SYSTEM_PROMPT = """You are a procedural ecosystem architect and also a DUNGEON MASTER for an outdoor voxel game featuring 'Little Doug'.
You generate strict JSON data for map chunks (16x16 grids, elevation 0 to 12).

Valid Biome Types (Choose ONE based on terrain continuity):
- lush_riverbank, river, stream, wetland, alpine_lake, estuary, mangrove_swamp, bog, fen
- pine_forest, ancient_forest, deciduous_forest, temperate_rainforest, boreal_forest, autumn_woods
- mountain_ridge, rocky_outcrop, alpine_tundra, canyon, steep_cliff, highland_plateau, foothills
- desert, badlands, salt_flat, oasis, savanna, shrubland, mesa, rolling_meadow
- snow, glacier, frozen_tundra, frozen_lake, ice_sheet, volcanic_field

Rules:
1. CONTINUITY: Match elevation values smoothly along the borders shared with neighboring chunks described in the prompt.
2. 16x16 Grid: Include an 'elevation_grid' (array of 16 rows, each with 16 integers).
3. Flora: Include a 'flora' array containing native vegetation objects with 'type' (e.g., 'deciduous_tree', 'pine_tree', 'cactus'), local coordinates 'x' and 'z' (ranges between -7.5 and 7.5), and 'height'.
4. Specimens: The 'specimens' array MUST contain ONLY 1 or 2 rare, well-spaced collectible items native to your chosen biome. Each item must include 'name' (string), 'x' (float between -6 and 6), 'y' (float between -6 and 6), 'points' (integer), and a short 'lore' (string, 1-2 sentences of educational or environmental background about the specimen).
Output ONLY raw, valid JSON. No conversational filler, no markdown outside of code blocks."""


def load_state():
    if not os.path.exists(STATE_FILE):
        initial_state = {
            "explored": [[0, 0]],
            "queue": [[0, 1], [1, 0], [-1, 0], [0, -1]]
        }
        with open(STATE_FILE, "w") as f:
            json.dump(initial_state, f, indent=2)
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_neighbor_context(x, y):
    """Inspects adjacent chunks on disk to provide seamless world-building context."""
    neighbors_info = []

    checks = [
        ("North (Y+1)", x, y + 1),
        ("South (Y-1)", x, y - 1),
        ("East (X+1)", x + 1, y),
        ("West (X-1)", x - 1, y)
    ]

    for label, nx, ny in checks:
        path = os.path.join(CHUNKS_DIR, f"chunk_{nx}_{ny}.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    biome = data.get("biome_type", "unknown biome")
                    grid = data.get("elevation_grid", [])

                    edge_summary = "Edge connected"
                    if grid:
                        if label.startswith("North"): edge_summary = f"South edge elevations: {grid[0]}"
                        elif label.startswith("South"): edge_summary = f"North edge elevations: {grid[-1]}"
                        elif label.startswith("East"): edge_summary = f"West edge elevations: {[row[0] for row in grid]}"
                        elif label.startswith("West"): edge_summary = f"East edge elevations: {[row[-1] for row in grid]}"

                    neighbors_info.append(f"- {label}: Biome is '{biome}'. {edge_summary}")
            except Exception:
                pass

    if not neighbors_info:
        return "This is a frontier chunk with no generated neighbors yet. Establish a fresh baseline biome."

    return "Adjacent established territory:\n" + "\n".join(neighbors_info)


def ask_workstation_for_chunk(coord):
    x, y = coord
    context_desc = get_neighbor_context(x, y)

    prompt = (
        f"Generate chunk data for outdoor grid coordinate X:{x}, Y:{y}.\n"
        f"{context_desc}\n\n"
        f"Select a biome_type that transitions logically from the neighbors above, "
        f"build a matching 16x16 elevation_grid, populate native flora objects in the 'flora' array, "
        f"and provide EXACTLY 8 native collectible specimens."
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.8
    }

    for attempt in range(3):
        try:
            print(f"[Pi Agent] Contacting workstation with neighbor context (Attempt {attempt+1}/3)...")
            response = requests.post(WORKSTATION_URL, json=payload, timeout=600)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)
        except Exception as e:
            print(f"[Warning] Generation attempt failed: {e}")
            time.sleep(5)

    return None


def run_expansion_cycle():
    state = load_state()

    if not state["queue"]:
        print("[Pi Agent] World expansion queue is empty!")
        return

    target_coord = state["queue"].pop(0)
    x, y = target_coord

    if abs(x) > MAX_BOUNDARY or abs(y) > MAX_BOUNDARY:
        print(f"[Pi Agent] Coordinate {target_coord} is outside max boundary ({MAX_BOUNDARY}). Skipping.")
        save_state(state)
        return

    print(f"\n[Pi Agent] Expanding world... Building chunk at coordinate: [{x}, {y}]")

    chunk_data = ask_workstation_for_chunk(target_coord)

    if chunk_data:
        chunk_data["chunk_coordinate"] = [x, y]
        filename = os.path.join(CHUNKS_DIR, f"chunk_{x}_{y}.json")
        with open(filename, "w") as f:
            json.dump(chunk_data, f, indent=2)

        if target_coord not in state["explored"]:
            state["explored"].append(target_coord)

        neighbors = [[x+1, y], [x-1, y], [x, y+1], [x, y-1]]
        for n in neighbors:
            nx, ny = n
            if abs(nx) <= MAX_BOUNDARY and abs(ny) <= MAX_BOUNDARY:
                if n not in state["explored"] and n not in state["queue"]:
                    state["queue"].append(n)

        save_state(state)
        print(f"[Pi Agent] Saved {filename}. Committing and pushing to GitHub...")

        os.system("git add .")
        os.system(f'git commit -m "Auto-expanded world chunk [{x}, {y}] with autonomous flora and terrain context"')
        os.system("git pull --rebase origin main")
        os.system("git push origin main")
        print("[Pi Agent] Chunk successfully pushed to GitHub & Live Site!")
    else:
        print("[Pi Agent] Chunk generation failed after retries. Putting coordinate back in queue.")
        state["queue"].append(target_coord)
        save_state(state)


if __name__ == "__main__":
    print("==================================================")
    print(" Voxel Little Doug - Autonomous Pi Agent Online   ")
    print("==================================================")
    while True:
        run_expansion_cycle()
        print(f"Sleeping for {INTERVAL_SECONDS / 60} minutes before next world-building cycle...\n")
        time.sleep(INTERVAL_SECONDS)
