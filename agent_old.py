import os
import json
import time
import requests

# --- CONFIGURATION ---
WORKSTATION_URL = "http://192.168.12.253:11434/v1/chat/completions"
MODEL_NAME = "qwen3:14b"
STATE_FILE = "world_state.json"
CHUNKS_DIR = "chunks"
MAX_BOUNDARY = 8                   # Max grid radius (prevents infinite expansion)
INTERVAL_SECONDS = 3600            # 1 hour between expansion cycles

os.makedirs(CHUNKS_DIR, exist_ok=True)

SYSTEM_PROMPT = """You are a procedural world generator for an outdoor voxel game featuring 'Little Doug'.
You generate strict JSON data for map chunks (16x16 grids).

Valid Biome Types (Choose ONE based on terrain variety):
- lush_riverbank
- river
- stream
- wetland
- alpine_lake
- estuary
- mangrove_swamp
- bog
- fen
- tidal_pool
- deep_ocean
- coral_reef
- pine_forest
- ancient_forest
- deciduous_forest
- old_growth_forest
- temperate_rainforest
- boreal_forest
- autumn_woods
- bamboo_grove
- woodland_edge
- mossy_grove
- mountain_ridge
- rocky_outcrop
- alpine_tundra
- canyon
- steep_cliff
- highland_plateau
- foothills
- misty_valley
- desert
- badlands
- salt_flat
- oasis
- savanna
- shrubland
- mesa
- rolling_meadow
- snow
- glacier
- frozen_tundra
- frozen_lake
- ice_sheet
- volcanic_field
- overgrown_ruins
- cave_mouth
- cairn_field

Rules:
1. Ensure elevation values (0 to 5) match neighboring chunks smoothly.
2. Include a 16x16 elevation_grid.
3. The 'specimens' array MUST contain EXACTLY 8 items. Array length == 8.
4. Set 'biome_type' using ONE string from the allowed list above. Vary the biome naturally across coordinates.
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


def ask_workstation_for_chunk(coord):
    x, y = coord
    prompt = (
        f"Generate chunk data for outdoor grid coordinate X:{x}, Y:{y}. "
        f"Select an appropriate biome_type from the system prompt list, a 16x16 elevation_grid, "
        f"and EXACTLY 8 collectible specimens or wildlife items."
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.85
    }

    # Retry loop for defensive parsing
    for attempt in range(3):
        try:
            print(f"[Pi Agent] Contacting workstation (Attempt {attempt+1}/3)...")
            response = requests.post(WORKSTATION_URL, json=payload, timeout=240)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]

            # Clean markdown code blocks if Llama/Qwen wraps them
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

    # Grab the next chunk to build
    target_coord = state["queue"].pop(0)
    x, y = target_coord

    # Enforce boundary limits for long-term sustainability
    if abs(x) > MAX_BOUNDARY or abs(y) > MAX_BOUNDARY:
        print(f"[Pi Agent] Coordinate {target_coord} is outside max boundary ({MAX_BOUNDARY}). Skipping.")
        save_state(state)
        return

    print(f"\n[Pi Agent] Expanding world... Building chunk at coordinate: [{x}, {y}]")

    chunk_data = ask_workstation_for_chunk(target_coord)

    if chunk_data:
        # Ensure coordinate is explicitly set in chunk data
        chunk_data["coordinate"] = [x, y]

        # Save chunk file
        filename = os.path.join(CHUNKS_DIR, f"chunk_{x}_{y}.json")
        with open(filename, "w") as f:
            json.dump(chunk_data, f, indent=2)

        # Update explored list
        if target_coord not in state["explored"]:
            state["explored"].append(target_coord)

        # Add new adjacent coordinates to the queue if within boundaries
        neighbors = [[x+1, y], [x-1, y], [x, y+1], [x, y-1]]
        for n in neighbors:
            nx, ny = n
            if abs(nx) <= MAX_BOUNDARY and abs(ny) <= MAX_BOUNDARY:
                if n not in state["explored"] and n not in state["queue"]:
                    state["queue"].append(n)

        save_state(state)
        print(f"[Pi Agent] Saved {filename}. Committing and pushing to GitHub...")

        # Git automated workflow (stage and commit first to prevent rebase conflicts)
        os.system("git add .")
        os.system(f'git commit -m "Auto-expanded world chunk [{x}, {y}]"')
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
