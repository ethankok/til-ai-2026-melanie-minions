import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
json_file = REPO_ROOT / "training" / "ae" / "data" / "novice_map_info.json"
py_file = REPO_ROOT / "ae" / "src" / "novice_map_data.py"

def main():
    with open(json_file, "r") as f:
        data = json.load(f)

    # Convert walls_list
    walls_list = []
    for item in data["walls_list"]:
        pos, dirs = item
        for d in dirs:
            walls_list.append((pos[0], pos[1], d))

    # Convert destructible_list
    destructible_list = []
    for item in data["destructible_list"]:
        pos, dirs = item
        for d in dirs:
            destructible_list.append((pos[0], pos[1], d))

    # Convert static_entities
    static_entities = []
    for item in data["static_entities"]:
        kind, pos = item
        static_entities.append((kind, (pos[0], pos[1])))

    with open(py_file, "w") as f:
        f.write('"""Pre-computed layout details for the Novice fixed map."""\n\n')
        
        f.write("# symmetric base configurations\n")
        f.write(f"BASE_LOCATIONS = {data['base_locations']}\n\n")
        
        f.write("# symmetric starting agent configurations\n")
        f.write(f"STARTING_LOCATIONS = {data['starting_locations']}\n\n")
        
        f.write("# Pre-computed set of wall edges (x, y, direction)\n")
        f.write(f"WALLS = {walls_list}\n\n")
        
        f.write("# Pre-computed set of destructible wall edges (x, y, direction)\n")
        f.write(f"DESTRUCTIBLE = {destructible_list}\n\n")
        
        f.write("# Pre-computed item coordinates (kind, (x, y))\n")
        f.write(f"STATIC_ENTITIES = {static_entities}\n")

    print(f"Generated python map data module at {py_file}")

if __name__ == "__main__":
    main()
