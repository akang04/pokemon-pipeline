"""
Fetch Pokémon data from PokéAPI and persist it to SQLite.

Usage:
    python fetch.py              # fetch all ~1025 Pokémon
    python fetch.py --limit 151  # fetch only Gen 1 (by Pokédex order)
    python fetch.py --offset 0 --limit 10  # quick test run
"""

import argparse
import time

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

from database import (
    Ability,
    BaseStat,
    Evolution,
    EvolutionChain,
    Generation,
    Move,
    Pokemon,
    PokemonAbility,
    PokemonMove,
    PokemonType,
    Type,
    get_engine,
    init_db,
    run_migrations,
)

POKEAPI_BASE = "https://pokeapi.co/api/v2"
REQUEST_DELAY = 0.1  # seconds between requests — polite to the public API


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

class PokeAPIClient:
    """Thin wrapper around requests with retry + rate-limit delay."""

    def __init__(self, base_url: str = POKEAPI_BASE, delay: float = REQUEST_DELAY):
        self.base_url = base_url
        self.delay = delay
        self._session = requests.Session()

    def get(self, endpoint: str) -> dict:
        return self._fetch(f"{self.base_url}/{endpoint}")

    def get_url(self, url: str) -> dict:
        """Fetch an absolute URL directly (for following hrefs in API responses)."""
        return self._fetch(url)

    def _fetch(self, url: str, retries: int = 3) -> dict:
        for attempt in range(retries):
            try:
                response = self._session.get(url, timeout=10)
                response.raise_for_status()
                time.sleep(self.delay)
                return response.json()
            except requests.exceptions.RequestException as exc:
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"  [retry {attempt + 1}/{retries}] {url} — waiting {wait}s ({exc})")
                time.sleep(wait)


# ---------------------------------------------------------------------------
# Per-resource fetchers (one function per API endpoint — add new ones here)
# ---------------------------------------------------------------------------

def fetch_pokemon_list(client: PokeAPIClient, limit: int = 1025, offset: int = 0) -> list[dict]:
    """Return a list of {name, url} stubs for Pokémon in the given range."""
    data = client.get(f"pokemon?limit={limit}&offset={offset}")
    return data["results"]


def fetch_pokemon(client: PokeAPIClient, identifier: int | str) -> dict:
    """Fetch the full data payload for one Pokémon by ID or name."""
    return client.get(f"pokemon/{identifier}")


def fetch_species(client: PokeAPIClient, identifier: int | str) -> dict:
    """Fetch species data for a Pokémon — contains generation and evolution chain refs."""
    return client.get(f"pokemon-species/{identifier}")


def fetch_ability(client: PokeAPIClient, ability_id: int) -> dict:
    """Fetch full ability data — contains multilingual effect descriptions."""
    return client.get(f"ability/{ability_id}")


def fetch_evolution_chain(client: PokeAPIClient, chain_id: int) -> dict:
    """Fetch a full evolution chain — contains a nested tree of species and triggers."""
    return client.get(f"evolution-chain/{chain_id}")


def fetch_move(client: PokeAPIClient, move_id: int) -> dict:
    """Fetch full move data — contains type, power, accuracy, PP, and damage class."""
    return client.get(f"move/{move_id}")


def fetch_all_generations(client: PokeAPIClient) -> dict:
    """
    Pre-fetch all generations and return a mapping:
        { generation_name: {"id": int, "region": str} }
    Keeps the main loop from firing one extra request per Pokémon.
    """
    index = client.get("generation")
    result = {}
    for entry in index["results"]:
        data = client.get_url(entry["url"])
        result[data["name"]] = {
            "id": data["id"],
            "region": data["main_region"]["name"] if data.get("main_region") else None,
        }
    return result


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _english_effect(effect_entries: list[dict]) -> str | None:
    """Return the short English effect string from a PokéAPI effect_entries list."""
    for entry in effect_entries:
        if entry["language"]["name"] == "en":
            return entry.get("short_effect") or entry.get("effect")
    return None


def _id_from_url(url: str) -> int:
    """Extract the numeric resource ID from a PokéAPI URL.

    "https://pokeapi.co/api/v2/type/10/" -> 10
    """
    return int(url.rstrip("/").rsplit("/", 1)[-1])


def parse_move(raw: dict) -> dict:
    """Normalise a raw /move/{id} response into a flat dict for DB insertion."""
    return {
        "id": raw["id"],
        "name": raw["name"],
        "type_id": _id_from_url(raw["type"]["url"]),
        "power": raw.get("power"),
        "accuracy": raw.get("accuracy"),
        "pp": raw.get("pp"),
        "damage_class": raw["damage_class"]["name"] if raw.get("damage_class") else None,
    }


def parse_pokemon_moves(raw: dict) -> list[dict]:
    """
    Extract one row per (move_id, learn_method) from a Pokémon's move list.

    PokéAPI repeats each move once per game it appeared in (version_group_details).
    We deduplicate to one row per (move, method) pair and keep the highest
    level_learned_at across all versions — later games often rebalance level-up moves.
    """
    best: dict[tuple, dict] = {}
    for m in raw["moves"]:
        move_id = _id_from_url(m["move"]["url"])
        move_name = m["move"]["name"]
        for vgd in m["version_group_details"]:
            method = vgd["move_learn_method"]["name"]
            level = vgd["level_learned_at"] or None
            key = (move_id, method)
            if key not in best:
                best[key] = {
                    "move_id": move_id,
                    "move_name": move_name,
                    "learn_method": method,
                    "level_learned": level,
                }
            elif level is not None:
                existing = best[key]["level_learned"]
                if existing is None or level > existing:
                    best[key]["level_learned"] = level
    return list(best.values())


def parse_pokemon(raw: dict, species: dict) -> dict:
    """Normalise raw API payloads into a flat dict ready for DB insertion."""
    stats = {s["stat"]["name"]: s["base_stat"] for s in raw["stats"]}
    return {
        "id": raw["id"],
        "name": raw["name"],
        "height": raw["height"],
        "weight": raw["weight"],
        "base_experience": raw.get("base_experience"),
        "sprite_url": (raw["sprites"] or {}).get("front_default"),
        "generation_name": species["generation"]["name"],
        "is_legendary": species.get("is_legendary", False),
        "is_mythical":  species.get("is_mythical",  False),
        "is_baby":      species.get("is_baby",      False),
        "types": [
            {
                "slot": t["slot"],
                "type_id": _id_from_url(t["type"]["url"]),
                "type_name": t["type"]["name"],
            }
            for t in raw["types"]
        ],
        "stats": {
            "hp": stats.get("hp"),
            "attack": stats.get("attack"),
            "defense": stats.get("defense"),
            "special_attack": stats.get("special-attack"),
            "special_defense": stats.get("special-defense"),
            "speed": stats.get("speed"),
        },
        "abilities": [
            {
                "ability_id": _id_from_url(a["ability"]["url"]),
                "ability_name": a["ability"]["name"],
                "is_hidden": a["is_hidden"],
            }
            for a in raw["abilities"]
        ],
        "moves": parse_pokemon_moves(raw),
        "evolution_chain_id": _id_from_url(species["evolution_chain"]["url"]),
    }


def parse_evolution_chain(chain_data: dict) -> list[dict]:
    """Flatten a nested evolution chain tree into a list of (from, to, trigger, min_level) rows.

    PokéAPI returns chains as a recursive tree: each node has a species and an
    evolves_to list of child nodes.  We walk it depth-first and emit one row per
    directed edge (parent → child).
    """
    rows: list[dict] = []

    def walk(node: dict, parent_name: str | None) -> None:
        name = node["species"]["name"]
        if parent_name is not None:
            details = (node.get("evolution_details") or [{}])[0]
            trigger_obj = details.get("trigger")
            rows.append({
                "from_name": parent_name,
                "to_name": name,
                "trigger": trigger_obj["name"] if trigger_obj else None,
                # min_level is 0 when not applicable — treat as NULL
                "min_level": details.get("min_level") or None,
            })
        for child in node.get("evolves_to", []):
            walk(child, name)

    walk(chain_data["chain"], None)
    return rows


# ---------------------------------------------------------------------------
# Database write helpers
# ---------------------------------------------------------------------------

def ensure_generation(session: Session, name: str, gen_map: dict) -> Generation:
    """Get or create a Generation row, using the pre-fetched gen_map for metadata."""
    meta = gen_map[name]
    obj = session.get(Generation, meta["id"])
    if obj is None:
        obj = Generation(id=meta["id"], name=name, region=meta["region"])
        session.add(obj)
    return obj


def ensure_type(session: Session, type_id: int, type_name: str) -> Type:
    """Get or create a Type row."""
    obj = session.get(Type, type_id)
    if obj is None:
        obj = Type(id=type_id, name=type_name)
        session.add(obj)
    return obj


def ensure_ability(session: Session, ability_id: int, ability_name: str, effect: str | None) -> Ability:
    """Get or create an Ability row."""
    obj = session.get(Ability, ability_id)
    if obj is None:
        obj = Ability(id=ability_id, name=ability_name, effect=effect)
        session.add(obj)
    return obj


def ensure_move(session: Session, move_data: dict) -> Move:
    """Get or create a Move row."""
    obj = session.get(Move, move_data["id"])
    if obj is None:
        obj = Move(**move_data)
        session.add(obj)
    return obj


def save_pokemon(session: Session, data: dict, gen_map: dict) -> None:
    """Upsert one Pokémon and all its Phase 1 related rows."""
    generation = ensure_generation(session, data["generation_name"], gen_map)

    # Upsert the Pokémon row
    pokemon = session.get(Pokemon, data["id"])
    if pokemon is None:
        pokemon = Pokemon(id=data["id"])
        session.add(pokemon)

    pokemon.name = data["name"]
    pokemon.generation_id = generation.id
    pokemon.height = data["height"]
    pokemon.weight = data["weight"]
    pokemon.base_experience = data["base_experience"]
    pokemon.sprite_url = data["sprite_url"]
    pokemon.is_legendary = data.get("is_legendary", False)
    pokemon.is_mythical  = data.get("is_mythical",  False)
    pokemon.is_baby      = data.get("is_baby",      False)

    # Flush so the PK exists before writing child rows
    session.flush()

    # Types — delete existing slots then re-insert (handles type changes on re-run)
    for pt in list(pokemon.types):
        session.delete(pt)
    session.flush()

    for t in data["types"]:
        type_obj = ensure_type(session, t["type_id"], t["type_name"])
        session.flush()
        session.add(PokemonType(pokemon_id=pokemon.id, type_id=type_obj.id, slot=t["slot"]))

    # Stats — upsert
    if pokemon.stats is None:
        session.add(BaseStat(pokemon_id=pokemon.id, **data["stats"]))
    else:
        for col, val in data["stats"].items():
            setattr(pokemon.stats, col, val)

    # Abilities — delete existing then re-insert
    for pa in list(pokemon.abilities):
        session.delete(pa)
    session.flush()

    for a in data["abilities"]:
        ability_obj = ensure_ability(session, a["ability_id"], a["ability_name"], a["effect"])
        session.flush()
        session.add(PokemonAbility(pokemon_id=pokemon.id, ability_id=ability_obj.id, is_hidden=a["is_hidden"]))

    # Moves — delete existing then re-insert (move_data populated by caller via move_cache)
    for pm in list(pokemon.moves):
        session.delete(pm)
    session.flush()

    for m in data.get("moves_resolved", []):
        ensure_move(session, m["move_data"])
        session.flush()
        session.add(PokemonMove(
            pokemon_id=pokemon.id,
            move_id=m["move_id"],
            learn_method=m["learn_method"],
            level_learned=m["level_learned"],
        ))


def save_evolution_chain(session: Session, chain_id: int, evolutions: list[dict]) -> None:
    """Upsert one evolution chain and all its evolution rows."""
    chain = session.get(EvolutionChain, chain_id)
    if chain is None:
        chain = EvolutionChain(id=chain_id)
        session.add(chain)
        session.flush()

    # Delete existing rows so re-runs are idempotent
    for ev in list(chain.evolutions):
        session.delete(ev)
    session.flush()

    for row in evolutions:
        from_pk = session.execute(select(Pokemon).where(Pokemon.name == row["from_name"])).scalar_one_or_none()
        to_pk = session.execute(select(Pokemon).where(Pokemon.name == row["to_name"])).scalar_one_or_none()
        if from_pk is None or to_pk is None:
            # Pokémon not in our DB (e.g. fetch was run with --limit)
            continue
        session.add(Evolution(
            chain_id=chain_id,
            from_pokemon_id=from_pk.id,
            to_pokemon_id=to_pk.id,
            trigger=row["trigger"],
            min_level=row["min_level"],
        ))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_species_update() -> None:
    """Back-fill is_legendary / is_mythical / is_baby for all Pokémon already in the DB.

    Only calls the /pokemon-species/{id} endpoint — ~1025 requests at 0.1 s each,
    so roughly 2 minutes total.  All other data is left untouched.
    """
    engine = init_db()
    run_migrations(engine)
    client = PokeAPIClient()

    with Session(engine) as session:
        pokemon_ids = sorted(row[0] for row in session.execute(select(Pokemon.id)).fetchall())

    total = len(pokemon_ids)
    print(f"Updating species classifications for {total} Pokémon...")

    with Session(engine) as session:
        for i, pid in enumerate(pokemon_ids, 1):
            print(f"[{i:>4}/{total}] species {pid}", end=" ... ", flush=True)
            try:
                species = fetch_species(client, pid)
                p = session.get(Pokemon, pid)
                if p is not None:
                    p.is_legendary = species.get("is_legendary", False)
                    p.is_mythical  = species.get("is_mythical",  False)
                    p.is_baby      = species.get("is_baby",      False)
                    session.commit()
                print("ok")
            except Exception as exc:
                session.rollback()
                print(f"FAILED — {exc}")

    print("\nDone.")


def run(limit: int = 1025, offset: int = 0) -> None:
    engine = init_db()
    run_migrations(engine)

    client = PokeAPIClient()

    print("Fetching generations...")
    gen_map = fetch_all_generations(client)
    print(f"  Found {len(gen_map)} generations.")

    print(f"Fetching Pokémon list (offset={offset}, limit={limit})...")
    pokemon_stubs = fetch_pokemon_list(client, limit=limit, offset=offset)
    total = len(pokemon_stubs)
    print(f"  {total} Pokémon to fetch.\n")

    ability_cache: dict[int, str | None] = {}   # ability_id -> effect text, fetched once per run
    move_cache: dict[int, dict | None] = {}      # move_id -> parsed move dict, fetched once per run
    chain_ids: set[int] = set()

    with Session(engine) as session:
        for i, stub in enumerate(pokemon_stubs, start=1):
            name = stub["name"]
            print(f"[{i:>4}/{total}] {name}", end=" ... ", flush=True)

            try:
                raw = fetch_pokemon(client, name)
                species = fetch_species(client, raw["id"])
                data = parse_pokemon(raw, species)

                for ab in data["abilities"]:
                    aid = ab["ability_id"]
                    if aid not in ability_cache:
                        try:
                            ab_data = fetch_ability(client, aid)
                            ability_cache[aid] = _english_effect(ab_data.get("effect_entries", []))
                        except Exception:
                            ability_cache[aid] = None
                    ab["effect"] = ability_cache[aid]

                # Resolve move data, fetching each unique move once
                moves_resolved = []
                for m in data["moves"]:
                    mid = m["move_id"]
                    if mid not in move_cache:
                        try:
                            move_cache[mid] = parse_move(fetch_move(client, mid))
                        except Exception:
                            move_cache[mid] = None
                    if move_cache[mid] is not None:
                        moves_resolved.append({**m, "move_data": move_cache[mid]})
                data["moves_resolved"] = moves_resolved

                save_pokemon(session, data, gen_map)
                session.commit()
                chain_ids.add(data["evolution_chain_id"])
                print("ok")
            except Exception as exc:
                session.rollback()
                print(f"FAILED — {exc}")

        # Second pass: fetch each unique evolution chain exactly once
        sorted_chains = sorted(chain_ids)
        total_chains = len(sorted_chains)
        print(f"\nFetching {total_chains} unique evolution chains...")
        for j, chain_id in enumerate(sorted_chains, start=1):
            print(f"[{j:>4}/{total_chains}] chain {chain_id}", end=" ... ", flush=True)
            try:
                chain_data = fetch_evolution_chain(client, chain_id)
                evolutions = parse_evolution_chain(chain_data)
                save_evolution_chain(session, chain_id, evolutions)
                session.commit()
                print("ok")
            except Exception as exc:
                session.rollback()
                print(f"FAILED — {exc}")

    print("\nDone.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Pokémon data into SQLite.")
    parser.add_argument("--limit", type=int, default=1025, help="Number of Pokémon to fetch")
    parser.add_argument("--offset", type=int, default=0, help="Pokédex offset to start from")
    parser.add_argument("--species-only", action="store_true",
                        help="Only update is_legendary/mythical/baby from species endpoint (~2 min)")
    args = parser.parse_args()

    if args.species_only:
        run_species_update()
    else:
        run(limit=args.limit, offset=args.offset)
