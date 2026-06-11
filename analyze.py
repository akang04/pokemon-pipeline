"""
Analysis helpers — pure pandas, no Streamlit imports.
All functions accept an optional engine so they're testable in isolation.
"""

import pandas as pd
from sqlalchemy import text

from database import get_engine

STAT_COLS = ["hp", "attack", "defense", "special_attack", "special_defense", "speed"]

STAT_LABELS = {
    "hp": "HP",
    "attack": "Attack",
    "defense": "Defense",
    "special_attack": "Sp. Atk",
    "special_defense": "Sp. Def",
    "speed": "Speed",
    "total_stat": "Total",
}

GEN_LABEL = {
    "generation-i":    "Gen I",
    "generation-ii":   "Gen II",
    "generation-iii":  "Gen III",
    "generation-iv":   "Gen IV",
    "generation-v":    "Gen V",
    "generation-vi":   "Gen VI",
    "generation-vii":  "Gen VII",
    "generation-viii": "Gen VIII",
    "generation-ix":   "Gen IX",
}

TYPE_COLORS = {
    "normal":   "#A8A878",
    "fire":     "#F08030",
    "water":    "#6890F0",
    "electric": "#F8D030",
    "grass":    "#78C850",
    "ice":      "#98D8D8",
    "fighting": "#C03028",
    "poison":   "#A040A0",
    "ground":   "#E0C068",
    "flying":   "#A890F0",
    "psychic":  "#F85888",
    "bug":      "#A8B820",
    "rock":     "#B8A038",
    "ghost":    "#705898",
    "dragon":   "#7038F8",
    "dark":     "#705848",
    "steel":    "#B8B8D0",
    "fairy":    "#EE99AC",
}

_LONG_QUERY = """
    SELECT
        p.id,
        p.name,
        p.height,
        p.weight,
        p.base_experience,
        p.sprite_url,
        g.id          AS generation_id,
        g.name        AS generation_name,
        g.region,
        t.id          AS type_id,
        t.name        AS type_name,
        pt.slot       AS type_slot,
        s.hp,
        s.attack,
        s.defense,
        s.special_attack,
        s.special_defense,
        s.speed
    FROM pokemon p
    JOIN generations   g  ON p.generation_id = g.id
    JOIN pokemon_types pt ON p.id            = pt.pokemon_id
    JOIN types         t  ON pt.type_id      = t.id
    JOIN base_stats    s  ON p.id            = s.pokemon_id
    ORDER BY p.id, pt.slot
"""


def load_pokemon_df(engine=None) -> pd.DataFrame:
    """
    One row per (pokemon, type).  Dual-type pokemon appear twice.
    Use this for per-type grouping where every type assignment counts.
    """
    if engine is None:
        engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(_LONG_QUERY, conn)
    df["total_stat"] = df[STAT_COLS].sum(axis=1)
    df["gen_label"] = df["generation_name"].map(GEN_LABEL).fillna(df["generation_name"])
    return df


def _evo_stage_series(engine) -> pd.Series:
    """
    Compute Stage 1 / Stage 2 / Stage 3 / Standalone for every Pokémon.

    Rules derived from the evolutions table:
      Standalone — appears in neither from nor to column
      Stage 1    — appears as from_pokemon only (base form that can evolve)
      Stage 2    — appears as both from and to (middle of 3-stage chain)
      Stage 2    — appears as to_pokemon only, whose parent is Stage 1 (end of 2-stage chain)
      Stage 3    — appears as to_pokemon only, whose parent is Stage 2 (end of 3-stage chain)
    """
    with engine.connect() as conn:
        evos = pd.read_sql(
            "SELECT from_pokemon_id, to_pokemon_id FROM evolutions", conn
        )
        all_ids = pd.read_sql("SELECT id FROM pokemon", conn)["id"].tolist()

    if evos.empty:
        return pd.Series("Standalone", index=all_ids, name="evo_stage")

    from_ids = set(evos["from_pokemon_id"])
    to_ids   = set(evos["to_pokemon_id"])

    def _stage(pid: int) -> str:
        is_from = pid in from_ids
        is_to   = pid in to_ids
        if not is_from and not is_to:
            return "Standalone"
        if is_from and not is_to:
            return "Stage 1"
        if is_from and is_to:
            return "Stage 2"
        # in to_ids only — check whether parent is a Stage 1 (= end of 2-stage chain)
        parents = evos.loc[evos["to_pokemon_id"] == pid, "from_pokemon_id"]
        if any(p in from_ids and p not in to_ids for p in parents):
            return "Stage 2"
        return "Stage 3"

    return pd.Series({pid: _stage(pid) for pid in all_ids}, name="evo_stage")


# Classification label priority: Baby > Legendary > Mythical > evo_stage
def _classify(row) -> str:
    if row["is_baby"]:
        return "Baby"
    if row["is_legendary"]:
        return "Legendary"
    if row["is_mythical"]:
        return "Mythical"
    return row["evo_stage"]


def load_pokemon_wide_df(engine=None) -> pd.DataFrame:
    """
    One row per pokemon.  Includes type_1 (primary) and type_2 (secondary, may be NaN).
    Use this for per-pokemon grouping and the explorer table.
    """
    df = load_pokemon_df(engine)

    primary   = df[df["type_slot"] == 1][["id", "type_name"]].rename(columns={"type_name": "type_1"})
    secondary = df[df["type_slot"] == 2][["id", "type_name"]].rename(columns={"type_name": "type_2"})

    keep = [
        "id", "name", "height", "weight", "base_experience", "sprite_url",
        "generation_id", "generation_name", "gen_label", "region",
    ] + STAT_COLS + ["total_stat"]

    base = (
        df[df["type_slot"] == 1][keep]
        .merge(primary,   on="id")
        .merge(secondary, on="id", how="left")
    )
    return base


def load_classification_df(engine=None) -> pd.DataFrame:
    """
    Returns a DataFrame with id, evo_stage, and classification for every Pokémon.
    Kept separate from load_pokemon_wide_df so it can be cached independently in the app,
    allowing the classification filter to update without busting the main data cache.
    """
    if engine is None:
        engine = get_engine()
    with engine.connect() as conn:
        species = pd.read_sql(
            "SELECT id, is_legendary, is_mythical, is_baby FROM pokemon", conn
        )
    evo_stages = _evo_stage_series(engine)
    species["evo_stage"] = species["id"].map(evo_stages)
    species["classification"] = species.apply(_classify, axis=1)
    return species[["id", "evo_stage", "classification"]]


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------

def stats_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average base stats grouped by type.
    Accepts the long df — dual-type pokemon count toward both types.
    Returns a DataFrame indexed by type_name, sorted by total_stat desc.
    """
    return (
        df.groupby("type_name")[STAT_COLS + ["total_stat"]]
        .mean()
        .round(1)
        .sort_values("total_stat", ascending=False)
    )


def stats_by_generation(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Average base stats per generation, sorted by generation number.
    Accepts the wide df to avoid counting dual-type pokemon twice.
    """
    return (
        df_wide.groupby(["generation_id", "generation_name", "gen_label"])[STAT_COLS + ["total_stat"]]
        .mean()
        .round(1)
        .reset_index()
        .sort_values("generation_id")
    )


def type_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Number of unique pokemon that have each type (counts both primary and secondary).
    Accepts the long df.
    """
    counts = (
        df.groupby("type_name")["id"]
        .nunique()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return counts


def top_n_by_stat(df_wide: pd.DataFrame, stat: str, n: int = 10) -> pd.DataFrame:
    """Top N pokemon for a given stat column."""
    cols = ["id", "name", "type_1", "type_2", "gen_label", stat]
    return df_wide.nlargest(n, stat)[cols].reset_index(drop=True)


def stat_correlation(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation matrix between the 6 base stats.
    Accepts the wide df so each Pokémon is counted exactly once.
    """
    return df_wide[STAT_COLS].corr().round(2)


def load_moves_for_pokemon(pokemon_id: int, engine=None) -> pd.DataFrame:
    """
    All moves a Pokémon can learn, with full move details.
    One row per (move, learn_method).  Sorted by method then level then name.
    """
    if engine is None:
        engine = get_engine()
    query = text("""
        SELECT
            m.name          AS move,
            t.name          AS type,
            m.damage_class  AS category,
            m.power,
            m.accuracy,
            m.pp,
            pm.learn_method,
            pm.level_learned
        FROM pokemon_moves pm
        JOIN moves m ON pm.move_id  = m.id
        JOIN types t ON m.type_id   = t.id
        WHERE pm.pokemon_id = :pid
        ORDER BY pm.learn_method, COALESCE(pm.level_learned, 999), m.name
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"pid": pokemon_id})


def load_abilities_for_pokemon(pokemon_id: int, engine=None) -> pd.DataFrame:
    """
    Abilities for a single Pokémon.
    Returns columns: name, effect, is_hidden.  Non-hidden abilities first.
    """
    if engine is None:
        engine = get_engine()
    query = text("""
        SELECT a.name, a.effect, pa.is_hidden
        FROM pokemon_abilities pa
        JOIN abilities a ON pa.ability_id = a.id
        WHERE pa.pokemon_id = :pid
        ORDER BY pa.is_hidden, a.name
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"pid": pokemon_id})


def load_evolution_chain_for_pokemon(pokemon_id: int, engine=None) -> pd.DataFrame:
    """
    All evolution edges that share a chain with the given Pokémon.
    Returns columns: from_id, from_name, from_sprite, to_id, to_name, to_sprite, trigger, min_level.
    Returns an empty DataFrame for Pokémon that don't evolve.
    """
    if engine is None:
        engine = get_engine()
    query = text("""
        SELECT
            p1.id         AS from_id,
            p1.name       AS from_name,
            p1.sprite_url AS from_sprite,
            p2.id         AS to_id,
            p2.name       AS to_name,
            p2.sprite_url AS to_sprite,
            e.trigger,
            e.min_level
        FROM evolutions e
        JOIN pokemon p1 ON e.from_pokemon_id = p1.id
        JOIN pokemon p2 ON e.to_pokemon_id   = p2.id
        WHERE e.chain_id = (
            SELECT chain_id FROM evolutions
            WHERE from_pokemon_id = :pid OR to_pokemon_id = :pid
            LIMIT 1
        )
        ORDER BY e.id
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"pid": pokemon_id})


def find_optimal_k(engine=None, k_max: int = 10) -> dict:
    """
    Compute KMeans inertia for k=2 through k_max.
    Returns {k: inertia} — use to plot an elbow curve and choose k.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    if engine is None:
        engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(f"SELECT {', '.join(STAT_COLS)} FROM base_stats", conn)
    X = StandardScaler().fit_transform(df[STAT_COLS])
    return {
        k: KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_
        for k in range(2, k_max + 1)
    }


def compute_stat_clusters(engine=None, k: int = 5) -> pd.DataFrame:
    """
    K-means cluster all Pokémon by their 6 base stats (StandardScaler normalized).
    Returns DataFrame with columns: id, name, hp, attack, defense,
    special_attack, special_defense, speed, cluster (int label 0..k-1).
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    if engine is None:
        engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            """
            SELECT p.id, p.name,
                   s.hp, s.attack, s.defense,
                   s.special_attack, s.special_defense, s.speed
            FROM pokemon p
            JOIN base_stats s ON p.id = s.pokemon_id
            """,
            conn,
        )
    X = StandardScaler().fit_transform(df[STAT_COLS])
    df["cluster"] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    return df
