from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    ForeignKey,
    Integer,
    String,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, relationship

DB_PATH = Path(__file__).parent / "data" / "pokemon.db"


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

class Generation(Base):
    __tablename__ = "generations"

    id = Column(Integer, primary_key=True)        # matches PokéAPI generation id
    name = Column(String, nullable=False, unique=True)  # e.g. "generation-i"
    region = Column(String)                        # e.g. "kanto"

    pokemon = relationship("Pokemon", back_populates="generation")


class Type(Base):
    __tablename__ = "types"

    id = Column(Integer, primary_key=True)         # matches PokéAPI type id
    name = Column(String, nullable=False, unique=True)  # e.g. "fire"

    pokemon_types = relationship("PokemonType", back_populates="type")
    moves = relationship("Move", back_populates="type")


# ---------------------------------------------------------------------------
# Core Pokémon table
# ---------------------------------------------------------------------------

class Pokemon(Base):
    __tablename__ = "pokemon"

    id = Column(Integer, primary_key=True)         # Pokédex number
    name = Column(String, nullable=False)
    generation_id = Column(Integer, ForeignKey("generations.id"))
    height = Column(Integer)                        # decimetres
    weight = Column(Integer)                        # hectograms
    base_experience = Column(Integer)
    sprite_url = Column(String)                     # front-default sprite
    is_legendary = Column(Boolean, default=False)
    is_mythical  = Column(Boolean, default=False)
    is_baby      = Column(Boolean, default=False)

    generation = relationship("Generation", back_populates="pokemon")
    types = relationship("PokemonType", back_populates="pokemon", cascade="all, delete-orphan")
    stats = relationship("BaseStat", back_populates="pokemon", uselist=False, cascade="all, delete-orphan")

    # Phase 2 relationships — wired up now so adding data later needs no schema change
    abilities = relationship("PokemonAbility", back_populates="pokemon", cascade="all, delete-orphan")
    moves = relationship("PokemonMove", back_populates="pokemon", cascade="all, delete-orphan")
    evolutions_from = relationship("Evolution", foreign_keys="Evolution.from_pokemon_id", back_populates="from_pokemon")
    evolutions_to = relationship("Evolution", foreign_keys="Evolution.to_pokemon_id", back_populates="to_pokemon")


# ---------------------------------------------------------------------------
# Phase 1: types and stats (populated by fetch.py)
# ---------------------------------------------------------------------------

class PokemonType(Base):
    __tablename__ = "pokemon_types"

    pokemon_id = Column(Integer, ForeignKey("pokemon.id"), primary_key=True)
    type_id = Column(Integer, ForeignKey("types.id"), primary_key=True)
    slot = Column(Integer, nullable=False)          # 1 = primary, 2 = secondary

    pokemon = relationship("Pokemon", back_populates="types")
    type = relationship("Type", back_populates="pokemon_types")


class BaseStat(Base):
    __tablename__ = "base_stats"

    pokemon_id = Column(Integer, ForeignKey("pokemon.id"), primary_key=True)
    hp = Column(Integer)
    attack = Column(Integer)
    defense = Column(Integer)
    special_attack = Column(Integer)
    special_defense = Column(Integer)
    speed = Column(Integer)

    pokemon = relationship("Pokemon", back_populates="stats")


# ---------------------------------------------------------------------------
# Phase 2: abilities (schema ready, not populated in Phase 1)
# ---------------------------------------------------------------------------

class Ability(Base):
    __tablename__ = "abilities"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    effect = Column(String)

    pokemon_abilities = relationship("PokemonAbility", back_populates="ability")


class PokemonAbility(Base):
    __tablename__ = "pokemon_abilities"

    pokemon_id = Column(Integer, ForeignKey("pokemon.id"), primary_key=True)
    ability_id = Column(Integer, ForeignKey("abilities.id"), primary_key=True)
    is_hidden = Column(Boolean, default=False)

    pokemon = relationship("Pokemon", back_populates="abilities")
    ability = relationship("Ability", back_populates="pokemon_abilities")


# ---------------------------------------------------------------------------
# Phase 2: moves (schema ready, not populated in Phase 1)
# ---------------------------------------------------------------------------

class Move(Base):
    __tablename__ = "moves"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    type_id = Column(Integer, ForeignKey("types.id"))
    power = Column(Integer)
    accuracy = Column(Integer)
    pp = Column(Integer)
    damage_class = Column(String)                   # "physical", "special", "status"

    type = relationship("Type", back_populates="moves")
    pokemon_moves = relationship("PokemonMove", back_populates="move")


class PokemonMove(Base):
    __tablename__ = "pokemon_moves"

    pokemon_id = Column(Integer, ForeignKey("pokemon.id"), primary_key=True)
    move_id = Column(Integer, ForeignKey("moves.id"), primary_key=True)
    learn_method = Column(String, primary_key=True)  # "level-up", "machine", "egg", "tutor"
    level_learned = Column(Integer)

    pokemon = relationship("Pokemon", back_populates="moves")
    move = relationship("Move", back_populates="pokemon_moves")


# ---------------------------------------------------------------------------
# Phase 2: evolutions (schema ready, not populated in Phase 1)
# ---------------------------------------------------------------------------

class EvolutionChain(Base):
    __tablename__ = "evolution_chains"

    id = Column(Integer, primary_key=True)

    evolutions = relationship("Evolution", back_populates="chain")


class Evolution(Base):
    __tablename__ = "evolutions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chain_id = Column(Integer, ForeignKey("evolution_chains.id"))
    from_pokemon_id = Column(Integer, ForeignKey("pokemon.id"))
    to_pokemon_id = Column(Integer, ForeignKey("pokemon.id"))
    trigger = Column(String)                        # "level-up", "trade", "use-item", etc.
    min_level = Column(Integer)

    chain = relationship("EvolutionChain", back_populates="evolutions")
    from_pokemon = relationship("Pokemon", foreign_keys=[from_pokemon_id], back_populates="evolutions_from")
    to_pokemon = relationship("Pokemon", foreign_keys=[to_pokemon_id], back_populates="evolutions_to")


# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------

def run_migrations(engine=None):
    """Add new columns to existing tables without dropping data. Safe to call repeatedly."""
    if engine is None:
        engine = get_engine()
    new_cols = [
        ("pokemon", "is_legendary", "INTEGER DEFAULT 0"),
        ("pokemon", "is_mythical",  "INTEGER DEFAULT 0"),
        ("pokemon", "is_baby",      "INTEGER DEFAULT 0"),
    ]
    with engine.connect() as conn:
        for table, col, typedef in new_cols:
            existing = [r[1] for r in conn.execute(text(f"PRAGMA table_info({table})")).fetchall()]
            if col not in existing:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {typedef}"))
        conn.commit()


def get_engine(db_path: Path = DB_PATH):
    return create_engine(f"sqlite:///{db_path}", echo=False)


def init_db(engine=None):
    """Create all tables. Safe to call on an existing database."""
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    return engine