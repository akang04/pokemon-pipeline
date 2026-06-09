import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from analyze import (
    GEN_LABEL,
    STAT_COLS,
    STAT_LABELS,
    TYPE_COLORS,
    load_abilities_for_pokemon,
    load_evolution_chain_for_pokemon,
    load_moves_for_pokemon,
    load_pokemon_df,
    load_pokemon_wide_df,
    stat_correlation,
    stats_by_generation,
    stats_by_type,
)
from database import get_engine

st.set_page_config(
    page_title="Pokémon Dashboard",
    page_icon="pokeball",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading — cached so the DB is only queried once per session
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    engine = get_engine()
    return load_pokemon_df(engine), load_pokemon_wide_df(engine)


@st.cache_data
def load_classifications():
    import pandas as pd
    engine = get_engine()
    with engine.connect() as conn:
        species = pd.read_sql("SELECT id, is_legendary, is_mythical, is_baby FROM pokemon", conn)
        evos    = pd.read_sql("SELECT from_pokemon_id, to_pokemon_id FROM evolutions", conn)

    from_ids = set(evos["from_pokemon_id"])
    to_ids   = set(evos["to_pokemon_id"])

    def _stage(pid):
        is_from, is_to = pid in from_ids, pid in to_ids
        if not is_from and not is_to:
            return "Standalone"
        if is_from and not is_to:
            return "Stage 1"
        if is_from and is_to:
            return "Stage 2"
        parents = evos.loc[evos["to_pokemon_id"] == pid, "from_pokemon_id"]
        return "Stage 2" if any(p in from_ids and p not in to_ids for p in parents) else "Stage 3"

    def _classify(row):
        if row["is_baby"]:      return "Baby"
        if row["is_legendary"]: return "Legendary"
        if row["is_mythical"]:  return "Mythical"
        return _stage(row["id"])

    species["evo_stage"]       = species["id"].apply(_stage)
    species["classification"]  = species.apply(_classify, axis=1)
    return species[["id", "evo_stage", "classification"]]


@st.cache_data
def get_abilities(pokemon_id: int):
    return load_abilities_for_pokemon(pokemon_id)


@st.cache_data
def get_evolution_chain(pokemon_id: int):
    return load_evolution_chain_for_pokemon(pokemon_id)


@st.cache_data
def get_moves(pokemon_id: int):
    return load_moves_for_pokemon(pokemon_id)


df_long, df_wide = load_data()
_class_df = load_classifications()
df_wide = df_wide.merge(_class_df, on="id", how="left")

if df_long.empty:
    st.warning("No data yet — run `python fetch.py` first.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

_CLASS_ORDER = ["Stage 1", "Stage 2", "Stage 3", "Standalone", "Baby", "Legendary", "Mythical"]

with st.sidebar:
    st.title("Filters")

    all_gens = sorted(df_wide["generation_name"].unique(), key=lambda g: df_wide.loc[df_wide["generation_name"] == g, "generation_id"].iloc[0])
    gen_options = [GEN_LABEL.get(g, g) for g in all_gens]
    selected_gen_labels = st.multiselect("Generation", gen_options, default=gen_options)
    selected_gens = [g for g in all_gens if GEN_LABEL.get(g, g) in selected_gen_labels]

    all_types = sorted(df_long["type_name"].unique())
    selected_types = st.multiselect("Type", all_types, default=all_types)

    available_classes = [c for c in _CLASS_ORDER if c in df_wide["classification"].unique()]
    selected_classes = st.multiselect("Classification", available_classes, default=available_classes)

    st.markdown("---")
    st.caption("Filters apply to all tabs.")

# Apply filters
filt_wide = df_wide[
    df_wide["generation_name"].isin(selected_gens)
    & df_wide["type_1"].isin(selected_types)
    & df_wide["classification"].isin(selected_classes)
]
filt_long = df_long[
    df_long["generation_name"].isin(selected_gens)
    & df_long["type_name"].isin(selected_types)
    & df_long["id"].isin(filt_wide["id"])
]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Pokémon Data Dashboard")
st.caption("Data sourced from PokéAPI · Phase 1: base stats, types, generations")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_by_type, tab_by_gen, tab_analysis, tab_explorer = st.tabs(
    ["Overview", "Stats by Type", "Stats by Generation", "Analysis", "Pokémon Explorer"]
)

# ── Overview ─────────────────────────────────────────────────────────────────

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pokémon", f"{filt_wide['id'].nunique():,}")
    c2.metric("Types",       filt_long["type_name"].nunique())
    c3.metric("Generations", filt_wide["generation_name"].nunique())
    c4.metric("Avg Total Stat", f"{filt_wide['total_stat'].mean():.0f}")

    st.markdown("### Pokémon count by type — primary vs secondary")

    primary_counts = filt_wide.groupby("type_1")["id"].nunique().rename("Primary")
    secondary_counts = (
        filt_wide[filt_wide["type_2"].notna()]
        .groupby("type_2")["id"].nunique()
        .rename("Secondary")
    )
    split = pd.concat([primary_counts, secondary_counts], axis=1).fillna(0).astype(int)
    split = split.sort_values("Primary", ascending=True)

    bar_colors = [TYPE_COLORS.get(t, "#888888") for t in split.index]
    y = np.arange(len(split))
    bar_h = 0.38

    fig, ax = plt.subplots(figsize=(7, max(4, len(split) * 0.55)))
    ax.barh(y + bar_h / 2, split["Primary"],   bar_h, color=bar_colors, edgecolor="white", label="Primary type")
    ax.barh(y - bar_h / 2, split["Secondary"],  bar_h, color=bar_colors, alpha=0.45, edgecolor="white", label="Secondary type")
    ax.set_yticks(y)
    ax.set_yticklabels(split.index)
    ax.set_xlabel("Number of Pokémon")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(loc="lower right", fontsize=8, framealpha=0.7)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Top bar = primary type · Bottom bar (faded) = secondary type. Flying appears almost exclusively as a secondary type.")

    with st.expander("Dual-type combination matrix"):
        st.caption(
            "Each cell shows how many Pokémon share that primary × secondary type combination. "
            "Blank cells = no Pokémon with that pairing exists in the current filter."
        )

        all_types_sorted = sorted(df_long["type_name"].unique())
        dual = filt_wide[filt_wide["type_2"].notna()][["type_1", "type_2", "id"]]

        if dual.empty:
            st.info("No dual-type Pokémon in the current filter.")
        else:
            combo_counts = (
                dual.groupby(["type_1", "type_2"])["id"]
                .nunique()
                .reset_index(name="n")
            )
            matrix = (
                combo_counts
                .pivot(index="type_1", columns="type_2", values="n")
                .reindex(index=all_types_sorted, columns=all_types_sorted, fill_value=0)
                .fillna(0)
                .astype(int)
            )

            diag_mask = np.eye(len(all_types_sorted), dtype=bool)
            annot = matrix.copy().astype(object)
            annot[matrix == 0] = ""

            fig, ax = plt.subplots(figsize=(11, 9))
            sns.heatmap(
                matrix,
                mask=diag_mask,
                annot=annot,
                fmt="",
                cmap="Blues",
                vmin=0,
                linewidths=0.4,
                linecolor="#eeeeee",
                cbar_kws={"shrink": 0.6, "label": "Pokémon count"},
                ax=ax,
            )
            ax.set_xlabel("Secondary Type", fontsize=9)
            ax.set_ylabel("Primary Type", fontsize=9)
            ax.tick_params(axis="x", labelsize=8, rotation=45)
            ax.tick_params(axis="y", labelsize=8, rotation=0)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Count missing combos (unordered — neither direction exists)
            existing_unordered = set()
            for _, row in combo_counts.iterrows():
                existing_unordered.add(frozenset([row["type_1"], row["type_2"]]))
            missing = [
                f"{t1} / {t2}"
                for i, t1 in enumerate(all_types_sorted)
                for t2 in all_types_sorted[i + 1:]
                if frozenset([t1, t2]) not in existing_unordered
            ]
            st.caption(
                f"{len(missing)} type combinations have no Pokémon at all "
                f"(in either order) for the current filter: "
                + ", ".join(missing) + "."
            )

# ── Stats by Type ─────────────────────────────────────────────────────────────

with tab_by_type:
    st.markdown("### Average base stats by type")

    sbt = stats_by_type(filt_long)

    if sbt.empty:
        st.info("No data for the current filter selection.")
    else:
        rank_stat = st.selectbox(
            "Rank by",
            options=STAT_COLS + ["total_stat"],
            index=6,
            format_func=lambda s: STAT_LABELS[s],
            key="type_rank_stat",
        )

        ranked = sbt[[rank_stat]].sort_values(rank_stat, ascending=True)
        bar_colors = [TYPE_COLORS.get(t, "#888888") for t in ranked.index]

        fig, ax = plt.subplots(figsize=(7, max(4, len(ranked) * 0.45)))
        bars = ax.barh(ranked.index, ranked[rank_stat], color=bar_colors, edgecolor="white")
        ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=8)
        ax.set_xlabel(f"Avg {STAT_LABELS[rank_stat]}")
        ax.set_xlim(0, ranked[rank_stat].max() * 1.15)
        ax.grid(axis="x", alpha=0.2)
        ax.tick_params(axis="y", labelsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption("Dual-type Pokémon count toward both types.")

        col_rename = {c: STAT_LABELS[c] for c in sbt.columns}
        sbt_display = sbt.rename(columns=col_rename)
        sbt_norm = (sbt_display - sbt_display.min()) / (sbt_display.max() - sbt_display.min()).replace(0, 1)

        with st.expander("All stats heatmap"):
            st.caption("Heatmap color shows relative rank within each stat column; numbers are actual averages.")
            fig2, ax2 = plt.subplots(figsize=(10, max(4, len(sbt_display) * 0.45)))
            sns.heatmap(
                sbt_norm,
                annot=sbt_display,
                fmt=".1f",
                cmap="YlOrRd",
                linewidths=0.4,
                linecolor="#eeeeee",
                cbar=False,
                ax=ax2,
            )
            ax2.set_ylabel("")
            ax2.set_xlabel("")
            ax2.tick_params(axis="x", labelsize=9)
            ax2.tick_params(axis="y", labelsize=9, rotation=0)
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        with st.expander("Show raw table"):
            st.dataframe(sbt_display, use_container_width=True)

# ── Stats by Generation ───────────────────────────────────────────────────────

with tab_by_gen:
    st.markdown("### Average base stats across generations")

    sbg = stats_by_generation(filt_wide)
    gen_counts = filt_wide.groupby("gen_label")["id"].nunique()
    sbg["pokemon_count"] = sbg["gen_label"].map(gen_counts)

    if sbg.empty:
        st.info("No data for the current filter selection.")
    else:
        ctrl_left, ctrl_right = st.columns([4, 1])
        with ctrl_left:
            stat_choice = st.multiselect(
                "Stats to display",
                options=STAT_COLS + ["total_stat"],
                default=["total_stat"],
                format_func=lambda s: STAT_LABELS[s],
            )
        with ctrl_right:
            show_all = st.checkbox("All 6 stats", value=False)

        if show_all:
            stat_choice = STAT_COLS

        if stat_choice:
            fig, ax = plt.subplots(figsize=(9, 4))

            # Secondary axis: muted count bars for context
            ax2 = ax.twinx()
            ax2.bar(
                sbg["gen_label"],
                sbg["pokemon_count"],
                color="#bbbbbb",
                alpha=0.25,
                width=0.6,
                zorder=0,
            )
            for _, row in sbg.iterrows():
                ax2.text(
                    row["gen_label"],
                    row["pokemon_count"],
                    f"n={int(row['pokemon_count'])}",
                    ha="center", va="bottom",
                    fontsize=6.5, color="#999999",
                )
            ax2.set_ylim(0, sbg["pokemon_count"].max() * 4)
            ax2.set_ylabel("Pokémon count", fontsize=8, color="#aaaaaa")
            ax2.tick_params(axis="y", labelsize=7, colors="#bbbbbb")

            # Primary axis: stat lines — drawn on top of bars
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)

            for stat in stat_choice:
                ax.plot(
                    sbg["gen_label"],
                    sbg[stat],
                    marker="o",
                    label=STAT_LABELS[stat],
                    linewidth=2,
                    zorder=2,
                )

            # Annotate each point only when a single stat is selected
            if len(stat_choice) == 1:
                stat = stat_choice[0]
                for _, row in sbg.iterrows():
                    ax.annotate(
                        f"{row[stat]:.0f}",
                        xy=(row["gen_label"], row[stat]),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=7.5, color="#444444",
                    )

            ax.set_xlabel("Generation")
            ax.set_ylabel("Average Base Stat")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with st.expander("Show raw table"):
            display = sbg.drop(columns=["generation_id", "generation_name"])
            display = display.rename(columns={
                "gen_label": "Generation",
                "pokemon_count": "Count",
                **{c: STAT_LABELS[c] for c in STAT_COLS + ["total_stat"]},
            })
            st.dataframe(display.set_index("Generation"), use_container_width=True)

# ── Analysis ──────────────────────────────────────────────────────────────────

with tab_analysis:
    st.markdown("### Base Stat Correlation")
    st.caption(
        "Pearson r measures linear association between two stats across all Pokémon in the current filter. "
        "r = 1 means they rise together perfectly; r = 0 means they're independent; r = −1 means one rises as the other falls."
    )

    if filt_wide.empty:
        st.info("No data for the current filter selection.")
    else:
        corr = stat_correlation(filt_wide)
        n_pokemon = len(filt_wide)

        # Rename axes for display
        label_order = [STAT_LABELS[c] for c in STAT_COLS]
        corr_display = corr.copy()
        corr_display.index   = label_order
        corr_display.columns = label_order

        # Mask upper triangle — lower triangle + diagonal only
        mask = np.zeros_like(corr_display, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True

        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.heatmap(
            corr_display,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1, vmax=1, center=0,
            linewidths=0.5,
            linecolor="#eeeeee",
            cbar_kws={"shrink": 0.75, "label": "Pearson r"},
            ax=ax,
        )
        ax.set_title(f"Base Stat Correlations  (n = {n_pokemon:,} Pokémon)", fontsize=11, pad=10)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9, rotation=0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Key findings — computed dynamically from the filtered data
        pairs = [
            (corr.loc[r, c], STAT_LABELS[r], STAT_LABELS[c])
            for i, r in enumerate(STAT_COLS)
            for j, c in enumerate(STAT_COLS)
            if i > j
        ]
        pairs.sort()
        strongest  = pairs[-1]
        most_indep = pairs[0]
        avg_r      = sum(p[0] for p in pairs) / len(pairs)

        st.markdown("#### Key Findings")
        k1, k2, k3 = st.columns(3)
        k1.metric(
            "Strongest link",
            f"{strongest[1]} & {strongest[2]}",
            f"r = {strongest[0]:.2f}",
        )
        k2.metric(
            "Most independent",
            f"{most_indep[1]} & {most_indep[2]}",
            f"r = {most_indep[0]:.2f}",
        )
        k3.metric(
            "Avg off-diagonal r",
            f"{avg_r:.2f}",
            help="Average Pearson r across all 15 stat pairs. "
                 "Higher means stats tend to rise together (generalist Pokémon); "
                 "lower means stats are more specialised.",
        )

        with st.expander("How to read this"):
            st.markdown(
                "Each cell shows the Pearson correlation coefficient between two base stats "
                "computed across every Pokémon in the current filter. "
                "The diagonal is always 1.0 (a stat is perfectly correlated with itself). "
                "The upper triangle is hidden to avoid redundancy — every pair appears once. "
                "\n\n"
                "Use the sidebar filters to narrow to a type or generation and see how the "
                "correlation structure changes — for example, Dragon-types show much tighter "
                "stat clustering than Bug-types."
            )

    st.markdown("### Physical Dimensions vs Stats")
    st.caption("Pearson r between a Pokémon's weight/height and its base stats.")

    dims = filt_wide[["weight", "height"] + STAT_COLS + ["total_stat"]].dropna(subset=["weight", "height"])
    if dims.empty:
        st.info("No data for the current filter selection.")
    else:
        def _r(x, y):
            return float(np.corrcoef(x, y)[0, 1])

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Weight × Defense",    f"r = {_r(dims['weight'], dims['defense']):.2f}")
        d2.metric("Weight × HP",         f"r = {_r(dims['weight'], dims['hp']):.2f}")
        d3.metric("Weight × Total",      f"r = {_r(dims['weight'], dims['total_stat']):.2f}")
        d4.metric("Height × Total",      f"r = {_r(dims['height'], dims['total_stat']):.2f}")
        st.caption(
            "Heavier Pokémon tend to be bulkier and stronger overall, but the relationship is moderate — "
            "weight is not a reliable proxy for power."
        )

    st.markdown("### Legendary / Mythical vs. Standard Pokémon")
    st.caption(
        "Generation and Type filters apply. The Classification filter controls which Standard "
        "Pokémon appear in the comparison — Legendary and Mythical are always the fixed group."
    )

    # Gen + Type filtered; Classification filter controls the Standard group only
    legend_df = df_wide[
        df_wide["generation_name"].isin(selected_gens)
        & df_wide["type_1"].isin(selected_types)
    ]

    if legend_df.empty:
        st.info("No data for the current filter selection.")
    else:
        is_leg = legend_df["classification"].isin(["Legendary", "Mythical"])
        leg = legend_df[is_leg]

        _non_leg_classes = [c for c in _CLASS_ORDER if c not in ("Legendary", "Mythical")]
        std_classes = [c for c in selected_classes if c not in ("Legendary", "Mythical")]
        std = legend_df[legend_df["classification"].isin(std_classes)]

        if set(std_classes) == set(_non_leg_classes):
            std_label = "Standard"
        elif len(std_classes) == 1:
            std_label = std_classes[0]
        else:
            std_label = " + ".join(std_classes)

        if leg.empty or std.empty:
            st.info("Select at least one non-Legendary classification to compare against.")
        else:
            stat_display = STAT_COLS + ["total_stat"]
            leg_avg = leg[stat_display].mean()
            std_avg = std[stat_display].mean()
            labels = [STAT_LABELS[s] for s in stat_display]

            x = np.arange(len(stat_display))
            bar_w = 0.35

            fig, ax = plt.subplots(figsize=(9, 4))
            bars1 = ax.bar(x - bar_w / 2, [leg_avg[s] for s in stat_display], bar_w,
                           label=f"Legendary / Mythical  (n={len(leg):,})", color="#9966CC")
            bars2 = ax.bar(x + bar_w / 2, [std_avg[s] for s in stat_display], bar_w,
                           label=f"{std_label}  (n={len(std):,})", color="#6688AA")
            ax.bar_label(bars1, fmt="%.0f", padding=3, fontsize=7.5, color="#444")
            ax.bar_label(bars2, fmt="%.0f", padding=3, fontsize=7.5, color="#444")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel("Average Base Stat")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            gap_total = leg_avg["total_stat"] - std_avg["total_stat"]
            biggest_gap_stat = max(STAT_COLS, key=lambda s: leg_avg[s] - std_avg[s])
            biggest_gap_val  = leg_avg[biggest_gap_stat] - std_avg[biggest_gap_stat]

            g1, g2, g3 = st.columns(3)
            g1.metric("Total Stat Gap", f"+{gap_total:.0f}",
                      help="Legendary/Mythical avg total minus comparison group avg total")
            g2.metric("Widest Per-Stat Gap", STAT_LABELS[biggest_gap_stat],
                      f"+{biggest_gap_val:.0f}")
            g3.metric("Groups compared", f"{len(leg):,} vs {len(std):,}")

# ── Pokémon Explorer ──────────────────────────────────────────────────────────

with tab_explorer:
    st.markdown("### Pokémon Explorer")

    col_search, col_sort = st.columns([3, 1])
    with col_search:
        search = st.text_input("Search by name", placeholder="e.g. char, bulb, eevee")
    with col_sort:
        sort_stat = st.selectbox("Sort by", options=STAT_COLS + ["total_stat"], index=6, format_func=lambda s: STAT_LABELS[s])

    explorer_df = filt_wide.copy()
    if search:
        explorer_df = explorer_df[explorer_df["name"].str.contains(search.strip(), case=False, na=False)]

    explorer_df = explorer_df.sort_values(sort_stat, ascending=False)

    display_cols = ["id", "name", "gen_label", "type_1", "type_2"] + STAT_COLS + ["total_stat"]
    col_rename = {
        "id": "#",
        "name": "Name",
        "gen_label": "Generation",
        "type_1": "Type 1",
        "type_2": "Type 2",
        **{c: STAT_LABELS[c] for c in STAT_COLS + ["total_stat"]},
    }

    st.dataframe(
        explorer_df[display_cols].rename(columns=col_rename).reset_index(drop=True),
        use_container_width=True,
        height=420,
    )

    st.caption(f"{len(explorer_df):,} Pokémon shown")

    # ── Pokémon Profile ──
    st.markdown("### Pokémon Profile")
    pokemon_names = explorer_df["name"].tolist()

    if pokemon_names:
        selected_name = st.selectbox("Select a Pokémon", pokemon_names)
        row = explorer_df[explorer_df["name"] == selected_name].iloc[0]
        pokemon_id = int(row["id"])

        # Header: sprite + key metrics
        col_sprite, col_meta = st.columns([1, 4])
        with col_sprite:
            if pd.notna(row.get("sprite_url")):
                st.image(row["sprite_url"], width=120)
        with col_meta:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Pokédex #", pokemon_id)
            m2.metric("Generation", row["gen_label"])
            m3.metric("Type 1", row["type_1"].title())
            m4.metric("Type 2", str(row["type_2"]).title() if pd.notna(row.get("type_2")) else "—")

        # Stat bar chart
        stat_vals = [row[s] for s in STAT_COLS]
        stat_labels_display = [STAT_LABELS[s] for s in STAT_COLS]

        fig, ax = plt.subplots(figsize=(6, 2.5))
        bar_color = TYPE_COLORS.get(row["type_1"], "#888888")
        bars = ax.barh(stat_labels_display[::-1], stat_vals[::-1], color=bar_color, edgecolor="white")
        ax.set_xlim(0, 255)
        ax.set_xlabel("Base Stat")
        ax.bar_label(bars, fmt="%d", padding=4, fontsize=9)
        ax.grid(axis="x", alpha=0.2)
        ax.set_title(f"{selected_name.title()} — Total: {int(row['total_stat'])}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Abilities + Evolution chain side by side
        col_ab, col_evo = st.columns(2)

        with col_ab:
            st.markdown("#### Abilities")
            abilities_df = get_abilities(pokemon_id)
            if abilities_df.empty:
                st.caption("No ability data.")
            else:
                for _, ab in abilities_df.iterrows():
                    label = ab["name"].replace("-", " ").title()
                    if ab["is_hidden"]:
                        label += " *(Hidden Ability)*"
                    st.markdown(f"**{label}**")
                    if pd.notna(ab["effect"]) and ab["effect"]:
                        st.caption(ab["effect"])

        with col_evo:
            st.markdown("#### Evolution Chain")
            evo_df = get_evolution_chain(pokemon_id)
            if evo_df.empty:
                st.caption("Does not evolve.")
            else:
                for _, evo in evo_df.iterrows():
                    e1, e2, e3, e4 = st.columns([2, 0.4, 2, 3])
                    with e1:
                        if pd.notna(evo["from_sprite"]):
                            st.image(evo["from_sprite"], width=64)
                        st.caption(evo["from_name"].title())
                    with e2:
                        st.markdown("**→**")
                    with e3:
                        if pd.notna(evo["to_sprite"]):
                            st.image(evo["to_sprite"], width=64)
                        st.caption(evo["to_name"].title())
                    with e4:
                        trigger = evo["trigger"].replace("-", " ").title() if pd.notna(evo["trigger"]) else ""
                        if pd.notna(evo["min_level"]) and evo["min_level"]:
                            trigger += f" (Lv. {int(evo['min_level'])})"
                        st.markdown(f"*{trigger}*")

        # Moves
        st.markdown("#### Moves")
        moves_df = get_moves(pokemon_id)

        if moves_df.empty:
            st.caption("No move data yet — fetch still in progress.")
        else:
            METHOD_LABELS = {
                "level-up": "Level Up",
                "machine":  "TM / HM",
                "egg":      "Egg Moves",
                "tutor":    "Tutor",
            }
            METHOD_ORDER = ["level-up", "machine", "egg", "tutor"]

            for method in METHOD_ORDER:
                group = moves_df[moves_df["learn_method"] == method].copy()
                if group.empty:
                    continue
                label = METHOD_LABELS.get(method, method.replace("-", " ").title())
                with st.expander(f"{label} ({len(group)})"):
                    group["move"]     = group["move"].str.replace("-", " ").str.title()
                    group["type"]     = group["type"].str.title()
                    group["category"] = group["category"].str.title()

                    if method == "level-up":
                        group = group.rename(columns={
                            "level_learned": "Lv.",
                            "move": "Move", "type": "Type",
                            "category": "Category",
                            "power": "Power", "accuracy": "Acc.", "pp": "PP",
                        })
                        show_cols = ["Lv.", "Move", "Type", "Category", "Power", "Acc.", "PP"]
                    else:
                        group = group.rename(columns={
                            "move": "Move", "type": "Type",
                            "category": "Category",
                            "power": "Power", "accuracy": "Acc.", "pp": "PP",
                        })
                        show_cols = ["Move", "Type", "Category", "Power", "Acc.", "PP"]

                    st.dataframe(
                        group[show_cols].reset_index(drop=True),
                        use_container_width=True,
                        hide_index=True,
                    )
    else:
        st.info("No Pokémon match the current search/filter.")
