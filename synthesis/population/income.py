import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

"""
This stage assigns a household income to each household of the synthesized
population. For that it looks up the municipality of each household in the
income database to obtain the municipality's income distribution (in centiles).
Then, for each household, a centile is selected randomly from the respective
income distribution and a random income within the selected stratum is chosen.
"""


def configure(context):
    context.stage("data.income.municipality")
    context.stage("synthesis.population.sampled")
    context.stage("synthesis.population.spatial.home.zones")
    context.stage("data.census.cleaned")

    context.config("random_seed")


MAXIMUM_INCOME_FACTOR = 1.2


def _sample_income(context, args):
    commune_id, random_seed = args
    df_households, df_income = context.data("households"), context.data("income")

    random = np.random.RandomState(random_seed)

    f = df_households["commune_id"] == commune_id
    df_selected = df_households[f]

    centiles = list(
        df_income[df_income["commune_id"] == commune_id][
            ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9"]
        ]
        .iloc[0]
        .values
        / 12
    )
    centiles = np.array([0] + centiles + [np.max(centiles) * MAXIMUM_INCOME_FACTOR])

    indices = random.randint(10, size=len(df_selected))
    lower_bounds, upper_bounds = centiles[indices], centiles[indices + 1]

    incomes = lower_bounds + random.random_sample(size=len(df_selected)) * (
        upper_bounds - lower_bounds
    )
    return f, incomes


def execute(context):
    random = np.random.RandomState(context.config("random_seed"))

    # Load data
    df_income = context.stage("data.income.municipality")
    df_population = context.stage("synthesis.population.sampled")

    df_households = df_population[["household_id", "consumption_units"]].drop_duplicates(
        "household_id"
    )

    df_homes = context.stage("synthesis.population.spatial.home.zones")[
        ["household_id", "commune_id"]
    ]

    df_households = pd.merge(df_households, df_homes)

    # Add household size
    df_households_size = df_population.groupby(["household_id"], as_index=False)[
        "person_id"
    ].count()
    df_households_size.rename(columns={"person_id": "household_size"}, inplace=True)
    df_households_size["household_size"] = df_households_size["household_size"].map(
        lambda x: str(x) + "_pers" if x < 5 else "5_pers_or_more"
    )
    df_households = pd.merge(df_households, df_households_size)

    # Add household type
    # compute single man or woman
    # household of 1 person
    df_households_info = df_population.groupby(["household_id"], as_index=False).agg(
        {"household_size": "first", "sex": "first"}
    )
    df_households_single_man = df_households_info.query(
        "household_size==1 and sex=='male'"
    ).reset_index()
    df_households_single_man["household_type"] = "Single_man"
    df_households_single_woman = df_households_info.query(
        "household_size==1 and sex=='female'"
    ).reset_index()
    df_households_single_woman["household_type"] = "Single_woman"

    # compute couple without child
    # 2 persons and all have couple=True
    df_households_couple = (
        df_population.query("couple==True")
        .groupby(["household_id"], as_index=False)
        .agg({"couple": "count", "household_size": "first"})
    )
    df_households_couple_without_child = df_households_couple.query(
        "couple==2 and household_size==2"
    ).reset_index()
    df_households_couple_without_child["household_type"] = "Couple_without_child"

    # compute couple with child
    # 3 persons or more, 2 persons in couple, and max age of person non couple < 25
    df_households_child = (
        df_population.query("age<25 and couple==False")
        .groupby(["household_id"], as_index=False)
        .agg({"person_id": "count"})
        .rename(columns={"person_id": "child_count"})
    )
    df_households_couple_with_child = (
        df_households_couple.merge(df_households_child, how="left", on="household_id")
        .query("household_size==child_count+2")
        .reset_index()
    )
    df_households_couple_with_child["household_type"] = "Couple_with_child"
    # compute single parent
    # 2 persons or more, no one in couple, oldest person = [25-60[, and others age < 25
    df_households_no_couple = df_population.query(
        "couple==False and household_size>=2"
    ).sort_values(["household_id", "age"], ascending=[True, False])

    df_households_no_couple_oldest = (
        df_households_no_couple.groupby(["household_id"], as_index=False)
        .agg({"age": "first", "person_id": "first"})
        .rename(columns={"age": "oldest_age", "person_id": "oldest_person_id"})
    )

    df_households_no_couple_second_oldest = (
        df_households_no_couple.merge(df_households_no_couple_oldest, how="left")
        .query("person_id!=oldest_person_id")
        .sort_values(["household_id", "age"], ascending=[True, False])
        .groupby(["household_id"], as_index=False)
        .agg({"age": "first"})
        .rename(columns={"age": "second_oldest_age"})
    )

    df_households_single_parent = df_households_no_couple_oldest.merge(
        df_households_no_couple_second_oldest, how="left", on="household_id"
    )
    df_households_single_parent = df_households_single_parent.query(
        "oldest_age>=25 and oldest_age<60 and second_oldest_age<25"
    ).reset_index()
    df_households_single_parent["household_type"] = "Single_parent"

    # combine type
    df_households = (
        df_households.merge(
            df_households_single_man[["household_id", "household_type"]],
            how="left",
            on="household_id",
            suffixes=("", "_1"),
        )
        .merge(
            df_households_single_woman[["household_id", "household_type"]],
            how="left",
            on="household_id",
            suffixes=("_1", "_2"),
        )
        .merge(
            df_households_couple_without_child[["household_id", "household_type"]],
            how="left",
            on="household_id",
            suffixes=("_2", "_3"),
        )
        .merge(
            df_households_couple_with_child[["household_id", "household_type"]],
            how="left",
            on="household_id",
            suffixes=("_3", "_4"),
        )
        .merge(
            df_households_single_parent[["household_id", "household_type"]],
            how="left",
            on="household_id",
            suffixes=("_5", "_5"),
        )
    )
    df_households.columns = [
        "household_id",
        "consumption_units",
        "commune_id",
        "household_size",
        "household_type_1",
        "household_type_2",
        "household_type_3",
        "household_type_4",
        "household_type_5",
    ]
    df_households[
        [
            "household_type_1",
            "household_type_2",
            "household_type_3",
            "household_type_4",
            "household_type_5",
        ]
    ] = df_households[
        [
            "household_type_1",
            "household_type_2",
            "household_type_3",
            "household_type_4",
            "household_type_5",
        ]
    ].fillna(
        ""
    )
    df_households["household_type"] = df_households.apply(
        lambda x: str(x["household_type_1"])
        + str(x["household_type_2"])
        + str(x["household_type_3"])
        + str(x["household_type_4"])
        + str(x["household_type_5"]),
        axis=1,
    )
    # compute complex households as other households
    df_households["household_type"] = df_households["household_type"].map(
        lambda x: "complex_hh" if x == "" else x
    )

    df_households = df_households[
        ["household_id", "consumption_units", "commune_id", "household_size", "household_type"]
    ]

    # Perform sampling per commune
    with context.parallel(dict(households=df_households, income=df_income)) as parallel:
        commune_ids = df_households["commune_id"].unique()
        random_seeds = random.randint(10000, size=len(commune_ids))

        for f, incomes in context.progress(
            parallel.imap(_sample_income, zip(commune_ids, random_seeds)),
            label="Imputing income ...",
            total=len(commune_ids),
        ):
            df_households.loc[f, "household_income"] = (
                incomes * df_households.loc[f, "consumption_units"]
            )

    # Cleanup
    df_households = df_households[["household_id", "household_income", "consumption_units"]]
    assert len(df_households) == len(df_households["household_id"].unique())
    return df_households
