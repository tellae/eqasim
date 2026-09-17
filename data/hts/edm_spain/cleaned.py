import pandas as pd
import numpy as np
import data.hts.hts as hts

"""
This stage cleans the the Spanish HTS (EMP).
"""

def configure(context):
    context.stage("data.hts.edm_spain.raw")

INCOME_CLASS_BOUNDS = [400, 800, 1000, 1200, 1500, 1800, 2000, 2500, 4000, 10000, 1e6]

PURPOSE_MAP = {
    "home": [1],
    "work": [2, 3],
    "education": [4],
    "shop": [5],
    "leisure": [8,9],
    "escort": [7],
    "task": [6,10,11],
    "other": [12],
}

MODES_MAP = {
    "car": [11,12,13,17,18,19],
    "car_passenger": [14,15,16],
    "pt": [1,2,3,4,5,6,7,8,9,10,23],
    "bike": [20,21,22],
    "walk": [24] 
}


def convert_time(x):
    return np.dot(np.array(x.split(":"), dtype = float), [3600.0, 60.0, 1.0])

def execute(context):
    df_households, df_persons, df_trips,df_spatial = context.stage("data.hts.edm_spain.raw")

    # Make copies
    df_households = pd.DataFrame(df_households, copy = True)
    df_persons = pd.DataFrame(df_persons, copy = True)
    df_trips = pd.DataFrame(df_trips, copy = True)

    #Merge commune into trips 
    df_trips = pd.merge(df_trips,df_spatial[["ZT1259","id"]].rename(columns={"ZT1259":"VORIZT1259"}),how="left",on="VORIZT1259")
    df_trips["origin_commune_id"] = df_trips["id"].str.split('_').str[1]
    df_trips = pd.merge(df_trips.drop(columns=["id"]),df_spatial[["ZT1259","id"]].rename(columns={"ZT1259":"VDESZT1259"}),how="left",on="VDESZT1259")
    df_trips["destination_commune_id"] = df_trips["id"].str.split('_').str[1]
    # Merge departement into households
    df_households["departement_id"] = "28"
    
    # Transform original IDs to integer (they are hierarchichal)
    df_households["edm_household_id"] = df_households["ID_HOGAR"].astype(int)
    df_persons["edm_person_id"] = df_persons["ID_IND"].astype(int)
    df_persons["edm_household_id"] = df_persons["ID_HOGAR"].astype(int)
    df_trips["edm_person_id"] = df_trips["ID_IND"].astype(int)
    df_trips["edm_household_id"] = df_trips["ID_HOGAR"].astype(int)
    df_trips["edm_trip_id"] = df_trips["ID_VIAJE"].astype(int)

    # Construct new IDs for households, persons and trips (which are unique globally)
    df_households["household_id"] = np.arange(len(df_households))

    df_persons = pd.merge(
        df_persons, df_households[["edm_household_id", "household_id","departement_id"]],
        on = "edm_household_id"
    )
    df_persons["person_id"] = np.arange(len(df_persons))

    df_trips = pd.merge(
        df_trips, df_persons[["edm_person_id", "edm_household_id", "person_id", "household_id"]],
        on = ["edm_person_id", "edm_household_id"]
    )
    df_trips["trip_id"] = np.arange(len(df_trips))

    # Trip flags
    df_trips = hts.compute_first_last(df_trips)

    # Weight
    df_persons["person_weight"] = df_persons["ELE_G_POND"].astype(float)
    df_households["household_weight"] = df_households["ELE_HOGAR_NUEVO"].astype(float)

    # Clean age
    df_persons["age"] = df_persons["EDAD_FIN"].astype(int)

    # Clean sex
    df_persons.loc[df_persons["C2SEXO"] == 1, "sex"] = "male"
    df_persons.loc[df_persons["C2SEXO"] == 2, "sex"] = "female"
    df_persons["sex"] = df_persons["sex"].astype("category")

    # Household size
    df_size = df_persons.groupby("household_id").size().reset_index(name = "household_size")
    df_households = pd.merge(df_households, df_size, on = "household_id")

    # Clean departement
    df_trips["origin_departement_id"] = "28"
    df_trips["destination_departement_id"] = "28"
    # Clean departement
    df_persons["departement_id"] = df_persons["departement_id"].astype("category")
    df_households["departement_id"] = df_households["departement_id"].astype("category")
    df_trips["origin_commune_id"] = df_trips["origin_commune_id"].astype("category")
    df_trips["destination_commune_id"] = df_trips["destination_commune_id"].astype("category")

    # Clean employment
    df_persons["employed"] = df_persons["C8ACTIV"].isin([1, 2])

    # Studies
    df_persons["studies"] = df_persons["C8ACTIV"].isin([2, 6])

    # Number of vehicles
    df_households["number_of_vehicles"] = df_households["B1NVE"].astype(int)
    df_households["number_of_bikes"] = 0

    # License
    df_persons["has_license"] = df_persons["C6CARNE"] > 2

    # Has subscription
    df_persons["has_pt_subscription"] = df_persons["C14ABONO"] == 1

    # Trip purpose
    df_trips["following_purpose"] = "invalid"
    df_trips["preceding_purpose"] = "invalid"

    for purpose, values in PURPOSE_MAP.items():
        df_trips.loc[df_trips["VORI"].isin(values), "following_purpose"] = purpose
        df_trips.loc[df_trips["VDES"].isin(values), "preceding_purpose"] = purpose

    assert np.count_nonzero(df_trips["following_purpose"] == "invalid") == 0
    assert np.count_nonzero(df_trips["preceding_purpose"] == "invalid") == 0

    # Trip mode
    for mode, values in MODES_MAP.items():
            df_trips.loc[df_trips["MODO_PRIORITARIO"].isin(values), "mode"] = mode

    df_trips["mode"] = df_trips["mode"].astype("category")

    # Further trip attributes
    df_trips["euclidean_distance"] = df_trips["DISTANCIA_VIAJE"] * 1000.0

    # Trip times
    df_trips["departure_time"] = 3600.0 * (df_trips["VORIHORAINI"] // 100) # hour
    df_trips["departure_time"] += 60.0 * (df_trips["VORIHORAINI"] % 100) # minute

    df_trips["arrival_time"] = 3600.0 * (df_trips["VDESHORAFIN"] // 100) # hour
    df_trips["arrival_time"] += 60.0 * (df_trips["VDESHORAFIN"] % 100) # minute

    df_trips = df_trips.sort_values(by = ["household_id", "person_id", "trip_id"])
    df_trips = hts.fix_trip_times(df_trips)

    # Durations
    df_trips["trip_duration"] = df_trips["arrival_time"] - df_trips["departure_time"]
    hts.compute_activity_duration(df_trips)

    # Add weight to trips
    df_trips = pd.merge(
        df_trips, df_persons[["person_id", "person_weight"]], on = "person_id", how = "left"
    ).rename(columns = { "person_weight": "trip_weight" })
    df_persons["trip_weight"] = df_persons["person_weight"]

    # Chain length
    df_count = df_trips[["person_id"]].groupby("person_id").size().reset_index(name = "number_of_trips")

    # People with at least one trip (number_of_trips > 0)
    df_persons = pd.merge(df_persons, df_count, on = "person_id", how = "left")

    # Nonrespondent of travel questionary section (number_of_trips = -1)
    df_persons["number_of_trips"] = df_persons["number_of_trips"].fillna(-1).astype(int)

    # Passenger attribute
    df_persons["is_passenger"] = df_persons["person_id"].isin(
        df_trips[df_trips["mode"] == "car_passenger"]["person_id"].unique()
    )

    # Calculate consumption units
    hts.check_household_size(df_households, df_persons)
    df_households = pd.merge(df_households, hts.calculate_consumption_units(df_persons), on = "household_id")

    # Socioprofessional class  
    df_persons["socioprofessional_class"] = df_persons["C10SECTOR"].fillna(9).astype(int) -1
    df_persons.loc[df_persons["C8ACTIV"] == 3, "socioprofessional_class"] = 7

    # Drop people that have NaN departure or arrival times in trips
    # Filter for people with NaN departure or arrival times in trips
    f = df_trips["departure_time"].isna()
    f |= df_trips["arrival_time"].isna()

    f = df_persons["person_id"].isin(df_trips[f]["person_id"])

    nan_count = np.count_nonzero(f)
    total_count = len(df_persons)

    print("Dropping %d/%d persons because of NaN values in departure and arrival times" % (nan_count, total_count))

    df_persons = df_persons[~f]
    df_trips = df_trips[df_trips["person_id"].isin(df_persons["person_id"].unique())]
    df_households = df_households[df_households["household_id"].isin(df_persons["household_id"])]

    # Fix activity types (because of inconsistent EGT data and removing in the timing fixing step)
    hts.fix_activity_types(df_trips)

    return df_households, df_persons, df_trips

def calculate_income_class(df):
    assert "household_income" in df
    assert "consumption_units" in df

    return np.digitize(df["household_income"] / df["consumption_units"], INCOME_CLASS_BOUNDS, right = True)
