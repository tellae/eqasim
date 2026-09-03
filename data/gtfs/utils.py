import zipfile, io
import pandas as pd
import geopandas as gpd
import shapely.geometry as geo
import os
import numpy as np
import datetime

REQUIRED_SLOTS = [
    "agency", "stops", "routes", "trips", "stop_times"
]

OPTIONAL_SLOTS = [
    "calendar", "calendar_dates", "fare_attributes", "fare_rules",
    "shapes", "frequencies", "transfers", "pathways", "levels",
    "feed_info", "translations", "attributions"
]

DTYPES = {
    "stops": {
        "stop_id": str, "parent_station": str
    },
    "agency": {
        "agency_id": str
    },
    "routes": {
        "agency_id": str
    }
}

def read_feed(path):
    feed = {}

    with zipfile.ZipFile(path, "r") as zip:
        available_slots = zip.namelist()
        prefix = None

        if "agency.txt" in available_slots:
            prefix = ""
        else:
            for slot in available_slots:
                if slot.endswith("agency.txt"):
                    prefix = slot.replace("agency.txt", "")
                    print("Warning: GTFS files seem to be located in: %s" % prefix)
                    break

            if prefix is None:
                raise RuntimeError("No GTFS data found in archive")

        for slot in REQUIRED_SLOTS:
            if not "%s%s.txt" % (prefix, slot) in available_slots:
                raise RuntimeError("Missing GTFS information: %s" % slot)

        if not "%scalendar.txt" % prefix in available_slots and not "%scalendar_dates.txt" % prefix in available_slots:
            raise RuntimeError("At least calendar.txt or calendar_dates.txt must be specified.")

        print("Loading GTFS data from %s ..." % path)

        for slot in REQUIRED_SLOTS + OPTIONAL_SLOTS:
            if "%s%s.txt" % (prefix, slot) in available_slots:
                print("  Loading %s.txt ..." % slot)

                with zip.open("%s%s.txt" % (prefix, slot)) as f:
                    feed[slot] = pd.read_csv(f, skipinitialspace = True, dtype = DTYPES.get(slot, None))
            else:
                print("  Not loading %s.txt" % slot)

    # Some cleanup

    for slot in ("calendar", "calendar_dates", "trips"):
        if slot in feed and "service_id" in feed[slot] and pd.api.types.is_string_dtype(feed[slot]["service_id"]):
            initial_count = len(feed[slot])
            feed[slot] = feed[slot][feed[slot]["service_id"].str.len() > 0]
            final_count = len(feed[slot])

            if final_count != initial_count:
                print("WARNING Removed %d/%d entries from %s with empty service_id" % (
                    initial_count - final_count, initial_count, slot
                ))

    if "stops" in feed:
        df_stops = feed["stops"]

        if not "parent_station" in df_stops:
            print("WARNING Missing parent_station in stops, setting to empty string")
            df_stops["parent_station"] = ""
        df_stops.loc[df_stops["parent_station"].isna() & (df_stops["location_type"] == 0), "location_type"] = 1

    if "transfers" in feed:
        df_transfers = feed["transfers"]

        if not "min_transfer_time" in df_transfers:
            df_transfers["min_transfer_time"] = 0

        f = df_transfers["min_transfer_time"].isna()
        if np.any(f):
            print("WARNING NaN numbers for min_transfer_time in transfers")
            df_transfers = df_transfers[~f]

        df_transfers["min_transfer_time"] = df_transfers["min_transfer_time"].astype(int)
        feed["transfers"] = df_transfers

    if "agency" in feed:
        df_agency = feed["agency"]
        if "agency_id" not in df_agency.columns:
            df_agency["agency_id"] = "generic"
        df_agency.loc[df_agency["agency_id"].isna(), "agency_id"] = "generic"

    if "routes" in feed:
        df_routes = feed["routes"]
        agency_id = feed["agency"]["agency_id"].values[0]

        if not "agency_id" in df_routes:
            df_routes["agency_id"] = agency_id

        df_routes.loc[df_routes["agency_id"].isna(), "agency_id"] = agency_id

    if "shapes" in feed: del feed["shapes"]
    feed["trips"]["shape_id"] = np.nan

    # Fixes for Nantes PDL
    for item in feed.keys():
        feed[item] = feed[item].drop(columns = [
            c for c in feed[item].columns if c.startswith("ext_")
        ])

    return feed

def write_feed(feed, path):
    print("Writing GTFS data to %s ..." % path)

    if path.endswith(".zip"):
        with zipfile.ZipFile(path, "w") as zip:
            for slot in REQUIRED_SLOTS + OPTIONAL_SLOTS:
                if slot in feed:
                    print("  Writing %s.txt ..." % slot)

                    # We cannot write directly to the file handle as it
                    # is binary, but pandas only writes in text mode.
                    zip.writestr("%s.txt" % slot, feed[slot].to_csv(index=None, lineterminator="\n"))
    else:
        if not os.path.exists(path):
            os.mkdir(path)

        if not os.path.isdir(path):
            raise RuntimeError("Should be a directory: %s" % path)

        for slot in REQUIRED_SLOTS + OPTIONAL_SLOTS:
            if slot in feed:
                with open("%s/%s.txt" % (path, slot), "w+", encoding="utf-8") as f:
                    print("  Writing %s.txt ..." % slot)
                    feed[slot].to_csv(f, index=None, lineterminator="\n")

def cut_feed(feed, df_area, crs = None):
    feed = copy_feed(feed)

    df_stops = feed["stops"]

    if "location_type" not in df_stops.columns or np.count_nonzero(df_stops["location_type"] == 1) == 0:
        print("Warning! Location types seem to be malformatted. Keeping all stops.")
        df_stations = df_stops.copy()
    else:
        df_stations = df_stops[df_stops["location_type"] == 1].copy()

    df_stations["geometry"] = [
        geo.Point(*xy)
        for xy in zip(df_stations["stop_lon"], df_stations["stop_lat"])
    ]

    df_stations = gpd.GeoDataFrame(df_stations, crs = "EPSG:4326")

    if not crs is None:
        print("Converting stops to custom CRS", crs)
        df_stations = df_stations.to_crs(crs)
    elif not df_area.crs is None:
        print("Converting stops to area CRS", df_area.crs)
        df_stations = df_stations.to_crs(df_area.crs)

    print("Filtering stations ...")
    initial_count = len(df_stations)

    df_stations = gpd.sjoin(df_stations, df_area, predicate = "within")
    final_count = len(df_stations)

    print("Found %d/%d stations inside the specified area" % (final_count, initial_count))
    inside_stations = df_stations["stop_id"]

    # 1) Remove stations that are not inside stations and not have a parent stop
    df_stops = feed["stops"]

    df_stops = df_stops[
        df_stops["parent_station"].isin(inside_stations) |
        (
            df_stops["parent_station"].isna() &
            df_stops["stop_id"].isin(inside_stations)
        )
    ]

    feed["stops"] = df_stops.copy()
    remaining_stops = feed["stops"]["stop_id"].unique()

    # 2) Remove stop times
    df_times = feed["stop_times"]
    df_times = df_times[df_times["stop_id"].astype(str).isin(remaining_stops.astype(str))]
    feed["stop_times"] = df_times.copy()

    # 3) Remove transfers
    if "transfers" in feed:
        df_transfers = feed["transfers"]
        df_transfers = df_transfers[
            df_transfers["from_stop_id"].isin(remaining_stops) &
            df_transfers["to_stop_id"].isin(remaining_stops)
        ]
        feed["transfers"] = df_transfers.copy()

    # 4) Remove pathways
    if "pathways" in feed:
        df_pathways = feed["pathways"]
        df_pathways = df_pathways[
            df_pathways["from_stop_id"].isin(remaining_stops) &
            df_pathways["to_stop_id"].isin(remaining_stops)
        ]
        feed["pathways"] = df_pathways.copy()

    # 5) Remove trips
    trip_counts = feed["stop_times"]["trip_id"].value_counts()
    remaining_trips = trip_counts[trip_counts > 1].index.values

    df_trips = feed["trips"]
    df_trips = df_trips[
        df_trips["trip_id"].isin(remaining_trips)
    ]
    feed["trips"] = df_trips.copy()

    feed["stop_times"] = feed["stop_times"][
        feed["stop_times"]["trip_id"].isin(df_trips["trip_id"].unique())
    ]

    # 6) Remove frequencies
    if "frequencies" in feed:
        df_frequencies = feed["frequencies"]
        df_frequencies = df_frequencies[
            df_frequencies["trip_id"].isin(remaining_trips)
        ]
        feed["frequencies"] = df_frequencies.copy()

    return feed

SLOT_COLLISIONS = [
    { "slot": "agency", "identifier": "agency_id", "references": [
        ("routes", "agency_id"), ("fare_attributes", "agency_id")] },
    { "slot": "stops", "identifier": "stop_id", "references": [
        ("stops", "parent_station"), ("stop_times", "stop_id"),
        ("transfers", "from_stop_id"), ("transfers", "to_stop_id"),
        ("pathways", "from_stop_id"), ("pathways", "to_stop_id")] },
    { "slot": "routes", "identifier": "route_id", "references": [
        ("trips", "route_id"), ("fare_rules", "route_id"),
        ("attributions", "route_id")] },
    { "slot": "trips", "identifier": "trip_id", "references": [
        ("stop_times", "trip_id"), ("frequencies", "trip_id"),
        ("attributions", "trip_id")] },
    { "slot": "calendar", "identifier": "service_id", "references": [
        ("calendar_dates", "service_id"), ("trips", "service_id")] },
    { "slot": "calendar_dates", "identifier": "service_id", "references": [
        ("trips", "service_id"), ("calendar", "service_id")] },
    { "slot": "fare_attributes", "identifier": "fare_id", "references": [
        ("fare_rules", "fare_id")] },
    { "slot": "shapes", "identifier": "shape_id", "references": [
        ("trips", "shape_id")] },
    { "slot": "pathways", "identifier": "pathway_id", "references": [] },
    { "slot": "levels", "identifier": "level_id", "references": [
        ("stops", "level_id")] },
    { "slot": "attributions", "identifier": "attribution_id" },
]

def copy_feed(feed):
    return {
        slot: feed[slot].copy() for slot in feed
    }

def merge_feeds(feeds):
    result = {}

    for k, feed in enumerate(feeds):
        result = merge_two_feeds(result, feed, "_m{}".format(k + 1))

    return result

def merge_two_feeds(first, second, suffix = "_merged"):
    feed = {}

    print("Merging GTFS data ...")

    first = copy_feed(first)
    second = copy_feed(second)

    for collision in SLOT_COLLISIONS:
        if collision["slot"] in first and collision["slot"] in second:
            df_first = first[collision["slot"]]
            df_second = second[collision["slot"]]

            if collision["identifier"] in df_first and collision["identifier"] in df_second:
                df_concat = pd.concat([df_first, df_second], sort = True).drop_duplicates()
                duplicate_ids = list(df_concat[df_concat[collision["identifier"]].duplicated()][collision["identifier"]].unique())

                if len(duplicate_ids) > 0:
                    print("   Found %d duplicate identifiers in %s" % (
                        len(duplicate_ids), collision["slot"]))

                    replacement_ids = [str(id) + suffix for id in duplicate_ids]

                    df_second[collision["identifier"]] = df_second[collision["identifier"]].replace(
                        duplicate_ids, replacement_ids
                    )

                    for ref_slot, ref_identifier in collision["references"]:
                        if ref_slot in second:
                            second[ref_slot][ref_identifier] = second[ref_slot][ref_identifier].replace(
                                duplicate_ids, replacement_ids
                            )

    for slot in REQUIRED_SLOTS + OPTIONAL_SLOTS:
        if slot in first and slot in second:
            feed[slot] = pd.concat([first[slot], second[slot]], sort = True).drop_duplicates()
        elif slot in first:
            feed[slot] = first[slot].copy()
        elif slot in second:
            feed[slot] = second[slot].copy()

    return feed

def gtfs_to_seconds(time_str):
    """Convertit du format HH:MM:SS en secondes"""
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s

def secondes_to_gtfs(secondes):
    """Convertit des sedonces au format HH:MM:SS"""
    h = secondes // 3600
    m = (secondes % 3600) // 60
    s = (secondes % 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


def get_active_services(feed, target_date):
    """Renvoie l'ensemble des service_id actifs pour la date target_date."""

    target_dt = datetime.datetime.strptime(target_date, "%Y%m%d")
    weekday = target_dt.strftime("%A").lower()

    active = set()

    if "calendar" in feed:
        cal = feed["calendar"]

        for _, r in cal.iterrows():
            if (
                r["start_date"] <= int(target_date) <= r["end_date"]
                and r[weekday] == 1
            ):
                active.add(str(r["service_id"]))

    if "calendar_dates" in feed:
        cald = feed["calendar_dates"]

        for _, r in cald.iterrows():
            if int(r["date"]) == int(target_date):

                if r["exception_type"] == 1:
                    active.add(str(r["service_id"]))

                elif r["exception_type"] == 2:
                    active.discard(str(r["service_id"]))

    return active

def filtre_periode(trip_infos, START_TIME, END_TIME):
    """Filtre les missions dont l'heure de départ est dans la période considérée"""
    start_sec = gtfs_to_seconds(START_TIME)
    end_sec = gtfs_to_seconds(END_TIME)

    return trip_infos[(trip_infos["departure_sec"] >= start_sec) & (trip_infos["departure_sec"] <= end_sec)]

def filtre_missions(trips, stop_times, START_TIME, END_TIME):
    """Permet de filtrer toutes les missions éligibles et qu'il faudra ensuite simplifier"""
    grouped = stop_times.groupby("trip_id")

    signatures = grouped["stop_id"].agg(tuple) # Creation de la liste des arrêts sous forme de tupple

    # Heure de départ au premier arrêt
    first_departures = grouped["departure_sec"].first()

    # Heure d'arrivée au dernier arrêt
    last_arrivals = grouped["arrival_sec"].last()

    trip_infos = pd.DataFrame({
        "trip_id": signatures.index,
        "signature": signatures.values,
        "departure_sec": first_departures.values,
        "duration_sec": (last_arrivals - first_departures).values,
    })

    #Ajout de l'information de la route_id et de la direction
    trip_infos = trip_infos.merge(trips[["trip_id", "route_id", "direction_id"]],on="trip_id",how="left") 

    # Filtre sur la période traitée
    trip_infos = filtre_periode(trip_infos, START_TIME, END_TIME)

    # Debug sur certains trip_id
    debug_trips = {
        "OCESN886700F1187_F:TER:FR:Line::6349C213-7B19-4078-BF2B-43CBECDA9545::87743716:87726000:13:710:20260831",
        "OCESN886734F1187_F:TER:FR:Line::6349C213-7B19-4078-BF2B-43CBECDA9545::87743716:87726000:13:1740:20260831",
        "OCESN886760F1187_F:TER:FR:Line::6349C213-7B19-4078-BF2B-43CBECDA9545::87743716:87726000:13:1240:20260831",
    }

    print("\n=== DEBUG TRIP_INFOS ===")
    print(
        trip_infos.loc[
            trip_infos["trip_id"].isin(debug_trips)
        ].to_string()
    ) 

    return trip_infos

def elect_reference_trip(missions):
    """Retourne le trip_id de la mission élue comme représentative"""
    durations = missions["duration_sec"] #ensemble des durées des missions

    median_duration = durations.median() # calcul de la médiane

    reference_trip = missions.iloc[(durations - median_duration).abs().argmin()]

    return reference_trip["trip_id"]   

def build_time_pattern(reference_trip_id, stop_times, trip_indices):
    """Stocke pour la mission de référence, les temps de parcours inter-arrêts"""
    idx = trip_indices[reference_trip_id]

    reference = stop_times.loc[idx].sort_values("stop_sequence")

    offset = reference.iloc[0]["departure_sec"] # Heure de départ de la mission sert d'offset

    # reference_stop_times["arrival_offset"] = reference_stop_times["arrival_time"].apply(gtfs_to_seconds) - offset
    # reference_stop_times["departure_offset"] = reference_stop_times["departure_time"].apply(gtfs_to_seconds) - offset

    return {
        "arrival_offset": (reference["arrival_sec"] - offset).to_numpy(),
        "departure_offset": (reference["departure_sec"] - offset).to_numpy()
    }

def apply_pattern_inplace(stop_times, trip_indices, reference_offsets, trip_id):
    """Met à jour directement les stop_times d'une mission sans recréer de DataFrame."""

    idx = trip_indices[trip_id]

    departure_time = stop_times.loc[idx, "departure_sec"].iloc[0]

    if len(idx) != len(reference_offsets["arrival_offset"]):
        raise ValueError(f"Trip {trip_id} incompatible avec le pattern")

    # Sauvegarde des anciennes valeurs
    old_arrivals = stop_times.loc[idx, "arrival_sec"].to_numpy().copy()
    old_departures = stop_times.loc[idx, "departure_sec"].to_numpy().copy()

    # Calcul des nouvelles valeurs
    new_arrivals = (
        departure_time + reference_offsets["arrival_offset"]
    )

    new_departures = (
        departure_time + reference_offsets["departure_offset"]
    )

    # Application
    stop_times.loc[idx, "arrival_sec"] = new_arrivals
    stop_times.loc[idx, "departure_sec"] = new_departures

    # Log détaillé uniquement pour certains trips
    debug_trips = {
        "OCESN886700F1187_F:TER:FR:Line::6349C213-7B19-4078-BF2B-43CBECDA9545::87743716:87726000:13:710:20260831",
        "OCESN886734F1187_F:TER:FR:Line::6349C213-7B19-4078-BF2B-43CBECDA9545::87743716:87726000:13:1740:20260831",
        "OCESN886760F1187_F:TER:FR:Line::6349C213-7B19-4078-BF2B-43CBECDA9545::87743716:87726000:13:1240:20260831",
    }

    if trip_id in debug_trips:
        print(f"\n=== Trip modifié : {trip_id} ===")

        for i, (old_a, new_a, old_d, new_d) in enumerate(
            zip(old_arrivals, new_arrivals, old_departures, new_departures),
            start=1
        ):
            print(
                f"Arrêt {i} | "
                f"Arrival : {secondes_to_gtfs(old_a)} -> {secondes_to_gtfs(new_a)} | "
                f"Departure : {secondes_to_gtfs(old_d)} -> {secondes_to_gtfs(new_d)}"
            )

def simplify_feed(feed, target_date, start_time="00:00:00", end_time="24:00:00"):
    # Lecture des fichiers
    trips = feed["trips"].copy()
    stop_times = feed["stop_times"].copy()
    stops = feed["stops"].copy()

    # Uniformisation des types pour éviter les problèmes de comparaison
    trips["service_id"] = trips["service_id"].astype(str)

    # Filtrage sur la date du GTFS souhaitée
    active = get_active_services(feed, target_date)

    print("SERVICES ACTIFS =====>", active)

    active_trips = trips[
        trips["service_id"].isin(active)
    ].copy()

    print(
        active_trips[["trip_id", "service_id"]].to_string()
    )

    # IMPORTANT :
    # on ne garde que les stop_times des trips actifs
    active_stop_times = stop_times[
        stop_times["trip_id"].isin(active_trips["trip_id"])
    ].copy()

    # Tri des stop_times par trip_id et stop_sequence
    active_stop_times = active_stop_times.sort_values(
        ["trip_id", "stop_sequence"]
    )

    # Création des colonnes pour avoir le temps en secondes
    active_stop_times["departure_sec"] = (
        active_stop_times["departure_time"].map(gtfs_to_seconds)
    )

    active_stop_times["arrival_sec"] = (
        active_stop_times["arrival_time"].map(gtfs_to_seconds)
    )

    # Récupération des missions éligibles
    missions_eligibles = filtre_missions(
        active_trips,
        active_stop_times,
        start_time,
        end_time
    )

    print(
        f"Nombre de groupes de missions : "
        f"{missions_eligibles.groupby(['route_id', 'direction_id', 'signature']).ngroups}"
    )

    # Stockage des numéros de lignes que l'on va modifier
    trip_indices = active_stop_times.groupby("trip_id").groups

    # Boucle par couple (route_id, direction_id, signature)
    for cle, missions in missions_eligibles.groupby(
        ["route_id", "direction_id", "signature"]
    ):

        if len(missions) < 2:
            continue

        # Élection de la mission représentative
        reference_trip_id = elect_reference_trip(missions)

        # Récupération du pattern (temps de parcours)
        pattern = build_time_pattern(
            reference_trip_id,
            active_stop_times,
            trip_indices,
        )

        print(
            f"Mise à jour des temps de {len(missions)} missions. "
            f"La mission référence est {reference_trip_id}"
        )

        for trip_id in missions["trip_id"]:

            if trip_id == reference_trip_id:
                continue

            apply_pattern_inplace(
                active_stop_times,
                trip_indices,
                pattern,
                trip_id,
            )

    # Conversion des secondes vers le format GTFS
    active_stop_times["arrival_time"] = (
        active_stop_times["arrival_sec"].map(secondes_to_gtfs)
    )

    active_stop_times["departure_time"] = (
        active_stop_times["departure_sec"].map(secondes_to_gtfs)
    )

    active_stop_times = active_stop_times.drop(
        columns=["arrival_sec", "departure_sec"]
    )

    # Remplacement des stop_times actifs par leur version simplifiée
    active_trip_ids = set(active_trips["trip_id"])

    stop_times = stop_times[
        ~stop_times["trip_id"].isin(active_trip_ids)
    ]

    stop_times = pd.concat(
        [stop_times, active_stop_times],
        ignore_index=True
    )

    # Filtre des stops non utilisés
    # used_stops = set(stop_times["stop_id"])
    # print(
    #     f"Seulement {len(used_stops)} utilisés sur {len(stops)} : "
    #     f"suppression de {len(stops) - len(used_stops)} stops"
    # )
    # stops = stops[stops["stop_id"].isin(used_stops)].copy()

    feed["trips"] = trips
    feed["stop_times"] = stop_times
    feed["stops"] = stops

    print("INFO : Simplification des GTFS terminée")

    return feed
