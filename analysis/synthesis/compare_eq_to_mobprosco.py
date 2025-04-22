
import os
import numpy as np
import pandas as pd

ANALYSIS_FOLDER = "analysis_population"
def configure(context):

    context.config("output_path")
    context.config("output_prefix", "ile_de_france_")
    context.config("sampling_rate")

    context.stage("synthesis.population.trips")
    context.stage("synthesis.population.enriched")
    context.stage("synthesis.population.spatial.locations")

    context.stage("data.od.cleaned")

def execute(context):

    # check output folder existence
    analysis_output_path = os.path.join(context.config("output_path"), ANALYSIS_FOLDER)
    if not os.path.exists(analysis_output_path):
        os.mkdir(analysis_output_path)

    
    prefix = context.config("output_prefix")
    sampling_rate = context.config("sampling_rate")
    df_trip_eq = context.stage("synthesis.population.trips")
    df_location_eq = context.stage("synthesis.population.spatial.locations")[["person_id", "activity_index", "commune_id"]]
    df_trip_eq["preceding_activity_index"] = df_trip_eq["trip_index"]
    df_trip_eq["following_activity_index"] = df_trip_eq["trip_index"] + 1  

    df_spatial = pd.merge(df_trip_eq, df_location_eq.rename(columns = {
        "activity_index": "preceding_activity_index",
        "commune_id": "origin_id"
    }), how = "left", on = ["person_id", "preceding_activity_index"])

    df_spatial = pd.merge(df_spatial, df_location_eq.rename(columns = {
        "activity_index": "following_activity_index",
        "commune_id": "destination_id"
    }), how = "left", on = ["person_id", "following_activity_index"])

    for df_data, name in zip(context.stage("data.od.cleaned"), ("work", "education")):
        pivot_data = pd.pivot_table(df_data,"weight",index="origin_id",columns="destination_id",aggfunc="sum")
        pivot_eq = pd.pivot_table(df_spatial[df_spatial["following_purpose"]==name],"person_id",index="origin_id",columns="destination_id",aggfunc="count")
        corr_dest = pivot_data.corrwith(pivot_eq).reset_index()
        corr_ori = pivot_data.corrwith(pivot_eq,axis=1).reset_index()
        corr = pd.merge(corr_ori,corr_dest,left_on="origin_id",right_on="destination_id")
        corr.to_csv(f"{analysis_output_path}/{prefix}{name}_corr.csv")
        #print(set(pivot_data.columns) - set(pivot_eq.columns) )
        pivot_data.sort_index().sort_index(axis=1).compare(pivot_eq.sort_index().sort_index(axis=1),result_names=("mob", "eq")).to_csv(f"{analysis_output_path}/{prefix}{name}_compare.csv")

