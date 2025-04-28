
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
    
    context.stage("data.spatial.municipalities")
    context.stage("data.od.cleaned")

def execute(context):

    # check output folder existence
    analysis_output_path = os.path.join(context.config("output_path"), ANALYSIS_FOLDER)
    if not os.path.exists(analysis_output_path):
        os.mkdir(analysis_output_path)

    # get trips eqasim data
    prefix = context.config("output_prefix")
    sampling_rate = context.config("sampling_rate")
    df_zones = context.stage("data.spatial.municipalities")
    required_communes = set(df_zones["commune_id"].unique()) 

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
    # stat for work and education
    for df_data, name in zip(context.stage("data.od.cleaned"), ("work", "education")):
        # get mob trajects volume
        pivot_data = pd.pivot_table(df_data,"weight",index="origin_id",columns="destination_id",aggfunc="sum")
        pivot_data = pivot_data.reindex(index=required_communes,columns=required_communes).fillna(0)   
        print("Number OD MOB for %s: %f " % (name,len(df_data[df_data["weight"]>0][["origin_id","destination_id"]].drop_duplicates())))
        # get eqasim trajects volume
        pivot_eq = pd.pivot_table(df_spatial[(df_spatial["following_purpose"]==name)],"person_id",index="origin_id",columns="destination_id",aggfunc="count")
        pivot_eq = pivot_eq.reindex(index=required_communes,columns=required_communes).fillna(0)
        pivot_eq = pivot_eq.div(sampling_rate)
        print("Number OD Eqasim all trajects for %s: %f" % (name,len(df_spatial[(df_spatial["following_purpose"]==name)][["origin_id","destination_id"]].drop_duplicates())))       
        # get eqasim Home to purpose trajects volume
        pivot_eq_home_only = pd.pivot_table(df_spatial[(df_spatial["preceding_purpose"]=='home')&(df_spatial["following_purpose"]==name)],"person_id",index="origin_id",columns="destination_id",aggfunc="count")
        pivot_eq_home_only = pivot_eq_home_only.reindex(index=required_communes,columns=required_communes).fillna(0)
        pivot_eq_home_only = pivot_eq_home_only.div(sampling_rate)
        print("Number OD Eqasim Home to %s trajects: %f" % (name,len(df_spatial[(df_spatial["preceding_purpose"]=='home')&(df_spatial["following_purpose"]==name)][["origin_id","destination_id"]].drop_duplicates())))
        # all trajects to purpose
        corr_dest = pivot_data.corrwith(pivot_eq).reset_index().rename(columns={0:"corr_dest"})
        corr_dest["mob_nbrOd"] = pivot_data[pivot_data>0].count().values
        corr_dest["eq_nbrOd"] = pivot_eq[pivot_eq>0].count().values
        corr_dest["diff_nbrOd_dest"] = corr_dest["mob_nbrOd"]- corr_dest["eq_nbrOd"]
        corr_ori = pivot_data.corrwith(pivot_eq,axis=1).reset_index().rename(columns={0:"corr_ori"})
        corr_ori["mob_nbrOd"] = pivot_data[pivot_data>0].count(axis=1).values
        corr_ori["eq_nbrOd"] = pivot_eq[pivot_eq>0].count(axis=1).values
        corr_ori["diff_nbrOd_ori"] = corr_ori["mob_nbrOd"]- corr_ori["eq_nbrOd"]
        corr = pd.merge(corr_ori[["origin_id","corr_ori","diff_nbrOd_ori"]],corr_dest[["destination_id","corr_dest","diff_nbrOd_dest"]],left_on="origin_id",right_on="destination_id")
        corr.to_csv(f"{analysis_output_path}/{prefix}{name}_corr.csv")
        print(f"Total Correlation all trajects for {name}: {np.corrcoef(pivot_data.to_numpy().flatten(),pivot_eq.to_numpy().flatten())[0,1]}")
        
        # trajects home to purpose
        corr_dest = pivot_data.corrwith(pivot_eq_home_only).reset_index().rename(columns={0:"corr_dest"})
        corr_dest["mob_nbrOd"] = pivot_data[pivot_data>0].count().values
        corr_dest["eq_nbrOd"] = pivot_eq_home_only[pivot_eq_home_only>0].count().values
        corr_dest["diff_nbrOd_dest"] = corr_dest["mob_nbrOd"]- corr_dest["eq_nbrOd"]
        corr_ori = pivot_data.corrwith(pivot_eq_home_only,axis=1).reset_index().rename(columns={0:"corr_ori"})
        corr_ori["mob_nbrOd"] = pivot_data[pivot_data>0].count(axis=1).values
        corr_ori["eq_nbrOd"] = pivot_eq_home_only[pivot_eq_home_only>0].count(axis=1).values
        corr_ori["diff_nbrOd_ori"] = corr_ori["mob_nbrOd"]- corr_ori["eq_nbrOd"]
        corr = pd.merge(corr_ori[["origin_id","corr_ori","diff_nbrOd_ori"]],corr_dest[["destination_id","corr_dest","diff_nbrOd_dest"]],left_on="origin_id",right_on="destination_id")
        corr.to_csv(f"{analysis_output_path}/{prefix}{name}_corr_home_only.csv")
        print(f"Total Correlation Home to {name} trajects: {np.corrcoef(pivot_data.to_numpy().flatten(),pivot_eq_home_only.to_numpy().flatten())[0,1]}")
