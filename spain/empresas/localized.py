import pandas as pd
import numpy as np
import geopandas as gpd
import os 

"""
This stage loads buildings from OSM to replace workplace informations.
"""

def configure(context):
    context.config("data_path")
    context.config("spain.buildings_path", "osm_spain/osm_building_202608.gpkg")

    context.stage("data.spatial.iris")
    context.config("crs","EPSG:3035")

def execute(context):

    filename = os.path.join(context.config("data_path"), context.config("spain.buildings_path"))
    df_empresas = gpd.read_file(filename)

    # Keep only places for workplace
    df_empresas = df_empresas[~(df_empresas["building"].isin(["house","residential","detached","garage","apartments"]))].to_crs(context.config("crs"))

    df_empresas = gpd.sjoin(df_empresas,context.stage("data.spatial.iris"),how="left")

    # cara work
    df_empresas["minimum_employees"] = 1.0
    df_empresas["maximum_employees"] = 1.0
    return df_empresas[[
        "commune_id", "minimum_employees", "maximum_employees", "geometry"
    ]]


def validate(context):
    filename = os.path.join(context.config("data_path"), context.config("spain.buildings_path"))
    if not os.path.isfile(filename):
        raise RuntimeError("OSM: Building data is not available")
    return os.path.getsize(filename)
   