
import pandas as pd
import geopandas as gpd
import os
import zipfile

"""
This stage loads the raw data of the Spanish HTS (EDM).
"""

MENAGES_COLUMNS = [
   "ID_HOGAR", "CODMUNI", "NOMMUNI", "CODPROV", "NOMPROV", "ELE_HOGAR_NUEVO", "B1NVE",  
]

PERSONNES_COLUMNS = [
     "ID_HOGAR", "ID_IND","C2SEXO" , "EDAD_FIN", "ELE_G_POND", 
     "C6CARNE", "C8ACTIV", "C10SECTOR", "C14ABONO"
]

DEPLACEMENTS_COLUMNS = [
    "ID_HOGAR", "ID_IND","ID_VIAJE" ,"VDES","VORI", "VORIHORAINI", "VDESHORAFIN", "VORIZT1259",
"VDESZT1259" ,"DISTANCIA_VIAJE" , "MODO_PRIORITARIO", "ELE_G_POND_ESC2"
]

def configure(context):
    context.config("data_path")

def execute(context):
    df_households = pd.read_excel(
        "%s/edm_spain_2018/EDM2018HOGARES.xlsx" % context.config("data_path"),sheet_name="HOGARES",
        usecols = MENAGES_COLUMNS
    )

    df_persons = pd.read_excel(
        "%s/edm_spain_2018/EDM2018INDIVIDUOS.xlsx" % context.config("data_path"),sheet_name="INDIVIDUOS",
         usecols = PERSONNES_COLUMNS
    )

    df_trips = pd.read_excel(
        "%s/edm_spain_2018/EDM2018VIAJES.xlsx" % context.config("data_path"),sheet_name="VIAJES",
         usecols = DEPLACEMENTS_COLUMNS
    )

    # Load spatial data
    df_spatial = gpd.read_file(
        "%s/edm_spain_2018/ZonificationZT1259_con_mun.geojson"
        % context.config("data_path"))

    return df_households, df_persons, df_trips, df_spatial

def validate(context):
    for name in ("EDM2018HOGARES.xlsx", "EDM2018INDIVIDUOS.xlsx", "EDM2018VIAJES.xlsx","ZonificationZT1259_con_mun.geojson"):
        if not os.path.exists("%s/edm_spain_2018/%s" % (context.config("data_path"), name)):
            raise RuntimeError("File missing from EDM: %s" % name)

    return [
        os.path.getsize("%s/edm_spain_2018/ZonificationZT1259_con_mun.geojson" % context.config("data_path")),
        os.path.getsize("%s/edm_spain_2018/EDM2018HOGARES.xlsx" % context.config("data_path")),
        os.path.getsize("%s/edm_spain_2018/EDM2018INDIVIDUOS.xlsx" % context.config("data_path")),
        os.path.getsize("%s/edm_spain_2018/EDM2018VIAJES.xlsx" % context.config("data_path"))
    ]