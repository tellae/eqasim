import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import KDTree
import os

"""
Loads and prepares income distributions by municipality for size of households (TAILLEM):
- Load data with centiles per municipality
- For those which only provide median: Attach another distribution with most similar median
- For those which are missing: Attach the distribution of the municiality with the nearest centroid
"""

def configure(context):
    context.config("data_path")
    context.stage("data.spatial.municipalities")

def execute(context):
    FILOSOFI_ATTRIBUTES = [
        {
            "name": "tax_referent_age",
            "modalities": [
                {"name": "0_29", "sheet": "TRAGERF_1", "col_pattern": "AGE1"},
                {"name": "30_39", "sheet": "TRAGERF_2", "col_pattern": "AGE2"},
                {"name": "40_49", "sheet": "TRAGERF_3", "col_pattern": "AGE3"},
                {"name": "50_59", "sheet": "TRAGERF_4", "col_pattern": "AGE4"},
                {"name": "60_74", "sheet": "TRAGERF_5", "col_pattern": "AGE5"},
                {"name": "75_or_more", "sheet": "TRAGERF_6", "col_pattern": "AGE6"}
            ]
        },
        {
            "name": "household_size",
            "modalities": [
                {"name": "1_pers", "sheet": "TAILLEM_1", "col_pattern": "TME1"},
                {"name": "2_pers", "sheet": "TAILLEM_2", "col_pattern": "TME2"},
                {"name": "3_pers", "sheet": "TAILLEM_3", "col_pattern": "TME3"},
                {"name": "4_pers", "sheet": "TAILLEM_4", "col_pattern": "TME4"},
                {"name": "5_pers_or_more", "sheet": "TAILLEM_5", "col_pattern": "TME5"}
            ]
        },
        {
            "name": "housing_tenure",
            "modalities": [
                {"name": "Owner", "sheet": "OCCTYPR_1", "col_pattern": "TOL1"},
                {"name": "Tenant", "sheet": "OCCTYPR_2", "col_pattern": "TOL2"}
            ]
        },
        {
            "name": "household_type",
            "modalities": [
                {"name": "Single_man", "sheet": "TYPMENR_1", "col_pattern": "TYM1"},
                {"name": "Single_wom", "sheet": "TYPMENR_2", "col_pattern": "TYM2"},
                {"name": "Couple_without_child", "sheet": "TYPMENR_3", "col_pattern": "TYM3"},
                {"name": "Couple_with_child", "sheet": "TYPMENR_4", "col_pattern": "TYM4"},
                {"name": "Single_parent", "sheet": "TYPMENR_5", "col_pattern": "TYM5"},
                {"name": "complex_hh", "sheet": "TYPMENR_6", "col_pattern": "TYM6"}
            ]
        },
        {
            "name": "income_source",
            "modalities": [
                {"name": "Salary", "sheet": "OPRDEC_1", "col_pattern": "OPR1"},
                {"name": "Unemployment", "sheet": "OPRDEC_2", "col_pattern": "OPR2"},
                {"name": "Independent", "sheet": "OPRDEC_3", "col_pattern": "OPR3"},
                {"name": "Pension", "sheet": "OPRDEC_4", "col_pattern": "OPR4"},
                {"name": "Property", "sheet": "OPRDEC_5", "col_pattern": "OPR5"},
                {"name": "None", "sheet": "OPRDEC_6", "col_pattern": "OPR6"}
            ]
        }
    ]

    # build full list of sheets
    sheet_list = []
    for attribute in FILOSOFI_ATTRIBUTES:
        sheet_list = sheet_list + [x["sheet"] for x in attribute["modalities"]]

    # read all needed sheets
    excel_df = pd.read_excel(
        "%s/filosofi_2015/FILO_DISP_COM.xls" % context.config("data_path"),
        sheet_name = sheet_list, skiprows = 5
    )

    df = pd.DataFrame()
    for attribute in FILOSOFI_ATTRIBUTES:
        for modality in attribute["modalities"]:
            sheet = modality["sheet"]
            col_pattern = modality["col_pattern"]

            # Load income distribution
            data = excel_df[sheet][["CODGEO"] +["%sD%d15" % (col_pattern, q) if q != 5 else col_pattern + "Q215" for q in range(1, 10)]]
            data.columns = ["commune_id", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9"]
            data["reference_median"] = data["q5"].values
            data["modality"] = modality["name"]
            data["attribute"] = attribute["name"]
            df = pd.concat([df, data])

    # Validation
    assert len(FILOSOFI_ATTRIBUTES) == len(df["attribute"].unique())
    assert len(sheet_list) == len(df["modality"].unique())

    return df[["commune_id", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "reference_median", "attribute", "modality"]]

def validate(context):
    if not os.path.exists("%s/filosofi_2015/FILO_DISP_COM.xls" % context.config("data_path")):
        raise RuntimeError("Filosofi data is not available")

    return os.path.getsize("%s/filosofi_2015/FILO_DISP_COM.xls" % context.config("data_path"))
