import shutil

"""
Writes out the consolidated GTFS feed
"""

def configure(context):
    context.config("output_path")
    context.config("output_prefix")

    context.stage("data.gtfs.simplified")

def execute(context):
    source_path = "{}/gtfs.zip".format(context.path("data.gtfs.simplified"))
    output_path = "{}/{}gtfs.zip".format(
        context.config("output_path"), context.config("output_prefix"))
    shutil.copyfile(source_path, output_path)
