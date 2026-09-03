import data.gtfs.utils as gtfs


def configure(context):
    context.stage("data.gtfs.cleaned")
    context.config("gtfs_date","dayWithMostServices")


def execute(context):
    # GTFS nettoyé/fusionné
    gtfs_path = "{}/gtfs.zip".format(context.path("data.gtfs.cleaned"))

    feed = gtfs.read_feed(gtfs_path)
    feed = gtfs.simplify_feed( feed, target_date=context.config("gtfs_date"))

    output_path = "{}/gtfs.zip".format(context.path())

    gtfs.write_feed(feed,output_path)

    return "gtfs.zip"