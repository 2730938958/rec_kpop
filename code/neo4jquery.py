from neo4j import GraphDatabase

# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"
# 在https://console-preview.neo4j.io/projects/7684b722-542a-484a-8334-46715ac42bd1/instances启动instance
# 需要花费点时间
URI = "neo4j+s://30acc171.databases.neo4j.io"
AUTH = ("neo4j", "A_Arjzc6q8TkRAC0wtSULmanpNpTLSmCJqtXmNrtyMY")

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()

records, summary, keys = driver.execute_query("""
    MATCH (ITZY:Artist {artistName: "ITZY"})-[r]-(related)
    RETURN ITZY, r, related
    """,
    database_="neo4j",
)

related_li = []
# Loop through results and do something with them
for record in records[:5]:
    related_li.append({
        'artistName': record.data()['related']['artistName'],
        'topTrack': record.data()['related']['topTrack'],
        'topTrackLink': record.data()['related']['topTrackLink'],
        'topTrackAlbum': record.data()['related']['topTrackAlbum'],
        'topTrackAlbumLink' : record.data()['related']['topTrackAlbumLink'],
    }) # obtain record as dict

# Summary information
# print("The query `{query}` returned {records_count} records in {time} ms.".format(
#     query=summary.query, records_count=len(records),
#     time=summary.result_available_after
# ))
print(related_li)
driver = GraphDatabase.driver(URI, auth=AUTH)
session = driver.session(database="neo4j")

# session/driver usage

session.close()
driver.close()