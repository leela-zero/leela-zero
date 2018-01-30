#!/usr/bin/env python3

import gzip
import pymongo

client = pymongo.MongoClient()
db = client.test

# MongoDB closes idle cursors after 10 minutes unless specific
# options are given. That means this query will time out before
# we finish. Rather than keeping it alive, increase the default
# batch size so we're sure to get all networks in the first fetch.
networks = db.networks.find(None, {"_id": False, "hash": True}).\
    sort("_id", pymongo.DESCENDING).batch_size(5000)

game_count = 0
total_game_count = 0
chunk_file = None

def get_chunk_name(hash):
    return "train_" + hash[0:8] + "_" + str(chunk_count) + ".gz"

for net in networks:
    print("Searching for {}".format(net['hash']))

    games = db.games.\
        find({"networkhash": net['hash']},
             {"_id": False, "data": True})

    chunk_count = 0
    if chunk_file:
        chunk_file.close()
    chunk_file = gzip.open(get_chunk_name(net['hash']), 'w', 1)

    for game in games:
        game_data = game['data']
        chunk_file.write(game_data.encode("ascii"))
        game_count += 1
        total_game_count += 1
        if game_count >= 64:
            chunk_file.close()
            chunk_count += 1
            chunk_file = gzip.open(get_chunk_name(net['hash']), 'w', 1)
            game_count = 0
            print("Net {} Chunk {} written".format(net['hash'][0:8], chunk_count))
        if total_game_count >= 275000:
            chunk_file.close()
            quit()

chunk_file.close()
