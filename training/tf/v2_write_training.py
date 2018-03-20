#!/usr/bin/env python3
#
# Used to dump training games in V2 format from MongoDB or V1 chunk files.
#
# Usage: v2_write_training [chunk_prefix]
#   If run without a chunk_prefix it reads from MongoDB.
#  With a chunk prefix, it uses all chunk files with that prefix
#  as input.
#
# Sets up a dataflow pipeline that:
# 1. Reads from input (MongoDB or v1 chunk files)
# 2. Split into a test set and a training set.
# 3. Converts from v1 format to v2 format.
# 4. Shuffle V2 records.
# 5. Write out to compressed v2 chunk files.
#

from chunkparser import ChunkParser
import glob
import gzip
import itertools
import multiprocessing as mp
import numpy as np
import pymongo
import sys

def mongo_fetch_games(q_out, num_games):
    """
        Read V1 format games from MongoDB and put them
        in the output queue (q_out)

        Reads a network list from MongoDB from most recents,
        and then reads games produced by those network until
        'num_games' has been read.
    """
    client = pymongo.MongoClient()
    db = client.test
    # MongoDB closes idle cursors after 10 minutes unless specific
    # options are given. That means this query will time out before
    # we finish. Rather than keeping it alive, increase the default
    # batch size so we're sure to get all networks in the first fetch.
    networks = db.networks.find(None, {"_id": False, "hash": True}).\
        sort("_id", pymongo.DESCENDING).batch_size(5000)

    game_count = 0
    for net in networks:
        print("Searching for {}".format(net['hash']))

        games = db.games.\
            find({"networkhash": net['hash']},
                 {"_id": False, "data": True})

        for game in games:
            game_data = game['data']
            q_out.put(game_data.encode("ascii"))

            game_count += 1
            if game_count >= num_games:
                q_out.put('STOP')
                return
            if game_count % 1000 == 0:
                print("{} games".format(game_count))

def disk_fetch_games(q_out, prefix):
    """
        Fetch chunk files off disk.

        Chunk files can be either v1 or v2 format.
    """
    files = glob.glob(prefix + "*.gz")
    for f in files:
        with gzip.open(f, 'rb') as chunk_file:
            v = chunk_file.read()
            q_out.put(v)
            print("In {}".format(f))
    q_out.put('STOP')

def fake_fetch_games(q_out, num_games):
    """
        Generate V1 format fake games. Used for testing and benchmarking
    """
    for _ in range(num_games):
        # Generate a 200 move 'game'
        # Generate a random game move.
        # 1. 18 binary planes of length 361
        planes = [np.random.randint(2, size=361).tolist() for plane in range(16)]
        stm = float(np.random.randint(2))
        planes.append([stm] * 361)
        planes.append([1. - stm] * 361)
        # 2. 362 probs
        probs = np.random.randint(3, size=362).tolist()
        # 3. And a winner: 1 or -1
        winner = [ 2 * float(np.random.randint(2)) - 1 ]

        # Convert that to a v1 text record.
        items = []
        for p in range(16):
            # generate first 360 bits
            h = np.packbits([int(x) for x in planes[p][0:360]]).tobytes().hex()
            # then add the stray single bit
            h += str(planes[p][360]) + "\n"
            items.append(h)
        # then side to move
        items.append(str(int(planes[17][0])) + "\n")
        # then probabilities
        items.append(' '.join([str(x) for x in probs]) + "\n")
        # and finally if the side to move is a winner
        items.append(str(int(winner[0])) + "\n")
        game = ''.join(items)
        game = game * 200
        game = game.encode('ascii')

        q_out.put(game)
    q_out.put('STOP')

def queue_gen(q, out_qs):
    """
        Turn a queue into a generator

        Yields items pulled from 'q' until a 'STOP' item is seen.
        The STOP item will be propogated to all the queues in
        the list 'out_qs' (if any).
    """
    while True:
        try:
            item = q.get()
        except:
            break
        if item == 'STOP':
            break
        yield item
    # There might be multiple workers reading from this queue,
    # and they all need to be stopped, so put the STOP token
    # back in the queue.
    q.put('STOP')
    # Stop any listed output queues as well
    for x in out_qs:
        x.put('STOP')

def split_train_test(q_in, q_train, q_test):
    """
        Stream a stream of chunks into separate train and test
        pools. 10% of the chunks are assigned to test.

        Uses hash sharding, so multiple runs will split chunks
        in the same way.
    """
    for item in queue_gen(q_in, [q_train, q_test]):
        # Use the hash of the game to determine the split. This means
        # that test games will never be used for training.
        h = hash(item) & 0xfff
        if h < 0.1*0xfff:
            # a test game.
            q_test.put(item)
        else:
            q_train.put(item)

class QueueChunkSrc:
    def __init__(self, q):
        self.q = q
        self.gen = None
    def next(self):
        print("Queue next")
        if self.gen is None:
            self.gen = queue_gen(self.q,[])
        try:
            return next(self.gen)
        except:
            return None


def chunk_parser(q_in, q_out, shuffle_size, chunk_size):
    """
        Parse input chunks from 'q_in', shuffle, and put
        chunks of moves in v2 format into 'q_out'

        Each output chunk contains 'chunk_size' moves.
        Moves are shuffled in a buffer of 'shuffle_size' moves.
        (A 2^20 items shuffle buffer is ~ 2.2GB of RAM).
    """
    workers = max(1, mp.cpu_count() - 2)
    parse = ChunkParser(QueueChunkSrc(q_in),
                        shuffle_size=shuffle_size,
                        workers=workers)
    gen = parse.v2_gen()
    while True:
        s = list(itertools.islice(gen, chunk_size))
        if not len(s):
            break
        s = b''.join(s)
        q_out.put(s)
    q_out.put('STOP')

def chunk_writer(q_in, namesrc):
    """
        Write a batch of moves out to disk as a compressed file.

        Filenames are taken from the generator 'namegen'.
    """
    for chunk in queue_gen(q_in,[]):
        filename = namesrc.next()
        chunk_file = gzip.open(filename, 'w', 1)
        chunk_file.write(chunk)
        chunk_file.close()
    print("chunk_writer completed")

class NameSrc:
    """
        Generator a sequence of names, starting with 'prefix'.
    """
    def __init__(self, prefix):
        self.prefix = prefix
        self.n = 0
    def next(self):
        print("Name next")
        self.n += 1
        return self.prefix + "{:0>8d}.gz".format(self.n)

def main(args):
    # Build the pipeline.
    procs=[]
    # Read from input.
    q_games = mp.SimpleQueue()
    if args:
        prefix = args.pop(0)
        print("Reading from chunkfiles {}".format(prefix))
        procs.append(mp.Process(target=disk_fetch_games, args=(q_games, prefix)))
    else:
        print("Reading from MongoDB")
        #procs.append(mp.Process(target=fake_fetch_games, args=(q_games, 20)))
        procs.append(mp.Process(target=mongo_fetch_games, args=(q_games, 275000)))
    # Split into train/test
    q_test = mp.SimpleQueue()
    q_train = mp.SimpleQueue()
    procs.append(mp.Process(target=split_train_test, args=(q_games, q_train, q_test)))
    # Convert v1 to v2 format and shuffle, writing 8192 moves per chunk.
    q_write_train = mp.SimpleQueue()
    q_write_test = mp.SimpleQueue()
    # Shuffle buffer is ~ 2.2GB of RAM with 2^20 (~1e6) entries. A game is ~500 moves, so
    # there's ~2000 games in the shuffle buffer. Selecting 8k moves gives an expected
    # number of ~4 moves from the same game in a given chunk file.
    #
    # The output files are in parse.py via another 1e6 sized shuffle buffer. At 8192 moves
    # per chunk, there's ~ 128 chunks in the shuffle buffer. With a batch size of 4096,
    # the expected max number of moves from the same game in the batch is < 1.14
    procs.append(mp.Process(target=chunk_parser, args=(q_train, q_write_train, 1<<20, 8192)))
    procs.append(mp.Process(target=chunk_parser, args=(q_test, q_write_test, 1<<16, 8192)))
    # Write to output files
    procs.append(mp.Process(target=chunk_writer, args=(q_write_train, NameSrc('train_'))))
    procs.append(mp.Process(target=chunk_writer, args=(q_write_test, NameSrc('test_'))))

    # Start all the child processes running.
    for p in procs:
        p.start()
    # Wait for everything to finish.
    for p in procs:
        p.join()
    # All done!

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main(sys.argv[1:])
