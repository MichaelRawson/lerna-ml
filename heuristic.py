#!/usr/bin/env python3

import logging as log
import json
from threading import Thread
from queue import Empty, Queue
from socketserver import StreamRequestHandler, TCPServer

import torch
from torch.nn.functional import softmax
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected

BATCH = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RUNNING = True
IN_QUEUE = Queue(maxsize=2 * BATCH)
OUT_QUEUE = Queue()
HOST = "localhost"
PORT = 1337

def read_input(line):
    record = json.loads(line)
    identifier = record['id']
    x = torch.tensor(record['nodes'], dtype=torch.long)
    num_nodes = len(x)
    edge_index = to_undirected(torch.tensor(record['edges'], dtype=torch.long).view(-1, 2).t(), num_nodes=num_nodes)
    data = Data(x=x, edge_index=edge_index)
    return (identifier, data)

def write_score(f, identifier, score):
    record = {"id": identifier, "score": score}
    string = json.dumps(record)
    f.write(f"{record}\n".encode('ascii'))
    f.flush()

def heuristic_task(model):
    model.eval()
    while RUNNING:
        batch = []
        ids = []
        while len(batch) < BATCH:
            identifier, data = IN_QUEUE.get()
            ids.append(identifier)
            batch.append(data)

        batch = Batch.from_data_list(batch).to(DEVICE)
        scores = softmax(model(batch.x, batch.edge_index, batch.batch), dim=1)[:, 1]
        for identifier, score in zip(ids, scores):
            OUT_QUEUE.put((identifier, score.item()))

class Handler(StreamRequestHandler):
    def handle(self):
        while True:
            try:
                while True:
                    identifier, score = OUT_QUEUE.get_nowait()
                    write_score(self.wfile, identifier, score)
            except Empty:
                pass

            line = self.rfile.readline()
            try:
                identifier, data = read_input(line)
                IN_QUEUE.put((identifier, data))
            except json.JSONDecodeError:
                return

class Server(TCPServer):
    allow_reuse_address = True

if __name__ == '__main__':
    import sys
    log.basicConfig(level=log.INFO)

    log.info("loading model")
    model = torch.load(sys.argv[1]).to(DEVICE)
    Thread(target=heuristic_task, args=(model,), daemon=True).start()
    log.info("model loaded")

    with Server((HOST, PORT), Handler) as server:
        log.info("server started")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
            log.info("server shutdown")
