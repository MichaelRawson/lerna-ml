#!/usr/bin/env python3

import json
from random import randrange

FLAVOURS = 16
NEW_NODE_MAX = 3
NEW_EDGE_MAX = 5

def augment(line, count):
    sys.stdout.write(line)
    record = json.loads(line)
    edges = record['edges']
    nodes = record['nodes']
    y = record['y']

    for _ in range(count):
        new_nodes = nodes[:]
        new_edges = edges[:]

        for _ in range(randrange(NEW_NODE_MAX)):
            flavour = randrange(FLAVOURS)
            new_nodes.append(flavour)

        limit = len(new_nodes)
        for _ in range(randrange(NEW_EDGE_MAX)):
            from_node = randrange(limit)
            to_node = randrange(limit)
            new_edges.append([from_node, to_node])

        new_record = {'nodes': new_nodes, 'edges': new_edges, 'y': y}
        json.dump(new_record, sys.stdout)
        sys.stdout.write('\n')

if __name__ == '__main__':
    import sys
    count = int(sys.argv[1])
    for line in sys.stdin:
        augment(line, count)
