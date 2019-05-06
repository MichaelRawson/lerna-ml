#!/usr/bin/env python3

import json
from random import randrange

def augment(line, count):
    sys.stdout.write(line)
    record = json.loads(line)
    edges = record['edges']
    limit = len(record['nodes'])
    for _ in range(count):
        new_edges = edges[:]
        from_node = randrange(limit)
        to_node = randrange(limit)
        new_edges.append([from_node, to_node])
        record['edges'] = new_edges
        json.dump(record, sys.stdout)
        sys.stdout.write('\n')

if __name__ == '__main__':
    import sys
    count = int(sys.argv[1])
    for line in sys.stdin:
        augment(line, count)
