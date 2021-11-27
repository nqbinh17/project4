import itertools
from collections import defaultdict

def Process2LineGraph(edges, text):
  edges = zip(*edges)
  new_edges = []
  nodes = text.split()
  from_node = defaultdict(list)
  for u, v in edges:
    if nodes[u] == nodes[v] == "IntNode":
        new_edges.append((u, v))
        new_edges.append((v, u))
        print(u, v)
    from_node[u].append(v)
  print(from_node)
  # opposite direction
  for key, community in from_node.items():
    n = len(community)
    for i in range(n):
      u = community[i]
      for j in range(i+1, n):
        v = community[j]
        new_edges.append((u, v))
        new_edges.append((v, u))
  # same direction
  for key, community in from_node.items():
    for u in community:
      if u in from_node:
        for v in from_node[u]:
          new_edges.append((key, v))
  return new_edges

def Process2LineGraph_old(size, edge, label):
  nText = ['' for _ in range(size)]
  nEdge = [[], []]

  # Process Nodes (text)
  for i, (u, v) in enumerate(zip(*edge)):
    nText[v] += label[i]

  # Process Edges
  def push2Edge(edges, type):
    if type == 1:
      for u, v in edges:
        nEdge[0].append(u)
        nEdge[1].append(v)
    elif type == 0:
      u, v = edges
      nEdge[0].append(u)
      nEdge[1].append(v)
  def oppositeDir(des):
    return list(itertools.permutations(des, 2))
  def sameDir(u, v):
    return [u, v]
  oneHop = {}
  for u, v in zip(*edge):
    if u not in oneHop:
      oneHop[u] = []
    oneHop[u].append(v)
  for orig, des in oneHop.items():
    push2Edge(oppositeDir(des), 1)
    for d in des:
      if d in oneHop:
        for hop1 in oneHop[d]:
          push2Edge(sameDir(d, hop1), 0)
  nLabel = ['' for _ in nEdge[0]]
  for e, (u, v) in enumerate(zip(*nEdge)):
    nLabel[e] = nText[u] + nText[v]
  assert len(nLabel) == len(nEdge[0])
  return nLabel, nEdge