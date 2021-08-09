import itertools

def Process2LineGraph(size, edge, label):
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