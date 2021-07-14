import itertools

def Process2LineGraph(text, edge, label):
  nText = [None for _ in text]
  nEdge = [[], []]

  # Process Nodes (text)
  for i, (u, v) in enumerate(zip(*edge)):
    if nText[v] != None:
      nText[v] += label[i]
    else:
      nText[v] = label[i]

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
  return nText, nEdge