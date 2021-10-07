import torch

class AutoLabel:
  def __init__(self, label_type = None):
    self.dep_labels = ['case', 'nsubj', 'npadvmod', 'intj', 'cc', 'auxpass', 'prt', 'pobj', 'dative', 
    'aux', 'acl', 'compound', 'oprd', 'advcl', 'xcomp', 'conj', 'ccomp', 'csubj', 'expl', 'preconj', 
    'agent', 'mark', 'nsubjpass', 'ROOT', 'poss', 'nummod', 'dobj', 'advmod', 'neg', 'acomp', 'prep',
    'amod', 'predet', 'appos', 'punct', 'relcl', 'pcomp', 'nmod', 'attr', 'dep', 'det', 'parataxis', 'quantmod',
    "meta", "csubjpass"]
    self.ucca_labels = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'L', 'N', 'P', 'R', 'S', 'U']
    self.label_type = label_type
    self.labels = {"UCCA": self.ucca_labels, "DEP": self.dep_labels}
    self.label_dict = {} # total 13 labels
    self.setupDict()

  def length(self):
    return len(self.label_dict)

  def setupDict(self):
    if self.label_type != None:
      self.labels = self.labels[self.label_type]
      for label in (self.labels):
        self.pushToDict(label)

  def pushToDict(self, label):
    if label not in self.label_dict:
      self.label_dict[label] = len(self.label_dict)

  def getIdx(self, label):
    assert label in self.label_dict, label
    return self.label_dict[label]

  def Label2Seq(self, label):
    if self.label_type == None:
      test_label = label[0][0]
      if test_label in self.ucca_labels:
        self.label_type = "UCCA"
      elif test_label in self.dep_labels:
        self.label_type = "DEP"
      
      assert self.label_type != None
      self.setupDict()

    label_list = []
    for l in label:
      label_list.append(torch.LongTensor(list(map(self.getIdx, l))))
    return label_list

class UCCALabel:
  def __init__(self):
    self.labels = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'L', 'N', 'P', 'R', 'S', 'U']
    self.label_dict = {} # total 13 labels
    self.setupDict()
  def length(self):
    return len(self.label_dict)
  def setupDict(self):
    for label in (self.labels):
      self.pushToDict(label)
  def pushToDict(self, label):
    if label not in self.label_dict:
      self.label_dict[label] = len(self.label_dict)
  def getIdx(self, label):
    assert label in self.label_dict, label
    return self.label_dict[label]
  def Label2Seq(self, label):
    label_list = []
    for l in label:
      label_list.append(torch.LongTensor(list(map(self.getIdx, l))))
    return label_list

class LineUCCALabel(UCCALabel):
    """
    1. This method follow Line Graph Enhanced AMR-to-Text
    2. Edge's label now becomes node representation 
    """
    def __init__(self):
        super().__init__()
        self.pad = "pad"
        self.pushToDict(self.pad)
    
    def getPadIndex(self):
        return self.label_dict[self.pad]
    def getIdx(self, label, max_len):
        idx = []
        for l in label:
            assert l in self.label_dict
            idx.append(self.label_dict[l])
        while len(idx) < max_len:
            idx.append(self.getPadIndex())
        return idx
    
    def Label2Seq(self, label, label_len, seq_len):
        """
        1. This function attempt for batched processing, thus need to be left-padded
        """
        label_list = [self.getIdx(item, label_len) for item in label]
        return torch.LongTensor(label_list)