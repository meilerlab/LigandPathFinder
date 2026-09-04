import numpy as np

class BinTree:
    def __init__(self):
        self.root = None

    def unravel(self):
        if self.root:
            return self.root.unravel()
        else:
            return []

class Node:
    def __init__(self, init_xyz=None, xyz=None, parent=None, left=None, right=None, origin=None, Lm=None, Rm=None, id=None, init_pdb=None):
        self.init_xyz = init_xyz
        self.xyz = xyz
        self.parent = parent
        self.left = left
        self.right = right
        self.origin = origin
        self.Lm = Lm
        self.Rm = Rm
        self.id = id
        self.init_pdb = init_pdb

    def unravel(self):
        sequence = []
        if self.left:
            sequence.extend(self.left.unravel())
        sequence.append(self)
        if self.right:
            sequence.extend(self.right.unravel())
        return sequence

    def dright(self):
        p1 = self.xyz
        p2 = self.Rm.xyz
        return np.linalg.norm(p1 - p2)

    def dleft(self):
        p1 = self.xyz
        p2 = self.Lm.xyz
        return np.linalg.norm(p1 - p2)