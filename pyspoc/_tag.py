from enum import Enum


class Tag(Enum):
    ASSOCIATION = "association"
    CLUSTERING = "clustering"
    DIRECTED = "directed"
    DISTANCE = "distance"
    DISTRIBUTION = "distribution"
    FRACTAL = "fractal"
    GAUSSIAN = "gaussian"
    PARAMETRIC = "parametric"
    NONLINEAR = "nonlinear"
    NONPARAMETRIC = "nonparametric"
    ORDERED = "ordered"
    SIGNED = "signed"
    SIGNAL_PROCESSING = "signal"
    SYMMETRICAL = "symmetrical"
    GRAPH_DIRECTED = "digraph"
    GRAPH_MIXED = "graph-mixed"
    GRAPH_ACYCLIC = "acyclic"
    GRAPH_TREE = "tree"
    GRAPH_FOREST = "forest"
    GRAPH_WEIGHTED = "weighted"
    GRAPH = "graph"


class Signature(Enum):
    BLOBBINESS = "blobbiness"
    CAUSAL = "causal"
    DENSITY = "density"
    DIMENSIONALITY = "dimensionality"
    NOISE = "noise"
    NONLINEARITY = "nonlinearity"
    SYMMETRY = "symmetry"
    TOPOLOGY = "topology"


class Type(Enum):
    DESCRIPTIVE = "descriptive"
    ERROR = "error"
    LIKELIHOOD = "likelihood"

