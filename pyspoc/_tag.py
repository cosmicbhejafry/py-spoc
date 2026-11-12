from enum import Enum


class Tag(Enum):
    ASSOCIATION = "association"
    CLUSTERING = "clustering"
    DIRECTED = "directed"
    DISTANCE = "distance"
    DISTRIBUTION = "distribution"
    EMPIRICAL = "empirical"
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
    GRAPH_ACYCLIC = "graph-acyclic"
    GRAPH_TREE = "graph-tree"
    GRAPH_FOREST = "graph-forest"
    GRAPH_WEIGHTED = "graph-weighted"
    GRAPH = "graph"


class Signature(Enum):
    ASSOCIATION = "association"
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

