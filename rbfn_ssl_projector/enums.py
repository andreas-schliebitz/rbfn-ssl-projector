from enum import Enum


class Architecture(str, Enum):
    SIMCLR = "simclr"
    SIMSIAM = "simsiam"
    MOCO = "moco"
    BYOL = "byol"
    SWAV = "swav"


class ProjectionHead(str, Enum):
    DEFAULT = "default"
    RBFN = "rbfn"
