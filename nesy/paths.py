"""Fixed locations. Raw datasets are read in place and never copied."""
from pathlib import Path

DATA = Path("/scratch/prj/bhi_zihe_imaging/mimic_cxr_full")
MIMIC = DATA / "mimic-cxr-jpg/mimic-cxr-jpg-2.1.0.physionet.org"
CHEXMASK = DATA / "CheXmask"
IMAGENOME = DATA / "Chest_ImaGenome"
MSCXR = DATA / "MS_CXR"
VINDR = DATA / "VinDr_cxr"
PADCHEST_GR = DATA / "PadChest_GR"

V2 = Path(__file__).resolve().parents[1]
MANIFESTS = V2 / "data/manifests"
SPLITS = V2 / "data/splits"
FEATURES = V2 / "data/features"
