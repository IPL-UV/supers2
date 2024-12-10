from supers2.main import predict, setmodel, predict_large, predict_rgbnir, uncertainty
from supers2.xai.lam import lam
from supers2.trained_models import SRmodels

models = list(SRmodels.model_dump()["object"].keys())
