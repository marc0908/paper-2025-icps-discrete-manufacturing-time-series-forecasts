from .time_series_library import TimeXer, Crossformer, iTransformer, DLinear, TimesNet, TimeMixer, NonStationary, FEDformer, Informer, Autoformer, ETSformer
from .duet.duet import DUET  # Pfad ggf. an den realen Dateinamen anpassen

ADAPTER = {
    "transformer_adapter": "paper_icps.forecasting.baselines.time_series_library.transformer_adapter",
    # evtl. weitere Adapter-Typen
}

__all__ = [
    "TimeXer",
    "Crossformer",
    "iTransformer",
    "DLinear",
    "DUET",
    "TimesNet",
    "TimeMixer",
    "NonStationary",
    "FEDformer",
    "Informer",
    "Autoformer",
    "ETSformer"
]