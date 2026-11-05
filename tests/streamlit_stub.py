"""Minimal fake `streamlit` + `pydeck` so app.py can be executed headlessly."""
import sys, types
from unittest.mock import MagicMock

PRESSED = []          # button labels (or prefixes) that return True this run
OVERRIDES = {}        # widget label -> forced return value
CALLS = []            # (function name, label) log for assertions
CLICK = {}            # {"route_points": [{"distance_m": ...}]} = what the user "clicked"


class State(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v


ST = None


class Ctx:
    """Stands in for columns / tabs / sidebar: same widgets as `st` itself."""
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name): return getattr(ST, name)


def build():
    global ST
    st = ST = types.ModuleType("streamlit")
    st.__version__ = "stub"
    st.session_state = State()

    def over(label, default):
        return OVERRIDES.get(label, default)

    def button(label, *a, **k):
        CALLS.append(("button", label))
        return any(label.startswith(p) for p in PRESSED)

    st.button = button
    def slider(label, *a, **k):
        if k.get("key") and st.session_state.get(k["key"]) is not None:
            return st.session_state[k["key"]]          # widgets with a key read session state
        return over(label, k.get("value", a[2] if len(a) > 2 else (a[0] if a else 0)))

    st.slider = slider
    st.number_input = lambda label, *a, **k: over(label, k.get("value", a[2] if len(a) > 2 else 0))
    st.checkbox = lambda label, value=False, **k: over(label, value)
    st.text_input = lambda label, value="", **k: over(label, value)
    st.selectbox = lambda label, options, index=0, **k: over(label, list(options)[index] if len(list(options)) else None)
    st.radio = lambda label, options, index=0, **k: over(label, list(options)[index])
    st.data_editor = lambda df, **k: df
    st.file_uploader = lambda *a, **k: None
    st.tabs = lambda labels: [Ctx() for _ in labels]
    st.columns = lambda spec, **k: [Ctx() for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.sidebar = Ctx()
    st.expander = lambda *a, **k: Ctx()
    st.spinner = lambda *a, **k: Ctx()
    st.dialog = lambda *a, **k: (lambda fn: fn)
    st.cache_data = lambda *a, **k: (lambda fn: fn) if not a or not callable(a[0]) else a[0]
    st.column_config = MagicMock()
    class Event:
        def __init__(self, objects): self.selection = {"objects": objects, "indices": {}}

    def pydeck_chart(deck, **k):
        CALLS.append(("pydeck_chart", k.get("key")))
        return Event(dict(CLICK)) if k.get("on_select") else None

    st.pydeck_chart = pydeck_chart
    st.metric = lambda label, value, *a, **k: CALLS.append(("metric", label, value))
    st.dataframe = lambda data, **k: CALLS.append(("dataframe", len(data)))
    st.__getattr__ = None
    def fallback(name):
        return lambda *a, **k: Ctx()
    st.__getattr__ = fallback  # PEP 562: any other st.xxx() is a no-op
    return st


def install():
    st = build()
    sys.modules["streamlit"] = st
    sys.modules["pydeck"] = MagicMock()
    return st
