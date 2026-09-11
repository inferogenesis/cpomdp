"""The registered operating point, stated once.

`research/gate_d4_registration.md` fixes these in its AMENDMENT of 2026-08-07 and the
DECLARATION of 2026-08-23. Two explorations and their tests read them, and a stale copy
in one of them would quote a different registered point for the same document, so they
live here and nowhere else in the package.
"""

import numpy as np

__all__ = [
    "BETA",
    "DECADES",
    "DECADES_FLOOR",
    "F_STAR",
    "F_STAR_SEXTIC",
    "KAPPA_MIN",
    "K_MIN",
    "LN10",
    "PUBLISHED_SIGMA_P",
    "SIGMA_P",
    "THRESHOLD",
]

LN10 = float(np.log(10.0))
"""One decade in natural-log units, the conversion the D2 fit's width needs."""

K_MIN = 10.0
"""The registered clearance floor: the gap clears the certified error tenfold."""

BETA = 0.05
"""The registered total budget on the fitted exponent's error."""

DECADES = 0.520
"""`D*`, the registered window width in decades, optimised at `k_min` and `β`."""

F_STAR = 0.0488
"""The registered edge fraction. Optimised against the quartic edge, and the
registration's AMENDMENT of 2026-08-23 records that it does not carry across to the
sextic one."""

SIGMA_P = 0.0359
"""The registered statistical term at `(k_min, D*)`."""

KAPPA_MIN = 0.1
"""The declared floor of the `κ` sweep, rationalised rather than derived."""

DECADES_FLOOR = 0.5
"""The floor on the window width, declared blind in the PRE-REGISTRATION of 2026-09-11,
and where `D*` landed."""

F_STAR_SEXTIC = 0.078157
"""`f*` under the sextic edge at `D = 0.5`, the RESULT of 2026-09-11. Not the quartic
`F_STAR` above, which does not carry across."""

THRESHOLD = 5.962165e-4
"""`T` in nats at `κ_min`, registered by the RESULT of 2026-09-11 under the declared
lattice. `c₂σ_min²` at the window's lower edge."""

PUBLISHED_SIGMA_P = (
    (5.0, 0.4343, 0.089),
    (10.0, 0.4343, 0.045),
    (30.0, 0.4343, 0.015),
    (K_MIN, DECADES, SIGMA_P),
)
"""Every `σ_p` the registration publishes, as `(k, D, σ_p)`. The first three sit at the
bias-only optimum its table used; the fourth is the operating point."""
