"""
Generalization animation for PHY 6937 - "Machine Learning in Physics".

Story:  data -> order 2 (underfit) -> order 10 (overfit) -> true-risk column
              -> order 3 (good) -> true curve

Point: the empirical risk is the only landscape we can descend, but the minimum
we actually want is the minimum of the infinite-data risk. Models that manage
that are said to generalize.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D
from numpy.polynomial import Polynomial

# ----------------------------------------------------------------- palette ---
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#dedcd6"
C_TRAIN = "#184f95"     # training data
C_UNDER = "#eb6834"     # order 2
C_OVER  = "#4a3aa7"     # order 10
C_GOOD  = "#2a78d6"     # order 3
C_TRUE  = INK

FPS = 25
XLO, XHI = 0.0, 16.0
YLO, YHI = -0.055, 0.155

# -------------------------------------------------------------- the truth ---
_K, _C = -0.00127, 0.058          # cubic: local min at x=3, local max at x=11.5


def true_curve(x):
    return _K * (x**3 / 3.0 - 7.25 * x**2 + 34.5 * x) + _C


SIGMA = 0.013
rng = np.random.default_rng(155)

x_train = np.linspace(XLO, XHI, 11)
y_train = true_curve(x_train) + rng.normal(0, SIGMA, x_train.size)

xg = np.linspace(XLO, XHI, 800)
xfine = np.linspace(XLO, XHI, 6001)
y_true_g = true_curve(xg)

MODELS = {
    2:  dict(color=C_UNDER, label="order 2"),
    10: dict(color=C_OVER,  label="order 10"),
    3:  dict(color=C_GOOD,  label="order 3"),
}
for d, m in MODELS.items():
    p = Polynomial.fit(x_train, y_train, d)
    m["poly"] = p
    m["yg"] = p(xg)
    m["emp"] = np.mean((p(x_train) - y_train) ** 2) * 1e3
    # risk we would get with infinite data: bias^2 integrated + noise floor
    m["true"] = (np.mean((p(xfine) - true_curve(xfine)) ** 2) + SIGMA**2) * 1e3

ORDER = [2, 10, 3]


def fmt(v):
    if v >= 1e4:
        return f"{v:.0e}"
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


# ------------------------------------------------------------------ figure ---
fig = plt.figure(figsize=(12, 7.0), dpi=120)
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(2, 1, height_ratios=[0.34, 1.0],
                      left=0.075, right=0.975, top=0.890, bottom=0.160,
                      hspace=0.10)
cap = fig.add_subplot(gs[0]); cap.axis("off")
cap.set_xlim(0, 1); cap.set_ylim(0, 1)
ax = fig.add_subplot(gs[1])

fig.suptitle("Empirical risk is what we can measure — generalization is what we want",
             fontsize=20, color=INK, y=0.966, fontweight="bold")

ax.set_xlim(XLO, XHI); ax.set_ylim(YLO, YHI)
ax.set_xlabel("$x$", fontsize=19, color=INK, labelpad=2)
ax.set_ylabel("$y$", fontsize=19, color=INK, rotation=0, labelpad=18)
ax.tick_params(labelsize=15, colors=MUTED, length=4)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.grid(True, color=GRID, lw=0.8, alpha=0.7)
ax.set_axisbelow(True)

# ----------------------------------------------------------------- artists ---
sc_train = ax.plot([], [], "o", ms=10, mfc=C_TRAIN, mec="white", mew=1.4,
                   zorder=7)[0]
ln_true = ax.plot([], [], "-", color=C_TRUE, lw=3.2, zorder=6)[0]
lines = {d: ax.plot([], [], "--", color=MODELS[d]["color"], lw=3.0, zorder=4,
                    dashes=(5, 3))[0] for d in ORDER}

cap_main = cap.text(0.0, 0.74, "", ha="left", va="center", fontsize=19.5,
                    color=INK)
cap_sub = cap.text(0.0, 0.26, "", ha="left", va="center", fontsize=16,
                   color=MUTED)

# risk table: three fixed columns so the empirical value can be bold on its own
X_LAB, X_EMP, X_TRU = 0.645, 0.845, 1.0
ROW_Y = (0.52, 0.30, 0.08)

tbl_units = cap.text(X_TRU, 1.02, "", ha="right", va="center", fontsize=13,
                     color=MUTED)
head_emp = cap.text(X_EMP, 0.78, "", ha="right", va="center", fontsize=13,
                    family="monospace", color=MUTED)
head_tru = cap.text(X_TRU, 0.78, "", ha="right", va="center", fontsize=13,
                    family="monospace", color=MUTED)
tbl_lab = [cap.text(X_LAB, y, "", ha="left", va="center", fontsize=17,
                    family="monospace", color=INK) for y in ROW_Y]
tbl_emp = [cap.text(X_EMP, y, "", ha="right", va="center", fontsize=17,
                    family="monospace", color=INK, fontweight="bold")
           for y in ROW_Y]
tbl_tru = [cap.text(X_TRU, y, "", ha="right", va="center", fontsize=17,
                    family="monospace", color=INK) for y in ROW_Y]

_legend = {"obj": None}


def set_legend(handles):
    if _legend["obj"] is not None:
        _legend["obj"].remove()
        _legend["obj"] = None
    if handles:
        _legend["obj"] = fig.legend(
            handles=handles, loc="lower center", ncol=len(handles),
            frameon=False, fontsize=15, bbox_to_anchor=(0.5, 0.005),
            handlelength=2.2, columnspacing=1.5, handletextpad=0.6)


# ---------------------------------------------------------------- timeline ---
def block(name, n):
    return [(name, i, n) for i in range(n)]


TIMELINE = (
    block("data", 70)
    + block("p2", 45) + block("p2_hold", 60)
    + block("p10", 55) + block("p10_hold", 50)
    + block("infinite", 30) + block("infinite_hold", 70)
    + block("p3", 45) + block("p3_hold", 70)
    + block("truth", 50) + block("truth_hold", 60)
    + block("final", 85)
)
N_FRAMES = len(TIMELINE)
STAGES = list(dict.fromkeys(nm for nm, _, _ in TIMELINE))
STAGE_INDEX = {}
for i, (nm, k, n) in enumerate(TIMELINE):
    STAGE_INDEX.setdefault(nm, i)


def pos(name):
    return STAGES.index(name)


def ease(t):
    return t * t * (3 - 2 * t)


CAPTIONS = {
    "data": ("We only ever see a finite, noisy sample.",
             "Everything we compute is built from these 11 points."),
    "p2": ("Order 2: too rigid to follow the data.",
           "High risk on the data — and everywhere else. Underfitting."),
    "p10": ("Order 10: empirical risk is exactly zero.",
            "A perfect minimum of the landscape we can measure."),
    "infinite": ("Now ask what the risk would be with infinite data.",
                 "The models do not change — only what we ask of them."),
    "infinite_hold": ("Order 10 has the worst risk of the two.",
                      "Zero empirical risk, largest true risk. Overfitting."),
    "p3": ("Order 3: a worse empirical risk…",
           "…and a far smaller risk on data it has never seen."),
    "truth": ("The curve the data actually came from.",
              "Order 3 found it without ever being shown it."),
    "final": ("The goal is the minimum you cannot measure,",
              "reached by descending the one you can."),
}
CAPTIONS["p2_hold"] = CAPTIONS["p2"]
CAPTIONS["p10_hold"] = CAPTIONS["p10"]
CAPTIONS["p3_hold"] = CAPTIONS["p3"]
CAPTIONS["truth_hold"] = CAPTIONS["truth"]

REVEAL_AT = {2: "p2", 10: "p10", 3: "p3"}
FOCUS = {"p2": 2, "p2_hold": 2, "p10": 10, "p10_hold": 10,
         "infinite": 10, "infinite_hold": 10, "p3": 3, "p3_hold": 3,
         "final": 3}


def update(f):
    name, k, n = TIMELINE[f]
    t = (k + 1) / n
    here = pos(name)

    # data ------------------------------------------------------------------
    sc_train.set_data(x_train, y_train)   # frame 0 is already a usable still

    # model curves ----------------------------------------------------------
    focus = FOCUS.get(name)
    for d in ORDER:
        stg = REVEAL_AT[d]
        if here < pos(stg):
            lines[d].set_data([], [])
            continue
        if name == stg:
            j = max(2, int(ease(t) * xg.size))
            lines[d].set_data(xg[:j], MODELS[d]["yg"][:j])
        else:
            lines[d].set_data(xg, MODELS[d]["yg"])
        if focus is None:
            lines[d].set_alpha(0.9); lines[d].set_linewidth(2.6)
        elif d == focus:
            lines[d].set_alpha(1.0); lines[d].set_linewidth(3.2)
        else:
            lines[d].set_alpha(0.28); lines[d].set_linewidth(2.0)

    # true curve ------------------------------------------------------------
    if here >= pos("truth"):
        j = max(2, int(ease(t) * xg.size)) if name == "truth" else xg.size
        ln_true.set_data(xg[:j], y_true_g[:j])
    else:
        ln_true.set_data([], [])

    # captions --------------------------------------------------------------
    main, sub = CAPTIONS[name]
    cap_main.set_text(main)
    cap_sub.set_text(sub)

    # risk table ------------------------------------------------------------
    show_true_col = here >= pos("infinite_hold") or (
        name == "infinite" and t > 0.55)
    if here >= pos("p2"):
        tbl_units.set_text(r"risk  $\times\,10^{-3}$")
        head_emp.set_text("empirical")
        head_tru.set_text("infinite data" if show_true_col else "")
    else:
        tbl_units.set_text("")
        head_emp.set_text("")
        head_tru.set_text("")

    r = 0
    for d in ORDER:
        if here < pos(REVEAL_AT[d]):
            continue
        col = MODELS[d]["color"]
        tbl_lab[r].set_text(MODELS[d]["label"]); tbl_lab[r].set_color(col)
        tbl_emp[r].set_text(fmt(MODELS[d]["emp"])); tbl_emp[r].set_color(col)
        tbl_tru[r].set_text(fmt(MODELS[d]["true"]) if show_true_col else "")
        tbl_tru[r].set_color(col)
        r += 1
    for j in range(r, 3):
        tbl_lab[j].set_text(""); tbl_emp[j].set_text(""); tbl_tru[j].set_text("")

    # legend ----------------------------------------------------------------
    h = []
    if here >= pos("data"):
        h.append(Line2D([], [], marker="o", ls="none", ms=9, mfc=C_TRAIN,
                        mec="white", mew=1.2, label="data ($n=11$)"))
    for d in ORDER:
        if here >= pos(REVEAL_AT[d]):
            h.append(Line2D([], [], ls="--", lw=2.6, color=MODELS[d]["color"],
                            label=MODELS[d]["label"]))
    if here >= pos("truth"):
        h.append(Line2D([], [], ls="-", lw=2.8, color=C_TRUE,
                        label="true curve"))
    set_legend(h)
    return []


if __name__ == "__main__":
    anim = FuncAnimation(fig, update, frames=N_FRAMES,
                         interval=1000 / FPS, blit=False)
    anim.save("/home/claude/out/generalization.mp4",
              writer=FFMpegWriter(fps=FPS, bitrate=4500), dpi=120)
    print(f"mp4: {N_FRAMES} frames, {N_FRAMES/FPS:.1f} s")

    for nm in ("data", "p2_hold", "p10_hold", "infinite_hold", "p3_hold",
               "truth_hold", "final"):
        update(STAGE_INDEX[nm] + 8)
        fig.savefig(f"/home/claude/out/frame_{nm}.png", dpi=90)
    for d in ORDER:
        print(d, round(MODELS[d]["emp"], 3), round(MODELS[d]["true"], 3))
