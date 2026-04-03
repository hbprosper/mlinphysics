import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim(0, 15)
ax.set_ylim(-0.8, 8.5)
ax.axis('off')
fig.patch.set_facecolor('#FAFAF8')

# ── palette ───────────────────────────────────────────────────────────────
COL_PROMPT_BG = '#E1F5EE';  COL_PROMPT_BD = '#0F6E56'
COL_SEP_BG    = '#F1EFE8';  COL_SEP_BD    = '#5F5E5A'
COL_TGT_BG    = '#EEEDFE';  COL_TGT_BD    = '#534AB7'
COL_DEL_BG    = '#FCEBEB';  COL_DEL_BD    = '#A32D2D'
COL_DEL_LINE  = '#C0392B'
COL_ARROW     = '#534AB7'
COL_LABEL     = '#26215C'
COL_VOC       = '#085041'

# ── geometry ──────────────────────────────────────────────────────────────
CELL_W  = 0.95
CELL_H  = 0.95
N_ROWS  = 5
N_COLS  = 7
GAP     = 0.85          # horizontal gap between the two grids
LEG_GAP = 0.65          # gap between right grid and legend

LEFT_X0 = 2.10
TOP_Y0  = 5.80

token_labels = ['<sos>', 'p', '<sep>', 't₁', '<eos>']
row_styles   = ['prompt', 'prompt', 'sep', 'target', 'delete']

# ── helpers ───────────────────────────────────────────────────────────────
def draw_cell(x0, y0, bg, bd, lw=1.5):
    rect = mpatches.FancyBboxPatch(
        (x0, y0), CELL_W, CELL_H,
        boxstyle="round,pad=0.05",
        facecolor=bg, edgecolor=bd, linewidth=lw, zorder=2)
    ax.add_patch(rect)

def draw_cross(x0, y0, color, lw=2.2):
    pad = 0.14
    for ys, ye in [((y0+pad, y0+CELL_H-pad), (y0+CELL_H-pad, y0+pad))]:
        ax.plot([x0+pad, x0+CELL_W-pad], list(ys),
                color=color, lw=lw, zorder=4, solid_capstyle='round')
        ax.plot([x0+pad, x0+CELL_W-pad], list(ye),
                color=color, lw=lw, zorder=4, solid_capstyle='round')

# ── left grid (5 × 7) ────────────────────────────────────────────────────
style_colors = {
    'prompt': (COL_PROMPT_BG, COL_PROMPT_BD),
    'sep':    (COL_SEP_BG,    COL_SEP_BD),
    'target': (COL_TGT_BG,    COL_TGT_BD),
    'delete': (COL_DEL_BG,    COL_DEL_BD),
}
for row in range(N_ROWS):
    bg, bd = style_colors[row_styles[row]]
    y0 = TOP_Y0 - row * CELL_H - CELL_H
    for col in range(N_COLS):
        x0 = LEFT_X0 + col * CELL_W
        draw_cell(x0, y0, bg, bd)
        if row_styles[row] == 'delete':
            draw_cross(x0, y0, COL_DEL_LINE)

# ── token labels (left) ───────────────────────────────────────────────────
style_tc = {s: c[1] for s, c in style_colors.items()}
for row, (tok, style) in enumerate(zip(token_labels, row_styles)):
    y_mid = TOP_Y0 - row * CELL_H - CELL_H / 2
    ax.text(LEFT_X0 - 0.18, y_mid, tok,
            ha='right', va='center', fontsize=15,
            color=style_tc[style], fontfamily='monospace', fontweight='bold')

# ── vocab labels (above left grid) ───────────────────────────────────────
for col in range(N_COLS):
    x_mid = LEFT_X0 + col * CELL_W + CELL_W / 2
    ax.text(x_mid, TOP_Y0 + 0.28, f'v{col+1}',
            ha='center', va='bottom', fontsize=15,
            color=COL_VOC, fontweight='bold')

# ── "Output logits" header ────────────────────────────────────────────────
ax.text(LEFT_X0 + N_COLS * CELL_W / 2, TOP_Y0 + 0.82,
        'Output logits',
        ha='center', va='bottom', fontsize=16,
        color=COL_LABEL, fontweight='bold')

# ── right grid (5 × 1) ───────────────────────────────────────────────────
RIGHT_X0 = LEFT_X0 + N_COLS * CELL_W + GAP

for row in range(N_ROWS):
    y0 = TOP_Y0 - row * CELL_H - CELL_H
    # row 0: no logit points here → cross
    # rows 1-4: valid targets (including row 4 = <eos>) → purple
    if row == 0:
        bg, bd, cross = COL_DEL_BG, COL_DEL_BD, True
    else:
        bg, bd, cross = COL_TGT_BG, COL_TGT_BD, False
    draw_cell(RIGHT_X0, y0, bg, bd)
    if cross:
        draw_cross(RIGHT_X0, y0, COL_DEL_LINE)

# ── "Targets" header ──────────────────────────────────────────────────────
ax.text(RIGHT_X0 + CELL_W / 2, TOP_Y0 + 0.28,
        'Targets',
        ha='center', va='bottom', fontsize=15,
        color=COL_LABEL, fontweight='bold')

# ── arrows (row r → target row r+1, slope ≈ -1) ──────────────────────────
for row in range(N_ROWS - 1):          # rows 0 .. 3
    src_x = LEFT_X0 + N_COLS * CELL_W
    src_y = TOP_Y0 - row * CELL_H - CELL_H / 2
    tgt_x = RIGHT_X0
    tgt_y = TOP_Y0 - (row + 1) * CELL_H - CELL_H / 2

    style = row_styles[row]
    if style in ('prompt', 'sep'):
        color, lw, alpha = COL_PROMPT_BD, 1.8, 0.70
    else:
        color, lw, alpha = COL_ARROW,     2.2, 1.00

    ax.annotate('', xy=(tgt_x, tgt_y), xytext=(src_x, src_y),
                arrowprops=dict(
                    arrowstyle='->', color=color,
                    lw=lw, mutation_scale=16,
                    connectionstyle='arc3,rad=0.0'),
                alpha=alpha, zorder=5)

# ── legend (right of right grid) ─────────────────────────────────────────
LEG_X0   = RIGHT_X0 + CELL_W + LEG_GAP
LEG_Y0   = TOP_Y0 - 0.10        # start near the top of the grid
SWATCH_W = 0.40
SWATCH_H = 0.34
LINE_GAP = 0.62

legend_items = [
    (COL_PROMPT_BG, COL_PROMPT_BD, 'Prompt tokens'),
    (COL_SEP_BG,    COL_SEP_BD,    'Separator token'),
    (COL_TGT_BG,    COL_TGT_BD,    'Active logits / targets'),
    (COL_DEL_BG,    COL_DEL_BD,    'No loss (deleted)'),
]

for bg, bd, label in legend_items:
    rect = mpatches.FancyBboxPatch(
        (LEG_X0, LEG_Y0 - SWATCH_H / 2), SWATCH_W, SWATCH_H,
        boxstyle="round,pad=0.04",
        facecolor=bg, edgecolor=bd, linewidth=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(LEG_X0 + SWATCH_W + 0.18, LEG_Y0,
            label, ha='left', va='center',
            fontsize=14, color='#2C2C2A')
    LEG_Y0 -= LINE_GAP

plt.tight_layout(pad=0.5)
plt.savefig('/mnt/user-data/outputs/logits_grid.png',
            dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("Saved.")
