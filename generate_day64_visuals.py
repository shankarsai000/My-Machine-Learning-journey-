"""Generate LinkedIn visuals for Day 64: Docker, CI/CD & Cloud Deployment."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.style.use("seaborn-v0_8-whitegrid")

NAVY = "#1B2A4A"
BLUE = "#2E86AB"
TEAL = "#17A398"
GREEN = "#18A558"
ORANGE = "#F18F01"
RED = "#D62828"
PURPLE = "#6C4AB6"
GRAY = "#6C757D"


def stage_box(ax, x, y, w, h, title, subtitle, color, title_fs=13, sub_fs=9.5):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=2,
        edgecolor=color,
        facecolor=color,
        alpha=0.12,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=title_fs,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        x + w / 2,
        y + h * 0.30,
        subtitle,
        ha="center",
        va="center",
        fontsize=sub_fs,
        color=GRAY,
    )


def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=2.5):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=22,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


# ============================================================
# VISUAL 1: End-to-end architecture diagram
# ============================================================
fig, ax = plt.subplots(figsize=(16, 7.2))
ax.set_xlim(0, 16)
ax.set_ylim(0, 7.2)
ax.axis("off")

ax.text(
    8,
    6.75,
    "Day 64: From Laptop to Live ML API",
    ha="center",
    fontsize=22,
    fontweight="bold",
    color=NAVY,
)
ax.text(
    8,
    6.30,
    "Docker · GitHub Actions · Artifact Registry · Cloud Run",
    ha="center",
    fontsize=13,
    color=GRAY,
)

# Row 1: developer -> github -> actions
stage_box(ax, 0.4, 4.4, 2.4, 1.5, "Developer", "Python · FastAPI\nML model · Git", BLUE)
stage_box(ax, 3.6, 4.4, 2.4, 1.5, "GitHub", "Branches · commits\nPull Requests", PURPLE)
stage_box(ax, 6.8, 4.4, 3.0, 1.5, "GitHub Actions", "Lint → Test → Build\nScan → Publish → Deploy", ORANGE)

# Row 2 continuation: registry -> cloud run -> live api
stage_box(ax, 10.6, 4.4, 2.6, 1.5, "Artifact Registry", "Versioned\nDocker images", TEAL)
stage_box(ax, 13.8, 4.4, 1.8, 1.5, "Cloud Run", "HTTPS · Scaling\nLogs · Revisions", GREEN)

# bottom row: live api + feedback arrows
stage_box(ax, 10.6, 1.2, 5.0, 1.6, "Live ML API", "Prediction requests from\nweb & mobile applications", GREEN, title_fs=14)

# arrows row 1
arrow(ax, 2.8, 5.15, 3.6, 5.15, color=BLUE)
arrow(ax, 6.0, 5.15, 6.8, 5.15, color=PURPLE)
arrow(ax, 9.8, 5.15, 10.6, 5.15, color=ORANGE)
arrow(ax, 13.2, 5.15, 13.8, 5.15, color=TEAL)

# actions -> registry (image push)
arrow(ax, 8.3, 4.4, 8.3, 2.8, color=ORANGE)
ax.text(8.55, 3.55, "push image", fontsize=9, color=ORANGE, rotation=90, va="center")

# registry -> cloud run (deploy)
arrow(ax, 11.9, 4.4, 11.9, 2.8, color=TEAL)
ax.text(12.05, 3.55, "deploy", fontsize=9, color=TEAL, rotation=90, va="center")

# cloud run -> live api
arrow(ax, 14.7, 4.4, 14.7, 2.8, color=GREEN)
ax.text(14.85, 3.55, "traffic", fontsize=9, color=GREEN, rotation=90, va="center")

# annotation: key lesson
ax.text(
    0.4,
    0.55,
    "Repeatable deployment pipeline — not a one-time manual upload.\n"
    "Train the model → serve it → test · package · secure · deploy · monitor · roll back.",
    fontsize=11.5,
    color=NAVY,
    style="italic",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3CD", edgecolor=ORANGE, alpha=0.9),
)

plt.tight_layout()
plt.savefig(
    "day64_architecture_diagram.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
print("[OK] Saved: day64_architecture_diagram.png")

# ============================================================
# VISUAL 2: CI/CD pipeline stages
# ============================================================
fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
ax.axis("off")

ax.text(
    8,
    5.55,
    "The Automated Pipeline — Every Push, Every PR",
    ha="center",
    fontsize=21,
    fontweight="bold",
    color=NAVY,
)

stages = [
    ("Push / PR", "code change", BLUE),
    ("Lint", "ruff", PURPLE),
    ("Test", "pytest · 4 tests", TEAL),
    ("Build", "Docker image", ORANGE),
    ("Scan", "Trivy · SBOM", RED),
    ("Publish", "Artifact Registry", TEAL),
    ("Deploy", "Cloud Run", GREEN),
]

w, h, gap, y = 2.0, 1.7, 0.28, 2.4
x = 0.35
for i, (title, sub, color) in enumerate(stages):
    stage_box(ax, x, y, w, h, title, sub, color, title_fs=13, sub_fs=9.5)
    if i < len(stages) - 1:
        arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2, color=color, lw=2.2)
    x += w + gap

# guard rails below
ax.text(
    8,
    1.15,
    "PRs test but NEVER deploy  •  deploy job waits for tests  •  deploy by immutable image digest",
    ha="center",
    fontsize=11.5,
    color=NAVY,
    bbox=dict(boxstyle="round,pad=0.45", facecolor="#E8F4FD", edgecolor=BLUE, alpha=0.9),
)

plt.tight_layout()
plt.savefig("day64_cicd_pipeline.png", dpi=300, bbox_inches="tight", facecolor="white")
print("[OK] Saved: day64_cicd_pipeline.png")

# ============================================================
# VISUAL 3: What changed — before vs after
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
fig.suptitle(
    "Day 64: Manual Deploy → Engineering Pipeline",
    fontsize=20,
    fontweight="bold",
    color=NAVY,
    y=0.98,
)

left = [
    "BUT IT WORKS ON MY MACHINE",
    "",
    "• Python version mismatch",
    "• Missing dependencies",
    "• Different OS libraries",
    "• Different env variables",
    "• Model won't load",
    "",
    "Manual: build → upload → deploy → pray",
]
right = [
    "EVERY CHANGE, AUTOMATICALLY",
    "",
    "• Docker packages app + runtime",
    "• CI lints, tests, builds image",
    "• Registry stores versioned images",
    "• Cloud Run serves HTTPS + scales",
    "• OIDC = no long-lived keys",
    "",
    "Pipeline: push → live in minutes",
]

for ax, lines, title, color in [
    (ax1, left, "BEFORE: Laptop Deployment", RED),
    (ax2, right, "AFTER: Platform Engineering", GREEN),
]:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.5,
        0.93,
        title,
        ha="center",
        fontsize=15,
        fontweight="bold",
        color=color,
        transform=ax.transAxes,
    )
    ax.text(
        0.06,
        0.82,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=11.5,
        color=NAVY,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor=color, linewidth=2),
    )

plt.tight_layout()
plt.savefig("day64_before_after.png", dpi=300, bbox_inches="tight", facecolor="white")
print("[OK] Saved: day64_before_after.png")

print("\n=== Day 64 LinkedIn Visuals Generated ===")
print("  1. day64_architecture_diagram.png")
print("  2. day64_cicd_pipeline.png")
print("  3. day64_before_after.png")
