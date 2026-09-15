import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings
warnings.filterwarnings('ignore')

# β-tin parameters (tetragonal)
a = 0.583  # nm
c = 0.318  # nm

# Atom coordinates (simplified)
# Primary atoms at corners and body center positions
atom_coords = np.array([
    [0.0, 0.0, 0.0],    # 000
    [0.5, 0.5, 0.5],    # ½½½ (body center)
    [1.0, 0.0, 0.0],    # 100
    [0.0, 1.0, 0.0],    # 010
    [0.0, 0.0, 1.0],    # 001
    [1.0, 1.0, 0.0],    # 110
    [1.0, 0.0, 1.0],    # 101
    [0.0, 1.0, 1.0],    # 011
    [1.0, 1.0, 1.0],    # 111
])

# Convert to Cartesian coordinates (nm)
atom_coords_cart = atom_coords.copy()
atom_coords_cart[:, 0] *= a
atom_coords_cart[:, 1] *= a
atom_coords_cart[:, 2] *= c

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# ===== Subplot 1: Unit Cell with Atoms =====
ax1 = fig.add_subplot(2, 3, 1, projection='3d')

# Draw unit cell edges
vertices = np.array([
    [0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0],
    [0, 0, c], [a, 0, c], [a, a, c], [0, a, c]
])

# Define cell edges
edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
    [4, 5], [5, 6], [6, 7], [7, 4],  # top face
    [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
]

for edge in edges:
    pts = vertices[edge]
    ax1.plot3D(*pts.T, 'b-', linewidth=2)

# Plot atoms
ax1.scatter(*atom_coords_cart.T, c='red', s=200, marker='o', alpha=0.8, edgecolors='darkred', linewidth=2)

# Label corner atoms
for i, coord in enumerate(atom_coords):
    label = f"({coord[0]:.1f},{coord[1]:.1f},{coord[2]:.1f})"
    ax1.text(atom_coords_cart[i, 0], atom_coords_cart[i, 1], atom_coords_cart[i, 2], label, fontsize=8)

ax1.set_xlabel(f'a = {a} nm', fontsize=10, fontweight='bold')
ax1.set_ylabel(f'a = {a} nm', fontsize=10, fontweight='bold')
ax1.set_zlabel(f'c = {c} nm', fontsize=10, fontweight='bold')
ax1.set_title('β-tin Unit Cell\n(Tetragonal, Body-Centered)', fontsize=12, fontweight='bold')
ax1.set_xlim(-0.05, a*1.1)
ax1.set_ylim(-0.05, a*1.1)
ax1.set_zlim(-0.05, c*1.1)

# ===== Subplot 2: Atom Distribution =====
ax2 = fig.add_subplot(2, 3, 2, projection='3d')

# Color atoms by type: corner atoms (blue) vs body-center (red)
corner_atoms = atom_coords_cart[atom_coords_cart[:, 2] < c*0.6]  # z < 0.6c
center_atoms = atom_coords_cart[atom_coords_cart[:, 2] >= c*0.6]  # z >= 0.6c

ax2.scatter(*corner_atoms.T, c='blue', s=150, marker='s', alpha=0.7, label='Corner/Edge Atoms', edgecolors='navy')
ax2.scatter(*center_atoms.T, c='red', s=150, marker='o', alpha=0.7, label='Body-Center Atoms', edgecolors='darkred')

# Draw cell box
for edge in edges:
    pts = vertices[edge]
    ax2.plot3D(*pts.T, 'k--', linewidth=1, alpha=0.3)

ax2.set_xlabel('X (nm)', fontsize=10)
ax2.set_ylabel('Y (nm)', fontsize=10)
ax2.set_zlabel('Z (nm)', fontsize=10)
ax2.set_title('Atom Type Distribution', fontsize=12, fontweight='bold')
ax2.legend()
ax2.set_xlim(-0.05, a*1.1)
ax2.set_ylim(-0.05, a*1.1)
ax2.set_zlim(-0.05, c*1.1)

# ===== Subplot 3: Top View (XY plane) =====
ax3 = fig.add_subplot(2, 3, 3)

ax3.add_patch(plt.Rectangle((0, 0), a, a, fill=False, edgecolor='blue', linewidth=2))
ax3.scatter(atom_coords_cart[:, 0], atom_coords_cart[:, 1], c='red', s=200, marker='o', alpha=0.8, edgecolors='darkred', linewidth=2)

for i, coord in enumerate(atom_coords):
    ax3.annotate(f"({coord[0]:.1f},{coord[1]:.1f})", 
                (atom_coords_cart[i, 0], atom_coords_cart[i, 1]), 
                fontsize=8, ha='center')

ax3.set_xlabel(f'X (a = {a} nm)', fontsize=10, fontweight='bold')
ax3.set_ylabel(f'Y (a = {a} nm)', fontsize=10, fontweight='bold')
ax3.set_title('Top View (XY Plane)', fontsize=12, fontweight='bold')
ax3.set_xlim(-0.1, a*1.2)
ax3.set_ylim(-0.1, a*1.2)
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal')

# ===== Subplot 4: Front View (XZ plane) =====
ax4 = fig.add_subplot(2, 3, 4)

ax4.add_patch(plt.Rectangle((0, 0), a, c, fill=False, edgecolor='blue', linewidth=2))
ax4.scatter(atom_coords_cart[:, 0], atom_coords_cart[:, 2], c='red', s=200, marker='o', alpha=0.8, edgecolors='darkred', linewidth=2)

for i, coord in enumerate(atom_coords):
    ax4.annotate(f"({coord[0]:.1f},{coord[2]:.1f})", 
                (atom_coords_cart[i, 0], atom_coords_cart[i, 2]), 
                fontsize=8, ha='center')

ax4.set_xlabel(f'X (a = {a} nm)', fontsize=10, fontweight='bold')
ax4.set_ylabel(f'Z (c = {c} nm)', fontsize=10, fontweight='bold')
ax4.set_title('Front View (XZ Plane)', fontsize=12, fontweight='bold')
ax4.set_xlim(-0.1, a*1.2)
ax4.set_ylim(-0.05, c*1.2)
ax4.grid(True, alpha=0.3)

# ===== Subplot 5: Side View (YZ plane) =====
ax5 = fig.add_subplot(2, 3, 5)

ax5.add_patch(plt.Rectangle((0, 0), a, c, fill=False, edgecolor='blue', linewidth=2))
ax5.scatter(atom_coords_cart[:, 1], atom_coords_cart[:, 2], c='red', s=200, marker='o', alpha=0.8, edgecolors='darkred', linewidth=2)

for i, coord in enumerate(atom_coords):
    ax5.annotate(f"({coord[1]:.1f},{coord[2]:.1f})", 
                (atom_coords_cart[i, 1], atom_coords_cart[i, 2]), 
                fontsize=8, ha='center')

ax5.set_xlabel(f'Y (a = {a} nm)', fontsize=10, fontweight='bold')
ax5.set_ylabel(f'Z (c = {c} nm)', fontsize=10, fontweight='bold')
ax5.set_title('Side View (YZ Plane)', fontsize=12, fontweight='bold')
ax5.set_xlim(-0.1, a*1.2)
ax5.set_ylim(-0.05, c*1.2)
ax5.grid(True, alpha=0.3)

# ===== Subplot 6: Atomic Distances =====
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')

# Calculate some key distances
distances_text = f"""
β-TIN UNIT CELL PARAMETERS
─────────────────────────────
Crystal System: Tetragonal (Body-Centered)
a = {a:.3f} nm
c = {c:.3f} nm
c/a ratio = {c/a:.3f}

ATOMIC INFORMATION
─────────────────────────────
Total atoms in unit cell: {len(atom_coords)}
Corner atoms: 8 (shared with 8 cells)
Body-center atoms: 1 (unique)

CALCULATED DISTANCES
─────────────────────────────
Cell edge (a): {a:.3f} nm
Cell height (c): {c:.3f} nm
Cell volume: {a*a*c:.6f} nm³

Nearest neighbor distance:
  (0,0,0) to (½,½,½): {np.sqrt((a/2)**2 + (a/2)**2 + (c/2)**2):.4f} nm

Space diagonal:
  (0,0,0) to (a,a,c): {np.sqrt(a**2 + a**2 + c**2):.4f} nm
"""

ax6.text(0.05, 0.95, distances_text, transform=ax6.transAxes, 
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('beta_tin_unit_cell.png', dpi=300, bbox_inches='tight')
print("✓ Gambar disimpan sebagai 'beta_tin_unit_cell.png'")
plt.show()

# Print atomic coordinates
print("\n" + "="*60)
print("KOORDINAT ATOM β-TIN (FRACTIONAL)")
print("="*60)
print(f"{'No':<3} {'X':<6} {'Y':<6} {'Z':<6} {'X(nm)':<10} {'Y(nm)':<10} {'Z(nm)':<10}")
print("-"*60)
for i, (frac, cart) in enumerate(zip(atom_coords, atom_coords_cart)):
    print(f"{i+1:<3} {frac[0]:<6.1f} {frac[1]:<6.1f} {frac[2]:<6.1f} {cart[0]:<10.4f} {cart[1]:<10.4f} {cart[2]:<10.4f}")
print("="*60)
