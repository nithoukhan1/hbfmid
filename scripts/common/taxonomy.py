"""Class names, clinical families and sibling definitions for HBFMID (data.yaml order)."""

CLASS_NAMES = ['Comminuted', 'Greenstick', 'Healthy', 'Linear', 'Oblique Displaced',
               'Oblique', 'Segmental', 'Spiral', 'Transverse Displaced', 'Transverse']
NC = len(CLASS_NAMES)

# Family = parent node in the FracTree (Healthy | Fracture -> Simple / Multifragmentary / Incomplete)
FAMILY = {
    'Comminuted': 'multifragmentary', 'Segmental': 'multifragmentary',
    'Greenstick': 'incomplete', 'Healthy': 'healthy',
    'Linear': 'simple', 'Oblique': 'simple', 'Oblique Displaced': 'simple',
    'Spiral': 'simple', 'Transverse': 'simple', 'Transverse Displaced': 'simple',
}
# Same bone-line pattern, differing only by displacement
BASE = {'Oblique Displaced': 'Oblique', 'Transverse Displaced': 'Transverse'}

# Classes with >= 10 independent groups (see results/eda/class_group_stats.csv). Gate B metric uses only these.
ELIGIBLE_IDS = [0, 1, 2, 4, 5, 7, 8, 9]


def family_of(i):
    return FAMILY[CLASS_NAMES[i]]


def base_of(i):
    n = CLASS_NAMES[i]
    return BASE.get(n, n)


def is_sibling(i, j):
    """True if i != j and both are in the same family (same parent), e.g. Transverse vs Oblique."""
    return i != j and family_of(i) == family_of(j)


def is_same_base(i, j):
    """True if i != j and the two differ only by displacement (Transverse vs Transverse Displaced)."""
    return i != j and base_of(i) == base_of(j)
