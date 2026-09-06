"""
triangulate.py
--------------

Pure-geometry core for simple-polygon triangulation. No pygame, no rendering,
no app state. Everything here operates on a sequence of (x, y) pairs and
returns *indices into that sequence*, so the caller decides how to draw.

Coordinate-system note
    The maths is orientation-agnostic. `triangulate()` measures the signed
    area of the input and internally works on a canonically-oriented copy of
    the index ring, so the caller may hand in vertices in either winding order
    and in either y-up (maths) or y-down (screen) space.

Epsilon note
    Cross products of two edges have units of (coordinate unit)^2. The default
    epsilons assume pixel-scale coordinates (magnitudes ~1e0 .. 1e3). If you
    ever feed normalised coordinates (0..1), scale the epsilons down or
    normalise via `scale_epsilon()`.

On the Date: 05-09-2026

I added My Notes to understand how the code in `triangulate.py` works.

I managed to learn these:
1. How the mouse and key input can be made robust against multiple-frame triggers.
2. How the triangulate algorithm worked in general.
3. The `as_point` function that reshapes list of vertices or ndarrays properly. 
4. The `orient` function used to determine which side of a directed line a point lies (+ve/left, -ve/right, or 0/collinear).
5. The `same_point` function that checks if two points are the same, whether their x and y coordinates are within the limits of an EPSILON.
6. The `segment_cross` function that utilizes`:
    1) the `orient` function to check if segments/edges cross each other.
    2) the `_on_segment` function to check if a indeed lies on an edge, if it is collinear with that edge and exists within its AABB/rectangular bounds 
7. The `point_in_triangle` function that checks if a point is contained by a triangle, using the `orient` function
8. The `signed_area` function that uses the Shoelace Method/Gauss's formula. The area is used to determine if the winding of the polygon's points is anticlockwise or not.
9. The functions, `is_self_intersecting` (uses `segment_cross`) and `has_duplicate_points` (uses `same_point`), which
    find if the polygon has segments/edges that intersect/cross or if the polygon has duplicate points, in which if it is true
    for either case, makes the polygon invalid.
10. Finally, the `triangulate` algorithm uses `_best_ear` and `is_ear` functions to decompose the polygon specified by vertices into an ordering
    of triangles. 
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

# Non-degeneracy threshold for a cross product, in (coordinate unit)^2.
CROSS_EPS = 1.0e-6
# Tolerance for the point-in-triangle containment test.
CONTAIN_EPS = 1.0e-9
# Tolerance used when deciding whether two vertices are the same point.
SAME_POINT_EPS = 1.0e-9
BIG = 1.0e30


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def as_points(vertices) -> np.ndarray:
    """Accept vec2s, tuples, lists or an (N, 2) array -> (N, 2) float64 array."""
    if isinstance(vertices, np.ndarray):
        return np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
    return np.asarray([(float(v[0]), float(v[1])) for v in vertices], dtype=np.float64)


def orient(a, b, c) -> float:
    """
    2D cross product of (b - a) and (c - a).

        > 0 : c lies to the left of a->b   (a, b, c is a positive turn)
        = 0 : collinear
        < 0 : c lies to the right of a->b

    This is exactly the determinant the original `verify_counter_clockwise`
    computed, isolated so it can be reused and tested.
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])


def signed_area(points: np.ndarray) -> float:
    """
    Shoelace signed area. Positive => the vertex order is a positive
    (counter-clockwise in a y-up frame) winding.
    """
    """
    My Notes.

    The `np.roll` is an optimized way of iterating and operating through an ndarray.
    `np.roll(a, -1)`, where a is an ndarray, shifts the whole array to the left
    such that the first value originally at index 0, is wrapped around and now appears
    at index N-1, where N is the length of the ndarray.

    1) `np.dot(x, np.roll(y, -1))`, performs a dot operation between each value of x and each value in the
    shifted array.
    2) This is similar with `np.dot(np.roll(x, -1), y)`.

    The reason for the shift of the y values in 1) is to ensure that the x values do not dot-multiply their original y-values
    but rather dot-multiply the y values ahead of them.
    Similar for shift of the x values in 2).
    So, for 1) x<0> dot-multiplies y<N-1> and for 2) y<0> dot-multiplies x<N-1>
    This ensures the loop closes back to the starting point.
    
    It does this to perform the shoelace formula to calculate the total area enclosed by the coordinates/points.

    `np.dot(x, np.roll(y, -1))`, dot-multiplies the elements of the two arrays pairwise and sums them up, so `∑(𝑥𝑖⋅𝑦𝑖+1)`
    `np.dot(np.roll(x, -1), y)`, hence computes `∑(𝑥𝑖+1⋅𝑦𝑖)`

    `0.5 * float(...)`: subtracting the two sums and multiplying by 0.5 yields the final area according to the mathematical
    formula:
    ```
    1/2 | <n>∑<i=1>[(𝑥𝑖⋅𝑦𝑖+1 - 𝑥𝑖+1⋅𝑦𝑖)] |
    ```

    Note on Orientation.
    If the points are ordered anticlockwise, the area will be positive.
    If they are ordered clockwise, the area will be negatice.
    But this code should always return a positive area because the points are validated
    to have an anti-clockwise winding.
    Still, this function is used to indeed check if the winding is clockwise or anticlockwise.
    """
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def point_in_triangle(p, a, b, c, eps: float = CONTAIN_EPS) -> bool:
    """
    Inclusive containment test for a *positively oriented* triangle (a, b, c).
    Points lying on an edge count as contained: that is deliberate, because a
    vertex sitting on a candidate diagonal makes the ear unsafe to clip.
    """
    """
    My Notes.
    why `>=-eps` is done is to return true both if the winding is anticlockwise (the point, p,
    falls on the left side of all the edges of the triangle (a, b, c)) and if
    if the point, p, lies on or is collinear to any of the segments, a->b, b->c, or c->a.

    For a triangle defined by counter-clockwise (CWW) order (a -> b -> c -> a), a point is
    inside the triangle if it lies to the left of all three directed edges:

    orient(a, b, p): checks if p is to the left of segment/edge a->b
    orient(b, c, p): checks if p is to the left of segment/edge b->c
    orient(c, a, p): checks if p is to the left of segment/edge c->a

    If all three are true, the point is trapped inside the the triangle OR lies
    on one of its edges.

    Now, if eps is a negative number, the, -eps => +eps. Then the check will be stricter,
    only returning true if the point is inside the triangle and False if it lies on any edge
    """
    return (orient(a, b, p) >= -eps
            and orient(b, c, p) >= -eps
            and orient(c, a, p) >= -eps)


def same_point(p, q, eps: float = SAME_POINT_EPS) -> bool:
    return abs(p[0] - q[0]) <= eps and abs(p[1] - q[1]) <= eps


def point_in_polygon(p, points: np.ndarray) -> bool:
    """Even-odd ray cast. Used by the verification tests, not by the clipper."""
    x, y = float(p[0]), float(p[1])
    n = len(points)
    inside = False
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xint:
                inside = not inside
    return inside


# ---------------------------------------------------------------------------
# Validity checks
# ---------------------------------------------------------------------------

def _on_segment(p, q, r, eps: float = CROSS_EPS) -> bool:
    """True when q lies on segment p-r, assuming the three are collinear."""
    """
    My Notes:
    Assuming the three are collinear, the below check verfies that point q
    is within the bounds of points p and r, essentially within their bounding
    rect. If that is the case and they ARE collinear (assuming), then q indeed
    lies on the segment/edge, p-r
    """
    return (min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps
            and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps)


def segments_cross(a, b, c, d, eps: float = CROSS_EPS) -> bool:
    """
    Proper + improper segment intersection test.
    Unlike the `np.sign(...) != np.sign(...)` version, this handles the
    collinear cases explicitly, so overlapping edges are reported instead of
    being silently missed (sign(0) != sign(0) is False).
    """
    """
    My Notes.
    orient(p, q, r) checks if point r is on the left (+ve), right (-ve),
    or collinear (0 or within eps) of the edge formed by p -> q. 
    If on the left, the winding of p->q->r is anticlockwise and correct.
    If on the right, that winding is clockwise and unnacceptable.
    If 0, or within EPS, there is no winding. They are collinear.
    """
    d1 = orient(a, b, c)
    d2 = orient(a, b, d)
    d3 = orient(c, d, a)
    d4 = orient(c, d, b)

    """
    My Notes.
    The below works to check if the segments formed by the appropriate points cross each other.
    If d1 > eps (+ve), it means c is left/at the left side of edge a->b
    If d2 < -eps (-ve), it means d is right/at the right side of edge a->b
    This means edge/segment c->d indeed crosses a->b.
    Similarly,
    if d1 < -eps (-ve), it means c is right of edge a->b
    if d2 > eps (+v2), it means d is left of edge a->b
    This also means that semgent c-?d indeed crosses a->b

    The checks for d3 and d4 are the same logic, but checking for the adjecency (whether to left or right)
    of a anb b relative to segment cd.

    This is essential for robustness against of collinearity and precision
    """
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
        ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True

    # It reaches here either if the segments do not cross or they are collinear
    """
    My Notes

    if they are collinear AND a point is found to lie on a segment,
    that is interpreted as a crossing or intersection and hence
    one of these conditions will be true.
    """
    if abs(d1) <= eps and _on_segment(a, c, b):
        return True
    if abs(d2) <= eps and _on_segment(a, d, b):
        return True
    if abs(d3) <= eps and _on_segment(c, a, d):
        return True
    if abs(d4) <= eps and _on_segment(c, b, d):
        return True

    # If there is neither an explicit crossing nor a collinear but sits on same
    # segment intersection, then there is indeed no edge/segment crossing or touching another
    # edge/segment
    return False


def is_self_intersecting(vertices) -> bool:
    """True when any two non-adjacent edges of the closed ring touch or cross."""
    pts = as_points(vertices)
    n = len(pts)
    if n < 4:
        return False

    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 1, n):
            # Adjacent edges legitimately share a vertex; skip them,
            # including the wrap-around pair (0, n-1).
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            c, d = pts[j], pts[(j + 1) % n]
            if segments_cross(a, b, c, d):
                return True
    return False


def has_duplicate_points(vertices, eps: float = 1.0e-6) -> bool:
    pts = as_points(vertices)
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if same_point(pts[i], pts[j], eps):
                return True
    return False


# ---------------------------------------------------------------------------
# Ear clipping
# ---------------------------------------------------------------------------

@dataclass
class Triangulation:
    """
    triangles : list of (a, b, c) index triples into the *original* vertex list.
    diagonals : list of (h, j) index pairs, one per clipped ear, in clip order.
    ok        : True when a full fan of n - 2 triangles was produced.
    message   : human-readable status, safe to render in a HUD.
    """
    triangles: list = field(default_factory=list)
    diagonals: list = field(default_factory=list)
    ok: bool = False
    message: str = ""

    @property
    def count(self) -> int:
        return len(self.triangles)


def _is_ear(pts, ring, k, strict_containment=True, cross_eps=CROSS_EPS):
    """
    Test the vertex at ring position `k` (neighbours k-1 and k+1).

    Returns (is_ear, diagonal_length_squared).

    Two conditions, and the second is the one the original code was missing:

      1. Convexity   -- orient(h, i, j) > eps in the canonical winding.
      2. Emptiness   -- no other polygon vertex lies inside triangle (h, i, j).

    For a convex polygon condition 2 is automatically satisfied, which is
    why the original algorithm looked correct on convex input and produced
    overlapping garbage on non-convex input.
    """
    m = len(ring)
    h = ring[(k - 1) % m]
    i = ring[k]
    j = ring[(k + 1) % m]

    a, b, c = pts[h], pts[i], pts[j]

    # If the point ordering is collinear or clockwise, it is invalid
    if orient(a, b, c) <= cross_eps:
        return False, BIG          # reflex or collinear -> not an ear

    """
    My Notes.

    This checks if a point in the ring of indexes of the polygon left to be processed,
    excluding indexes h, i, or j, fall inside the triang formed by h, i, j.
    If they do, return False and the diagonal distance as BIG. It is invalid
    because `strict_containment` is true

    If `strict_containment = True`, it flags if any point is exactly contained
    in the triangle or if it touches the edge.
    If `strict_containment = False`, it only flags if any point is exactly contained
    in the triangle.
    This is controlled by eps. 
    """

    eps = CONTAIN_EPS if strict_containment else -CONTAIN_EPS
    for idx in ring:
        if idx in (h, i, j):
            continue
        p = pts[idx]
        # A duplicated coordinate at a corner is not a genuine blocker.
        if same_point(p, a) or same_point(p, b) or same_point(p, c):
            continue
        if point_in_triangle(p, a, b, c, eps):
            return False, BIG


    """
    If there is no point inside the triangle a, b, c from indices h, i, j
    then return True (ok), and the length of the diagonal of the triangle,
    which is magnitude |c-a|, as the best distance
    """
    dx = c[0] - a[0]
    dy = c[1] - a[1]
    return True, dx * dx + dy * dy


def _best_ear(pts, ring, strict_containment=True, require_empty=True):
    """Pick the valid ear whose diagonal is shortest. Returns k or -1."""
    best_k, best_d = -1, BIG
    m = len(ring)
    for k in range(m):
        if require_empty:
            ok, d = _is_ear(pts, ring, k, strict_containment)
        else:
            h, i, j = ring[(k - 1) % m], ring[k], ring[(k + 1) % m]
            ok = orient(pts[h], pts[i], pts[j]) > CROSS_EPS
            dx = pts[j][0] - pts[h][0]
            dy = pts[j][1] - pts[h][1]
            d = dx * dx + dy * dy
        if ok and d < best_d:
            best_k, best_d = k, d
    return best_k


def triangulate(vertices, validate: bool = True) -> Triangulation:
    """
    Ear-clipping triangulation of a simple polygon, shortest-diagonal-first.

    Returns index data only; the caller owns all drawing.
    """
    pts = as_points(vertices)
    n = len(pts)
    result = Triangulation()

    if n < 3:
        result.message = f"Need at least 3 vertices (have {n})."
        return result

    if validate:
        if has_duplicate_points(pts):
            result.message = "Polygon has duplicate vertices."
            return result
        if is_self_intersecting(pts):
            result.message = "Polygon is self-intersecting."
            return result

    area = signed_area(pts)
    if abs(area) <= CROSS_EPS:
        result.message = "Polygon is degenerate (zero area)."
        return result

    # Canonical winding: work on a ring whose signed area is positive so that
    # "convex" always means orient(h, i, j) > 0. The caller's own ordering is
    # left untouched; only this local index ring is reversed.
    ring = list(range(n))
    if area < 0.0:
        ring.reverse()

    guard = 0
    max_iterations = n + 8

    while len(ring) > 3:
        guard += 1
        if guard > max_iterations:
            result.message = "Clipping stalled; aborted by iteration guard."
            return result

        k = _best_ear(pts, ring, strict_containment=True)
        if k < 0:
            # Tier 2: allow vertices that merely touch the boundary of the ear.
            k = _best_ear(pts, ring, strict_containment=False)
        if k < 0:
            # Tier 3: convexity only. Reachable only on badly degenerate input;
            # recorded in the message so the result is never silently wrong.
            k = _best_ear(pts, ring, require_empty=False)
            if k >= 0:
                result.message = ("Fell back to convexity-only clipping; "
                                  "input is numerically degenerate.")
        if k < 0:
            result.message = "No clippable ear found (polygon is not simple)."
            return result

        m = len(ring)
        h = ring[(k - 1) % m]
        i = ring[k]
        j = ring[(k + 1) % m]

        result.triangles.append((h, i, j))
        result.diagonals.append((h, j))
        ring.pop(k)

    # The final three survivors form the last triangle. The original loop
    # exited here and dropped it.
    result.triangles.append(tuple(ring))
    result.ok = (len(result.triangles) == n - 2)
    if result.ok and not result.message:
        result.message = f"OK - {n - 2} triangles, {n - 3} diagonals."
    return result