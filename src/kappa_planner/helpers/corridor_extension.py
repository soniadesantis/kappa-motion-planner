"""Extend covered corridor edges without changing the free-space union."""

from collections import deque


def extend_perpendicular_corridors(bounds, headings):
    """Push fully covered edges outward to the perpendicular neighbor's wall.

    Only consecutive pairs with perpendicular intended axes are considered.
    Headings are integer quarter turns (0: right, 1: up, 2: left, 3: down).
    A vertical edge can move in x only if its entire y span is inside the
    neighbor; a horizontal edge similarly requires its entire x span inside.
    The added strip is contained in that neighbor, so the union of free space
    is preserved. Partially covered edges and aligned pairs are left alone.

    Revisit only pairs adjacent to a changed rectangle until no eligible edge
    remains. Coordinates move monotonically and are copied from existing walls,
    giving finite termination and an idempotent result. These are closed-set
    comparisons without a tolerance that could enlarge the original union.
    """
    rectangles = [list(b) for b in bounds]
    if len(rectangles) != len(headings):
        raise ValueError('Expected one heading per corridor.')
    eligible = {j for j in range(len(rectangles)-1) if (headings[j]-headings[j+1]) % 2}
    pending = deque(sorted(eligible))
    queued = set(pending)
    changes = []
    while pending:
        pair = pending.popleft()
        queued.remove(pair)
        for index, neighbor in ((pair, pair+1), (pair+1, pair)):
            a, b = rectangles[index], rectangles[neighbor]
            changed = False
            for axis in (0, 1):
                along, transverse = 2*axis, 2*(1-axis)
                if not (b[transverse] <= a[transverse]
                        and a[transverse+1] <= b[transverse+1]):
                    continue
                for side in (0, 1):
                    edge = along+side
                    old, target = a[edge], b[edge]
                    inside = b[along] <= old <= b[along+1]
                    outward = target < old if side == 0 else target > old
                    if inside and outward:
                        a[edge] = target
                        changed = True
                        changes.append(dict(corridor=index, neighbor=neighbor,
                                            edge=('xmin', 'xmax', 'ymin', 'ymax')[edge],
                                            before=old, after=target))
            if changed:
                for affected in (index-1, index):
                    if affected in eligible and affected not in queued:
                        queued.add(affected)
                        pending.append(affected)
    return [tuple(b) for b in rectangles], changes
