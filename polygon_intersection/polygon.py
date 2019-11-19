"""
    Defined polygon structure of points.

    Using Point objects, rather than tuples or arrays, makes it 
    easier to manipulate.

    Can choose to store edges in addition to speed up this code. The 
    Polygon is always assumed to be "closed", that is, once three or
    more points exist, then there is a final closing edge from the 
    final point, back to the first point.
"""

from polygon_intersection.point import Point
from polygon_intersection.edge import Edge
from polygon_intersection.util import computeAngleSign

class Polygon:
    """Represents polygon of points in Cartesian space."""

    #   pts = [[57.0, 134.0], up left
    #          [1910.0, 144.0], up right
    #          [1915.0, 998.0], down right
    #          [57.0, 969.0]] down left
    def __init__(self, pts=[]):
        """
        Creates polygon from list of points. If omitted, polygon is empty.
        """
        if len(pts) == 0:
            self.points = []
        else:
            self.points = [None, None, None, None]
            center = [0, 0]
            for pt in pts:
                center[0] += pt[0]
                center[1] += pt[1]
            center[0]= center[0] / 4;
            center[1]= center[1] / 4;
            for pt in pts:
                left = pt[0] < center[0]
                up = pt[1] < center[1]
                if up and left:
                    idx = 0
                elif up and not left:
                    idx = 1
                elif not up and not left:
                    idx = 2
                else:
                    idx = 3
                self.points[idx] = Point(pt[0], pt[1])


    def copy(self):
        """Return copy of polygon."""
        return Polygon(self.points)

    def add(self, x, y):
        """Extend polygon with additional (x,y) point."""
        self.points.append(Point(x,y))
        n = len(self.points)

    def get(self, n):
        """Returns the nth point from polygon (based on zero)."""
        return self.points[n]

    def remove(self, n):
        """Delete the nth point from polygon (based on zero)."""
        del self.points[n]

    def numPoints(self):
        """Return the number of points in polygon."""
        return len(self.points)

    def numEdges(self):
        """Return the number of edges in polygon."""
        if len(self.points) < 1:
            return 0
        elif len(self.points) == 2:
            return 1
        else:
            return len(self.points)

    def getPoints(self):
        res = []
        for i, p in enumerate(self.points):
            res.append([p.x(), p.y()])
        return res

    def getContourPoints(self):
        res = self.getPoints()
        res.append([self.points[0].x(), self.points[0].y()])
        return res

    def getArea(self):
        area = 0
        N = len(self.points)
        for i in range(N-1):
            area += self.points[i].x() * self.points[i+1].y() - self.points[i].y() * self.points[i+1].x()
        area += self.points[N-1].x() * self.points[0].y() - self.points[N-1].y() * self.points[0].x()
        area = abs(area)
        return area

    def valid(self):
        """A polygon becomes valid with three or more points."""
        return len(self.points) >= 3

    def convex(self):
        """
        Determine if polygon is convex and in standard-form,
        which means, points and edges are in counter-clockwise
        ordering, with polygon interior on the left of the edges.
        """
        if not self.valid() or not self.simple():
            return False

        for i in range(len(self.points)-2):
            sign = computeAngleSign(self.points[i].x(),
                                    self.points[i].y(),
                                    self.points[i+1].x(),
                                    self.points[i+1].y(),
                                    self.points[i+2].x(),
                                    self.points[i+2].y())
            if sign < 0:
                return False
        
        # check final one
        sign = computeAngleSign(self.points[-2].x(),
                                self.points[-2].y(),
                                self.points[-1].x(),
                                self.points[-1].y(),
                                self.points[0].x(),
                                self.points[0].y())
        return sign >= 0
                                
    def simple(self):
        """
        Determine if a polygon is simple, that is, doesn't have 
        two different edges that intersect each other.
        """
        all = list(self.edges())
        for i in range(0, len(all)-1):
            e = all[i]
            for j in range(i+1, len(all)):
                if e.intersect(all[j]):
                    return False
        
        return True

    def intersect(self, p):
        """Return true if two polygons intersect. Checks edges."""
        for e in self.edges():
            for o in p.edges():
                if e.intersect(o) is not None:
                    return True
        return False

    def __iter__(self):
        """Return points in the polygon in order."""
        for pt in self.points:
            yield pt

    def edges(self):
        """Return edges in the polygon, in order."""
        order = []
        for i in range(0, len(self.points)-1):
            order.append(Edge(self.points[i], self.points[i+1]))

        if self.valid():
            n = len(self.points)
            order.append(Edge(self.points[n-1], self.points[0]))

        # Now link edges to next one in the chain. Make sure to
        # link back to start
        for i in range(len(order)-1):
            order[i].setNext(order[i+1])
        order[-1].setNext(order[0])
        return order
                             
    def __str__(self):
        """Return string representation."""
        s = '{'
        for pt in self.points:
            s += str(pt)
        return s + '}'

    def __eq__(self, other):
        """Standard equality check."""
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        """Standard not-equality check."""
        return not self.__eq__(other)

