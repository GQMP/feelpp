# PolygonList :
#   List of 2d parametric points
#   List of polygons (list of points counter clockwise), with holes (list of points clockwise)
#       - [[0,1,2,3],[4,5,6],[7,8,9]]
#       - [0,1,2,3]
#   Pts list in the same polygon cannot have a point in common

# Valid angles polygon, [o,i] :
#   -1 if no angles valid in any polygon between o&i
#   id_poly if i lies in an angle o of the polygon id_poly
#       (the set is closed then open, counter-clockwise
#   valid_angle store the index of the correct angle
#       (of the right list, and no problems because lists never have point in common

# from collections import defaultdict
from itertools import chain

import numpy as np
import sympy as sp

# todo : check if I can replace by checking closed segments...
def is_intersecting_segments_equation(a,b,c,d) :
    x0,y0 = a
    x1,y1 = b
    x2,y2 = c
    x3,y3 = d

    determinant = sp.factor(-(x0-x1)*(y2-y3) + (y0-y1)*(x2-x3))
    eq_is_collinear = sp.factor(sp.Eq((x2-x0)*(y1-y3) - (x1-x3)*(y2-y0),0) & sp.Eq(determinant,0))

    eq_collinear_intersect = sp.factor(
        ((x0 - x2 < 0) & (x2 - x1 < 0)) |
        ((x0 - x3 < 0) & (x3 - x1 < 0))
    )

    eq_intersect = (
        (-(y2-y3)*(x3-x1) + (x2-x3)*(y3-y1)),
        (-(y0-y1)*(x3-x1) + (x0-x1)*(y3-y1))
    )

    eq_non_collinear_intersect = sp.factor(
        (
            ((determinant >= 0) & (eq_intersect[0] > 0) & (eq_intersect[0] - determinant < 0)) |
            ((determinant <= 0) & (eq_intersect[0] < 0) & (eq_intersect[0] - determinant > 0))
        ) & (
            ((determinant >= 0) & (eq_intersect[1] > 0) & (eq_intersect[1] - determinant < 0)) |
            ((determinant <= 0) & (eq_intersect[1] < 0) & (eq_intersect[1] - determinant > 0))
        )
    )

    return (eq_is_collinear & eq_collinear_intersect) | (~eq_is_collinear & eq_non_collinear_intersect)


def is_point_in_angle_equation(o,a,b,c) :
    (ax,ay),(bx,by),(cx,cy),(ox,oy) = a,b,c,o

    # Project all points in a circle center by O then verification is A'B' x A'C' > 0
    # with A' = (A - O) / ||OA|| + O & so on, then expand ->
    # === >
    oax,obx,ocx = ax - ox,bx - ox,cx - ox
    oay,oby,ocy = ay - oy,by - oy,cy - oy
    ra,rb,rc = sp.sqrt(oax*oax + oay*oay),sp.sqrt(obx*obx + oby*oby),sp.sqrt(ocx*ocx + ocy*ocy)

    return sp.factor(ra*(oby*ocx - obx*ocy) + rb*(ocy*oax - ocx*oay) + rc*(oay*obx - oax*oby) < 0)


def is_point_in_triangle_equation(a,b,c,p) :
    px,py = p

    eq = True
    for i,j in [(a,b),(b,c),(c,a)] :
        (ix,iy),(jx,jy) = i,j
        eq &= (jx - ix) * (py - iy) - (jy - iy) * (px - ix) > 0

    return eq


def is_point_in_simple_polygon_equation(pts_arr,id_p,polygon,gamma) :
    xp,yp = pts_arr[id_p]
    xg,yg = gamma

    eq_halfline = False
    for i,j in chain(zip(polygon,polygon[1:]),[(polygon[-1],polygon[0])]) :
        (xi,yi),(xj,yj) = pts_arr[i],pts_arr[j]
        determinant = yg * (xj - xi) + xg * (yi - yj)
        t = yg * (xj - xp) + xg * (yp - yj)
        w = (yj - yi) * (xp - xj) + (xi - xj) * (yp - yj)
        eq_halfline ^= sp.factor(
            ((determinant >= 0) & (t > 0) & (t < determinant) & (w > 0)) |
            ((determinant <= 0) & (t < 0) & (t > determinant) & (w < 0))
        )

    eq_not_point_in_polygon = True
    for i in polygon :
        xi,yi = pts_arr[i]
        eq_not_point_in_polygon &= ~sp.Eq(xi - xp,0) & ~sp.Eq(yi - yp,0)

    return eq_not_point_in_polygon & eq_halfline
    # return False


def is_point_on_segment_equation(a,b,p) :
    (ax,ay),(bx,by),(px,py) = a,b,p

    always_checked = sp.Eq((bx - ax) * (py - ay) - (by - ay) * (px - ax),0)
    cx_not_zero = ((px - ax > 0) & (px - bx < 0) & (bx - ax > 0)) | ((px - ax < 0) & (px - bx > 0) & (bx - ax < 0))
    cy_not_zero = ((py - ay > 0) & (py - by < 0) & (by - ay > 0)) | ((py - ay < 0) & (py - by > 0) & (by - ay < 0))
    cx_cy_zero = ~sp.Eq(ax,bx) & ~sp.Eq(ay,by) & sp.Eq(ax,px) & sp.Eq(ay,py)
    return always_checked & (cx_not_zero | cy_not_zero | cx_cy_zero)


class PolygonList :
    def __init__(self,pts_arr : list,polygons : list,symbols : list) :
        self.pts_arr  = pts_arr.copy()
        self.polygons = []
        for p in polygons :
            if isinstance(p,list) and isinstance(p[0],list) :
                self.polygons.append(p)
            else :
                assert(isinstance(p,list))
                self.polygons.append([p])
        self.intersections = None
        self.valid_angles = None
        self.valid_angles_id_poly = None
        self.symbols = symbols.copy()
        self.gamma_symbols = None
        self.gamma = None


    def compute_gamma(self,mus) :
        # List all directions
        dirs_x = []
        dirs_y = []
        for i in range(len(self.pts_arr)) :
            for j in range(i+1,len(self.pts_arr)) :
                ax,ay = self.pts_arr[i]
                bx,by = self.pts_arr[j]
                dirs_x.append(sp.Abs(bx - ax))
                dirs_y.append(sp.Abs(by - ay))

        # Then choose one not in the list ............
        dx,dy = sp.simplify(2*sp.Max(*dirs_x)),sp.simplify(sp.Min(1,*dirs_y))

        # And as it is complicated to manipulate, compute in advance
        self.gamma_symbols = sp.symbols("gamma_x gamma_y",real = True)
        self.gamma = mus.numerical_results(dx),mus.numerical_results(dy)


    def compute_valid_angles(self,mus) :
        num_pts = len(self.pts_arr)
        self.valid_angles = np.ones((num_pts,num_pts),dtype = np.int) * -1
        self.valid_angles_id_poly = np.ones((num_pts,num_pts),dtype = np.int) * -1

        for id_poly in range(len(self.polygons)) :
            self.compute_valid_angles_polygon(mus,id_poly)


    def compute_valid_angles_polygon(self,mus,id_poly) :
        # valid_pts = list(chain(*[line for line in self.polygons[id_poly]]))
        # non_valid_pts = set(range(len(self.pts_arr))) - valid_pts

        for i,(a,o,b) in [
            (i,(line[(i - 1 + len(line)) % len(line)],line[i],line[(i + 1) % len(line)]))
                    for line in self.polygons[id_poly] for i in range(len(line))
            ] :
            for c in chain(*[line for line in self.polygons[id_poly]]) :
                if c == a or c == o :
                    pass
                    # self.valid_angles[o,c] = -1
                    # self.valid_angles_id_poly[o,c] = -1
                elif c == b :
                    self.valid_angles[o,c] = i
                    self.valid_angles_id_poly[o,c] = id_poly
                else :
                    # self.valid_angles[o,c] = -1
                    # self.valid_angles_id_poly[o,c] = -1
                    eq = is_point_in_angle_equation(self.pts_arr[o],self.pts_arr[a],self.pts_arr[b],self.pts_arr[c])
                    if mus.numerical_results(eq).all() :
                        self.valid_angles[o,c] = i
                        self.valid_angles_id_poly[o,c] = id_poly


    def compute_intersections(self,mus) :
        def S(a,b) :
            return (a,b) if a < b else (b,a)

        num_pts = len(self.pts_arr)
        self.intersections = np.ones((num_pts,num_pts),dtype = np.int) * -1  # ,[0 for _ in range(num_pts * num_pts)])
        self.valid_angles = np.ones((num_pts,num_pts),dtype = np.int) * -1
        self.valid_angles_id_poly = np.ones((num_pts,num_pts),dtype = np.int) * -1

        for id_poly in range(len(self.polygons)) :
            self.compute_intersections_polygon(mus,id_poly)


    def compute_intersections_polygon(self,mus,id_poly) :
        self.compute_valid_angles_polygon(mus,id_poly)

        def S(a,b) :
            return (a,b) if a < b else (b,a)

        constraint_segments = set(S(i,j)
            for i,j in chain(
                *[zip(line,line[1:]) for line in self.polygons[id_poly]],
                ((line[-1],line[0]) for line in self.polygons[id_poly])
            )
        )

        polygon_pts = sorted(set(pt for pt in chain(*[line for line in self.polygons[id_poly]])))
        for i in range(len(polygon_pts)) :
            for j in range(i,len(polygon_pts)) :
                segment = polygon_pts[i],polygon_pts[j]
                if segment in constraint_segments :
                    self.intersections[segment] = self.intersections[segment[::-1]] = np.iinfo(np.int).max
                elif self.check_polygon_segment_in_polygon(mus,segment,id_poly) :
                    self.intersections[segment] = self.intersections[segment[::-1]] = id_poly


    def check_polygon_segment_in_polygon(self,mus,segment,id_poly) :
        lines = self.polygons[id_poly]
        A,B = segment
        if A == B : return False
        def S(x0,x1) : return (x0,x1) if x0 < x1 else (x1,x0)

        # Check if segment lies in the polygon at its starting points
        exclude_segments = []
        for X,Y in [(A,B),(B,A)] :
            if self.valid_angles_id_poly[X,Y] != id_poly :
                return False
            for line in lines :
                if self.valid_angles[X,Y] < len(line) and line[self.valid_angles[X,Y]] == X :
                    exclude_segments.extend([
                        S((self.valid_angles[X,Y] - 1 + len(line)) % len(line),X),
                        S((self.valid_angles[X,Y] + 1            ) % len(line),X)])
                    break

        # Then check if segment intersect other segments of the polygon
        all_pts = set(chain(*[line for line in lines]))
        for (i,j) in chain(*[zip(line,line[1:]) for line in lines],((line[0],line[-1]) for line in lines)) :
            i,j = S(i,j)
            if (i,j) in exclude_segments : continue

            eq_intersecting = is_intersecting_segments_equation(
                self.pts_arr[A],self.pts_arr[B],self.pts_arr[i],self.pts_arr[j]
            )

            checks = eq_intersecting
            if mus.numerical_results(checks).any() :
                return False  # Intersections exists with the mus tested...

        # Finally, the stupids tests needed because all segments are tested open...
        eq_checks = False
        for p in all_pts :
            if p == A or p == B : continue

            eq_on_diagonal = is_point_on_segment_equation(self.pts_arr[A],self.pts_arr[B],self.pts_arr[p])
            eq_checks |= eq_on_diagonal
                    # | ((self.valid_angles[p,i] >= 0) & (self.valid_angles[p,j] >= 0)) # Removed...
        if mus.numerical_results(eq_checks).any() :
            return False

        return True


    def list_triangles_in_polygon(self,mus,id_polygon) :
        def is_cw(x0,x1,x2) :
            (ax,ay),(bx,by),(cx,cy) = x0,x1,x2
            return (bx - ax) * (cy - ay) - (cx - ax) * (by - ay) > 0

        lines = self.polygons[id_polygon]
        list_pts = sorted(set(pt for pt in chain(*(line for line in lines))))
        num_pts = len(list_pts)
        for i in range(num_pts) :
            for j in range(i + 1,num_pts) :
                for k in range(j + 1,num_pts) :
                    a,b,c = list_pts[i],list_pts[j],list_pts[k]
                    if (self.intersections[a,b] == id_polygon or self.intersections[a,b] == np.iinfo(np.int).max) \
                   and (self.intersections[b,c] == id_polygon or self.intersections[b,c] == np.iinfo(np.int).max) \
                   and (self.intersections[a,c] == id_polygon or self.intersections[a,c] == np.iinfo(np.int).max) :
                        # Check the order of points
                        # Todo : Single mu must be ok
                        if mus.numerical_results(is_cw(self.pts_arr[a],self.pts_arr[b],self.pts_arr[c])).any() :
                            yield (a,b,c)
                        else :
                            yield (a,c,b)


    def list_triangles(self,mus) :
        for i in range(len(self.polygons)) :
            for t in self.list_triangles_in_polygon(mus,i) :
                yield (i,t)


    def is_triangle_in_polygon(self,mus,id_poly : int,triangle : tuple([int,int,int])) -> bool :
        a,b,c = triangle

        lines = self.polygons[id_poly]
        # Checks if its points are points of the polygon (restrictive condition)
        for i in [a,b,c] :
            if all(i not in line for line in lines) :
                return False

        # Checks if the 3 points of the triangle sends their segments with the same angle of the polygon
        #   (In case of hole produced with the same line)
        for (i,j,k) in [(a,b,c),(b,c,a),(c,a,b)] :
            if self.valid_angles[i,j] != self.valid_angles[i,k] and \
                    self.valid_angles_id_poly[i,j] == id_poly and self.valid_angles_id_poly[i,k] == id_poly :
                return False

        # Checks if its segments lies in the polygon
        borders = set(chain(*[zip(line,line[1:]) for line in lines],((line[-1],line[0]) for line in lines)))
        for i,j in [(a,b),(b,c),(c,a)] :
            if self.intersections[i,j] < 0 :
                return False
            if self.intersections[i,j] != np.iinfo(np.int).max and self.intersections[i,j] != id_poly :
                return False
            if self.intersections[i,j] == np.iinfo(np.int).max and (i,j) not in borders :
                return False

        # Checks if no holes lies in the triangle
        for i in range(len(lines)) :
            pt = self.pts_arr[lines[i][0]]
            eq = is_point_in_triangle_equation(self.pts_arr[a],self.pts_arr[b],self.pts_arr[c],pt)
            if mus.numerical_results(eq).any() : # Todo : Single mu must be ok
                return False

        return True


    def polygon_minus_triangle(self,mus,id_polygon,triangle) :
        lines = self.polygons[id_polygon]
        a,b,c = triangle

        id_broken_lines = set()
        broken_lines = {}

        # Get the lines that change
        for x in [a,b,c] :
            for i,line in enumerate(lines) :
                if x in line :
                    broken_lines[x] = line
                    id_broken_lines.add(i)
                    break

        # Break them correctly
        if broken_lines[a] is broken_lines[b] and broken_lines[a] is broken_lines[c] :
            broken_line = broken_lines[a] # broken_lines[a][break_a:] + broken_lines[a][:-break_a]

            break_a = self.valid_angles[a,b] ## if self.valid_angles_id_poly[a,b] == id_polygon \
                 ##else self.valid_angles[a,c] # if self.valid_angles[a,c] >= 0 \
                 # else [i for i in range(len(broken_line))
                 #         if broken_line[(i-1+len(broken_line))%len(broken_line)] == c
                 #         and broken_line[i] == a
                 #         and broken_line[(i+1)%len(broken_line)] == b][0]
            # break_b = ((self.valid_angles[b,c] if self.valid_angles_id_poly[b,c] == id_polygon
            #      else self.valid_angles[b,a]) - break_a + len(broken_line)) % len(broken_line) # if self.valid_angles[b,c] >= 0
            #      # else (break_a + 1)) - break_a + len(broken_line)) % len(broken_line)
            # break_c = ((self.valid_angles[c,a] if self.valid_angles_id_poly[c,a] == id_polygon
            #      else self.valid_angles[c,b]) - break_a + len(broken_line)) % len(broken_line) # if self.valid_angles[c,b] >= 0
            #      # else (break_a - 1)) - break_a + len(broken_line)) % len(broken_line)

            break_b = (self.valid_angles[b,c] - break_a + len(broken_line)) % len(broken_line) # if self.valid_angles[b,c] >= 0
                 # else (break_a + 1)) - break_a + len(broken_line)) % len(broken_line)
            break_c = (self.valid_angles[c,a] - break_a + len(broken_line)) % len(broken_line) # if self.valid_angles[c,b] >= 0
                 # else (break_a - 1)) - break_a + len(broken_line)) % len(broken_line)

            ordered_line = broken_line[break_a - len(broken_line):] + broken_line[:break_a]
            broken_lines[a] = (b,ordered_line[:break_b])
            broken_lines[b] = (c,ordered_line[break_b:break_c])
            broken_lines[c] = (a,ordered_line[break_c:])
        elif broken_lines[a] is not broken_lines[b] \
         and broken_lines[a] is not broken_lines[c] \
         and broken_lines[b] is not broken_lines[c] :
            for x,y in [(a,b),(b,c),(c,a)] :
                break_pos = self.valid_angles[x,y]
                ordered_line = broken_lines[x][break_pos - len(broken_lines[x]):] + broken_lines[x][:break_pos]
                broken_lines[x] = (x,ordered_line)
        else :
            for x,y,z in [(a,b,c),(b,c,a),(c,a,b)] :
                if broken_lines[x] is not broken_lines[y] : continue
                break_z = self.valid_angles[z,x]
                broken_lines[z] = (z,broken_lines[z][break_z - len(broken_lines[z]):] + broken_lines[z][:break_z])
                break_x = self.valid_angles[x,z]
                break_y = (self.valid_angles[y,z] - break_x + len(broken_lines[x])) % len(broken_lines[x])

                ordered_line = broken_lines[x][break_x - len(broken_lines[x]):] + broken_lines[x][:break_x]
                broken_lines[x] = (y,ordered_line[:break_y])
                broken_lines[y] = (x,ordered_line[break_y:])

                break

        # Links all remaining lines correctly
        new_polygons = []
        next_pt = { b : a,c : b,a : c }
        while len(broken_lines) > 0 :
            new_poly = []
            start = next(iter(broken_lines.keys()))
            follow = None
            while follow != start :
                if follow is None : follow = start
                tmp,line = broken_lines[follow]
                del broken_lines[follow]
                follow = tmp
                new_poly += line
                new_poly.append(follow)
                follow = next_pt[follow]
            if len(new_poly) >= 3 :
                new_polygons.append([new_poly])

        # Put holes generated just before into respective polygons
        id_to_delete = []
        for poly in new_polygons :
            for id_hole,hole in enumerate(new_polygons) :
                if poly is hole : continue

                eq = is_point_in_simple_polygon_equation(self.pts_arr,hole[0][0],poly[0],self.gamma_symbols)
                if mus.numerical_results(eq,[*self.gamma_symbols],[*self.gamma]).any() : # Todo : Single mu must be ok
                    poly.append(hole[0])
                    id_to_delete.append(id_hole)
        for e in sorted(id_to_delete,reverse = True) :
            del new_polygons[e]

        # Puts remaining holes into respective polygons
        for i,l in enumerate(lines) :
            if i in id_broken_lines : continue

            for poly in new_polygons :
                eq = is_point_in_simple_polygon_equation(self.pts_arr,l[0],poly[0],self.gamma_symbols)
                if mus.numerical_results(eq,[self.gamma_symbols],[*self.gamma]).any() : # Todo : Single mu must be ok
                    poly.append(l)

        # Delete the old polygon
        old_symbol = self.symbols[id_polygon]
        self.delete_polygon(id_polygon)

        # Then add the new ones
        for poly in new_polygons :
            self.polygons.append(poly)
            self.symbols.append(old_symbol)
        for k in range(len(new_polygons)) :
            id_poly = len(self.polygons)-k-1
            self.compute_intersections_polygon(mus,id_poly)


    def delete_polygon(self,id_poly) :
        del self.polygons[id_poly] ; del self.symbols[id_poly]
        self.valid_angles[self.valid_angles_id_poly == id_poly] = -1
        self.valid_angles_id_poly[self.valid_angles_id_poly == id_poly] = -1
        self.valid_angles_id_poly[self.valid_angles_id_poly > id_poly] -= 1
        self.intersections[self.intersections == id_poly] = -1
        self.intersections[(self.intersections > id_poly) & (self.intersections != np.iinfo(np.int).max)] -= 1


    def merge_polygons(self,mus,i,j) :
        first_line_id = -1
        second_line_id = -1
        for k in range(len(self.polygons[i])) :
            for l in range(len(self.polygons[j])) :
                common_pts = list(set(self.polygons[i][k]).intersection(set(self.polygons[j][l])))
                if len(common_pts) > 1 :
                    first_line_id = k
                    second_line_id = l
                    break
            if first_line_id >= 0 : break

        if first_line_id < 0 :
            return False

        first_line = self.polygons[i][first_line_id]
        second_line = self.polygons[j][second_line_id]
        edges_first = set(zip(first_line,first_line[1:] + [first_line[0]]))
        edges_second = set(zip(second_line,second_line[1:] + [second_line[0]]))
        del self.polygons[i][first_line_id]
        del self.polygons[j][second_line_id]

        all_edges = ((edges_first  - set([(l,k) for k,l in edges_second]))
                   | (edges_second - set([(l,k) for k,l in edges_first ])))
        permutation = {}
        for k,l in all_edges :
            permutation[k] = l

        new_lines = []
        while len(permutation) > 0 :
            s,n = permutation.popitem()
            current_line = [s]
            while n != s :
                current_line.append(n)
                n = permutation.pop(n)
            new_lines.append(current_line)


        found_main_line = False
        if len(self.polygons[i]) == 0 and len(self.polygons[j]) == 0 and len(new_lines) == 1 :
            found_main_line = True
        if not found_main_line and len(self.polygons[j]) > 0 :
            eq = is_point_in_simple_polygon_equation(self.pts_arr,new_lines[0][0],self.polygons[j][0],self.gamma_symbols)
            if mus.numerical_results(eq,[*self.gamma_symbols],
                                     [*self.gamma]).any() :  # Todo : Single mu must be ok
                self.polygons[i][0],self.polygons[j][0] = self.polygons[j][0],self.polygons[i][0]
                found_main_line = True
        if not found_main_line :
            for k in range(len(new_lines)) :
                test_pt = (new_lines[0][0] if k != 0 else
                      self.polygons[i][0] if len(self.polygons[i]) != 0 else
                      self.polygons[j][0] if len(self.polygons[j]) != 0 else
                         new_lines[1][0])

                eq = is_point_in_simple_polygon_equation(self.pts_arr,test_pt,new_lines[k],self.gamma_symbols)
                if mus.numerical_results(eq,[*self.gamma_symbols],
                                            [*self.gamma]).any() :  # Todo : Single mu must be ok
                    self.polygons[i][0],new_lines[k] = new_lines[k],self.polygons[i][0]
                    found_main_line = True
                    break

        self.polygons[i].extend(self.polygons[j])
        self.polygons[i].extend(new_lines)
        del self.polygons[j]
        del self.symbols[j]

        return True



# Some tests


def test_point_segment() :
    h,l = sp.symbols("h,l",real = True)
    eq = is_point_on_segment_equation((0,0),(l,h),(l/3,h/3))
    # print(eq)
    # print("Allo ?")
    # print(sp.simplify(eq))


def test_intersection_segments() :
    ax,ay,bx,by,cx,cy,dx,dy = sp.symbols("ax,ay,bx,by,cx,cy,dx,dy",real = True)
    eq = is_intersecting_segments_equation((ax,ay),(bx,by),(cx,cy),(dx,dy))

    eq1 = eq.subs([(ax,-1),(ay, 0),(bx, 1),(by, 0),
                   (cx, 0),(cy,-1),        (dy, 1)])
    assert(sp.simplify(eq1 | ~((dx > -2) & (dx < 2))))
    # assert(not sp.simplify(eq1 & ~((dx > -2) & (dx < 2)))) #)


def test_polygon_list() :
    np.set_printoptions(linewidth = 125)

    h,l = sp.symbols("h,l",real = True)
    pts_arr = [
        (0,0),(0,h),(l,0),(l,h),
        (l/3,h/3),(l/3,2*h/3),(2*l/3,h/3),(2*l/3,2*h/3),
    ]
    polygons = [
        [[0,2,3,1],[4,5,7,6]]
    ]
    from muslist import MusList
    mus = MusList([],[h,l],[0.,0.],[1.,1.],[0.5,0.5])
    mus.make_ranges(10)

    polylist = PolygonList(pts_arr,polygons,list(sp.symbols("O0:"+str(len(polygons)))))
    polylist.compute_intersections(mus)
    print("List : ",list(polylist.list_triangles_in_polygon(mus,0)))

    print(polylist.polygons)
    print(polylist.valid_angles)
    print(polylist.intersections)
