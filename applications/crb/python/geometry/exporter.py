from itertools import zip_longest

from muslist import MusList
from polygon import PolygonList

import sympy as sp

def export_mus(file,mus : MusList,polygons_list : PolygonList) :
    for mu,mu_min,mu_max in zip(mus.geom_ref,mus.geom_min,mus.geom_max) :
        # file.write("{} = {}; // ]{},{}[ \n".format(
        #     str(mu),str((sp.Rational((mu_min + mu_max) / 2)).evalf()),mu_min,mu_max
        # ))
        file.write("{0} = DefineNumber[{1},Name \"Parameters/{0}\"]; // ]{2},{3}[ \n".format(
            str(mu),str((sp.Rational((mu_min + mu_max) / 2)).evalf()),mu_min,mu_max
        ))


def export_pts(file,mus : MusList,polygons_list : PolygonList) :
    for i,(x,y) in enumerate(polygons_list.pts_arr) :
        file.write("Point({}) = {{ {},{},0 }};\n".format(
            i,str(x.subs(zip(mus.geom_mus,mus.geom_ref))),str(y.subs(zip(mus.geom_mus,mus.geom_ref)))
        ))


def export_edges(file,mus : MusList,polygons_list : PolygonList) :
    edges = {}

    edge_id = 1
    for poly in polygons_list.polygons :
        for line in poly :
            for i,j in zip_longest(line,line[1:],fillvalue = line[0]) :
                if edges.get((i,j)) is not None : continue

                file.write("Line({}) = {{ {},{} }};\n".format(edge_id,i,j))
                edges[i,j] = edge_id
                edges[j,i] = -edge_id
                edge_id += 1

    return edges


def export_polygons(file,mus : MusList,polygons_list : PolygonList) :
    export_mus(file,mus,polygons_list)
    export_pts(file,mus,polygons_list)
    edges = export_edges(file,mus,polygons_list)

    line_loop_id = 1
    for polygons_id,poly in enumerate(polygons_list.polygons) :
        list_loops = []
        for line in poly :
            file.write("Line Loop({}) = {{ {} }};\n".format(line_loop_id,
                ",".join(map(lambda edge_id : str(edges[edge_id]),zip_longest(line,line[1:],fillvalue = line[0])))
            ))

            list_loops.append(line_loop_id)
            line_loop_id += 1

        file.write("Plane Surface({}) = {{ {} }};\n".format(polygons_id+1,",".join(map(str,list_loops))))

    from itertools import compress
    for symbol in list(set(polygons_list.symbols)) :
        id_polys = compress(range(1,len(polygons_list.polygons)+1),[x == symbol for x in polygons_list.symbols])
        file.write("Physical Surface(\"{}\") = {{ {} }};\n".format(str(symbol),",".join(map(str,id_polys))))








