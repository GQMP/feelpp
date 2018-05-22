from itertools import chain

# import cProfile

import numpy as np
import sympy as sp

from form import separate_form,simplify_form,mul_ind_integral,apply_transform_opt
from muslist import MusList
from polygon import PolygonList


def construct_transformation(coords : list,mus_bar : list,mus : list) :
    """
    Express linear transformation of "parametric" coordinates to "reference" coordinates,
    using a parametrized simplex (of dimension len(coords)) as a barycentric basis.
    :param coords: List of "reference" coordinates.
    :param mus_bar: List of parameters to give to the functions of the points, for the reference simplex.
    :param mus: List of parameters to give to the functions of the points, for the parametrized simplex.
    :return: 2-uplet :
        - List of the "reference" coordinates expressed by the "parametric" coordinates.
        - Double list of vertex functions of the simplex,
            p[i][j], with i the point index and j the coordinate index.
    """
    # assert(len(mus) == len(mus_bar))

    # Dimension of the simplex
    D = len(coords)

    # Expression of points
    p = sp.symbols(['p%d_:%d' % (j,D) for j in range(D+1)],cls = sp.Function)

    # Matrix of "reference" points
    P = sp.Matrix()
    for i in range(D + 1) :
        P = P.col_insert(i,sp.Matrix([p[i][j](*mus_bar) for j in range(D)]))

    # Matrix of "parametric" points with a row of ones
    M = sp.Matrix()
    for i in range(D + 1) :
        M = M.col_insert(i,sp.Matrix([p[i][j](*mus) for j in range(D)]))
    M = M.row_insert(0,sp.ones(1,D+1))

    # 1 | "parametric" coordinates
    X = sp.Matrix(coords)
    X = X.row_insert(0,sp.ones(1,1))

    # Compute barycentric coordinates on the simplex at *mus of the "parametric" coordinates
    Y = M**-1 * X

    # Compute expression of "reference" coordinates on the simplex at *mus_bar using the barycentric coordinates
    T = sp.factor(P * Y) # sp.factor speedup the replacement time

    return tuple(T[i,0] for i in range(D)),p


def construct_triangles_form(mus : MusList,list_polygons : PolygonList,form,funcs,coords) :
    mus_symbols = mus.geom_mus
    mus_bar_symbols = mus.geom_ref
    mus_all = mus.geom_mus + mus.basic_mus

    funcs = funcs + list_polygons.symbols

    new_coords = sp.symbols("x0 x1")
    T,p = construct_transformation(coords,mus_bar_symbols,mus_symbols)
    Tx_base,Ty_base = T
    print(T)

    func_p = []
    for i,(px,py) in enumerate(list_polygons.pts_arr) :
        func_p.append((sp.Lambda(tuple(mus_symbols),px),sp.Lambda(tuple(mus_symbols),py)))
    ind_zero = sp.Lambda(tuple(coords),sp.Integer(0))
    ind_one = sp.Lambda(tuple(coords),sp.Integer(1))

    list_triangles_forms = {}
    for poly_id,(i,j,k) in list_polygons.list_triangles(mus) :
        print("Triangle :",(i,j,k))
        Tx,Ty = Tx_base,Ty_base

        for c,z in zip(range(len(p)),[i,j,k]) :
            for d in range(len(p[c])) :
                Tx = Tx.replace(p[c][d],func_p[z][d])
                Ty = Ty.replace(p[c][d],func_p[z][d])
        Tx = sp.simplify(Tx)
        Ty = sp.simplify(Ty)

        # Todo : Repair for fixed mus...
        to_subs = (
            list(zip(mus_symbols,mus_bar_symbols)) +
            list(zip(mus_bar_symbols,mus_symbols)) +
            list(zip(coords,new_coords))
        )
        Tx_inv = Tx.subs(to_subs,simultaneous = True)
        Ty_inv = Ty.subs(to_subs,simultaneous = True)

        local_form = form.copy()
        for id in range(len(list_polygons.polygons)) :
            local_form = local_form.replace(list_polygons.symbols[id],ind_zero if poly_id != id else ind_one)

        e = apply_transform_opt(local_form,funcs,list(zip(coords,[Tx_inv,Ty_inv])),list(zip(new_coords,[Tx,Ty])))
        sep_form = separate_form(e,mus_all,new_coords)

        list_triangles_forms[(i,j,k)] = sep_form

        for (c,m,x) in sep_form :
            print("  Cst :",c)
            print("  Mus :",m)
            print("  Xys :",x)
            print()

    return list_triangles_forms,new_coords


def construct_triangulation(mus : MusList,list_polygons : PolygonList,form,funcs,coords) :
    mus_all = mus.geom_mus + mus.basic_mus
    polygon_form = []
    list_triangles_forms,new_coords = construct_triangles_form(mus,list_polygons,form,funcs,coords)

    final_polygons = []

    def replace_triangle(active_polygon,polygon_form,list_triangles_forms,new_omega) :
        actual_len = len(polygon_form) + max(len(triangle_form) for triangle,triangle_form in list_triangles_forms.items()) + 1
        selected_triangle = None
        actual_new_form = None

        for triangle,triangle_form in list_triangles_forms.items() :
            if not list_polygons.is_triangle_in_polygon(mus,active_polygon,triangle) : continue

            new_form = simplify_form(
                polygon_form + [(c,m,mul_ind_integral(x,new_omega)) for (c,m,x) in triangle_form],
                mus_all,list(new_coords) + list(coords))
            if len(new_form) < actual_len :
                selected_triangle = triangle
                actual_new_form = new_form
                actual_len = len(new_form)

        if selected_triangle is None :
            return None

        list_polygons.polygon_minus_triangle(mus,active_polygon,selected_triangle)

        polygon_form.clear()
        polygon_form.extend(actual_new_form)

        # delete_intersecting_triangles(mus,pts_arr,list_triangles_forms,selected_triangle)
        return list(selected_triangle)

    identifier = 0
    while len(list_polygons.polygons) > 0 :
        print(list_polygons.polygons)
        active_polygon = len(list_polygons.polygons) - 1
        old_omega = list_polygons.symbols[active_polygon]
        new_omega = sp.Function(str(old_omega) + "_P" + str(identifier))
        new_triangle = replace_triangle(active_polygon,polygon_form,list_triangles_forms,new_omega)
        if new_triangle is not None :
            final_polygons.append((new_omega,new_triangle))
        else :
            new_omega = sp.Function(str(old_omega) + "_E" + str(identifier))
            final_polygons.append((new_omega,list_polygons.polygons[active_polygon]))
            polygon_form.extend([
                (c,m,mul_ind_integral(x,new_omega))
                    for (c,m,x) in separate_form(form,mus_all,new_coords)
            ])
            list_polygons.delete_polygon(active_polygon)
        identifier += 1

    polygon_form = simplify_form(polygon_form,mus_all,list(new_coords) + list(coords))
    return final_polygons,polygon_form,new_coords


def compute_linear_transforms(mus : MusList,list_polygons : PolygonList,coords) :
    """
    /!\ This function suppose that all polygons have a linear transformation to their reference
    """
    mus_symbols = mus.geom_mus
    mus_bar_symbols = mus.geom_ref

    T,p = construct_transformation(coords,mus_bar_symbols,mus_symbols)
    Tx_base,Ty_base = T
    # Tx_base = Tx_base.subs(chain(zip(mus_symbols,mus_bar_symbols),zip(mus_bar_symbols,mus_symbols)),simultaneous = True)
    # Ty_base = Ty_base.subs(chain(zip(mus_symbols,mus_bar_symbols),zip(mus_bar_symbols,mus_symbols)),simultaneous = True)

    func_p = []
    for i,(px,py) in enumerate(list_polygons.pts_arr) :
        func_p.append((sp.Lambda(tuple(mus_symbols),px),sp.Lambda(tuple(mus_symbols),py)))

    transforms = []
    from itertools import combinations
    for polygon in list_polygons.polygons :
        a,b,c = None,None,None
        pts = None
        for (i,j,k) in combinations(chain(*polygon),3) :
            pts = list_polygons.pts_arr[i],list_polygons.pts_arr[j],list_polygons.pts_arr[k]
            # Vector product
            ax,ay = pts[1][0] - pts[0][0],pts[1][1] - pts[0][1]
            bx,by = pts[2][0] - pts[0][0],pts[2][1] - pts[0][1]
            expr = sp.Eq(sp.simplify(ax*by - ay*bx),0)
            if mus.numerical_results(expr).all() : continue

            a,b,c = i,j,k
            break

        if a is None :
            # print("No valid triangle in this polygon")
            transforms.append(None)
            continue

        Tx,Ty = Tx_base,Ty_base
        for c in range(len(p)) :
            for d in range(len(p[c])) :
                Tx = Tx.replace(p[c][d],sp.Lambda(tuple(mus_symbols),pts[c][d]))
                Ty = Ty.replace(p[c][d],sp.Lambda(tuple(mus_symbols),pts[c][d]))
        transforms.append((sp.Poly(sp.simplify(Tx),*coords).as_expr(),sp.Poly(sp.simplify(Ty),*coords).as_expr()))

    return transforms




def fest_construct_triangles_form() :
    np.set_printoptions(linewidth = 125)

    h,w,a,b = sp.symbols("h,w,a,b",real = True)
    h_bar,w_bar,a_bar,b_bar = sp.symbols("h_bar,w_bar,a_bar,b_bar",real = True)
    polygons = [
        [[0,2,3,1],[4,5,7,6]],
        [[4,6,7,5]],
    ]
    pts_arr = [
        (0,0),(0,1),(1,0),(1,1),
        (a,b),(a,b+h),(a+w,b),(a+w,b+h),
    ]
    from muslist import MusList
    mus = MusList([],[h,w,a,b],[0.,0.,0.,0.],[0.5,0.5,0.5,0.5],[h_bar,w_bar,a_bar,b_bar])
    mus.make_ranges(10)
    omegas = list(sp.Function("O" + str(i),nargs = 2) for i in range(len(polygons)))
    polylist = PolygonList(pts_arr,polygons,omegas)

    polylist.compute_intersections(mus)
    polylist.compute_gamma(mus)

    x,y = sp.symbols("x y")
    u,v = sp.Function("u",nargs = 2),sp.Function("v",nargs = 2)
    grad_u = sp.Matrix(2,1,[sp.diff(u(x,y),x),sp.diff(u(x,y),y)])
    grad_v = sp.Matrix(2,1,[sp.diff(v(x,y),x),sp.diff(v(x,y),y)])
    form = sp.Integral((grad_u.T * grad_v)[0,0] * omegas[1](x,y) + u(x,y) * v(x,y) * omegas[0](x,y),x,y)

    # construct_triangles_form(mus,polylist,form,[u,v],[x,y])
    final_polygons,form,new_coords = construct_triangulation(mus,polylist,form,[u,v],[x,y])

    print("Polygons ",len(final_polygons),": ")
    for omega,triangle in final_polygons :
        print("  ",omega,"->",triangle)
    print()

    print("Form ",len(form),": ")
    for cst,mus,xys in form :
        print("  Cst :",cst)
        print("  Mus :",mus)
        print("  Xys :",xys)
        print()


def fest_test() :
    np.set_printoptions(linewidth = 125)

    h,w,a,b = sp.symbols("h,w,a,b",real = True)
    h_bar,w_bar,a_bar,b_bar = sp.symbols("h_bar,w_bar,a_bar,b_bar",real = True)
    polygons = [[[0, 2, 3, 1], [4, 5, 7, 6]], [[7, 5, 4]]]
    pts_arr = [
        (0,0),(0,1),(1,0),(1,1),
        (a,b),(a,b+h),(a+w,b),(a+w,b+h),
    ]
    from muslist import MusList
    mus = MusList([],[h,w,a,b],[0.,0.,0.,0.],[0.5,0.5,0.5,0.5],[h_bar,w_bar,a_bar,b_bar])
    mus.make_ranges(10)
    omegas = list(sp.Function("O" + str(i),nargs = 2) for i in range(len(polygons)))
    polylist = PolygonList(pts_arr,polygons,omegas)

    polylist.compute_intersections(mus)
    polylist.compute_gamma(mus)

    print(polylist.is_triangle_in_polygon(mus,1,(7,5,4)))


def construct() :
    np.set_printoptions(linewidth = 250)

    l = sp.symbols("h",real = True,positive = True)
    l_bar = sp.symbols("h_bar",real = True, positive = True)
    pts_arr = [
        (sp.Rational(0)  ,sp.Rational(0)),(sp.Rational(0)  ,sp.Rational(1)),
        (sp.Rational(1)  ,sp.Rational(1)),(sp.Rational(1)  ,sp.Rational(0)),
        (sp.Rational(1,3),sp.Rational(0)),(sp.Rational(1,3),l             ),
        (sp.Rational(2,3),             l),(sp.Rational(2,3),sp.Rational(0))
    ]
    polygons = [
        [[4,5,6,7,3,2,1,0]],
        [[7,6,5,4]],
    ]

    mu = sp.symbols("mu",real = True)
    from muslist import MusList
    mus = MusList([mu],[l],[0.],[1.],[l_bar])
    mus.make_ranges(10)

    omegas = list(sp.Function("O" + str(i),nargs = 2) for i in range(len(polygons)))
    polylist = PolygonList(pts_arr,polygons,omegas)
    polylist.compute_intersections(mus)
    polylist.compute_gamma(mus)

    x,y = sp.symbols("x y")
    f = sp.Function("f",nargs = 2)
    u,v = sp.Function("u",nargs = 2),sp.Function("v",nargs = 2)
    grad_u = sp.Matrix(2,1,[sp.diff(u(x,y),x),sp.diff(u(x,y),y)])
    grad_v = sp.Matrix(2,1,[sp.diff(v(x,y),x),sp.diff(v(x,y),y)])
    def dot(a,b) : return (a.T * b)[0,0]
    form = sp.Integral(dot(grad_u,grad_v) * (omegas[0](x,y) + omegas[1](x,y)) - mu * f(x,y) * v(x,y) * omegas[1](x,y),x,y)

    with open("test.geo","w+") as file :
        from exporter import export_polygons
        export_polygons(file,mus,polylist)

    final_polygons,form,new_coords = construct_triangulation(mus,polylist,form,[u,v,f],[x,y])
    poly_symbols,poly_lines = zip(*final_polygons)

    polylist.polygons = list(map(lambda x : [x],poly_lines))
    polylist.symbols  = list(poly_symbols)
    with open("test2.geo","w+") as file :
        from exporter import export_polygons
        export_polygons(file,mus,polylist)

    from merge_polygons import merge_polygons_from_form
    merge_polygons_from_form(mus,polylist,form,new_coords)
    with open("test3.geo","w+") as file :
        from exporter import export_polygons
        export_polygons(file,mus,polylist)


    print("Points ",len(pts_arr))
    for k,(x,y) in enumerate(pts_arr) :
        print("  ",k,"->",(x,y))
    print()

    print("Polygons ",len(polylist.polygons),": ")
    for omega,triangle in zip(polylist.symbols,polylist.polygons) :
        print("  ",omega,"->",triangle)
    print()

    print("Form ",len(form),": ")
    for cst,mus,xys in form :
        print("  Cst :",cst)
        print("  Mus :",mus)
        print("  Xys :",xys)
        print()


def construct_2() :
    np.set_printoptions(linewidth = 250)

    w,h = sp.symbols("w h ",real = True)
    w_bar,h_bar = sp.symbols("w_bar h_bar",real = True)
    pts_arr = [
        (sp.Rational(0)  ,sp.Rational(0)  ),(sp.Rational(0)  ,sp.Rational(1)  ),
        (sp.Rational(1)  ,sp.Rational(1)  ),(sp.Rational(1)  ,sp.Rational(0)  ),
        (sp.Rational(1,3),sp.Rational(1,3)),(sp.Rational(1,3),h               ),
        (w               ,h               ),(               w,sp.Rational(1,3))
    ]
    polygons = [
        [[3,2,1,0],[4,5,6,7]],
        [[7,6,5,4]],
    ]

    mu = sp.symbols("mu",real = True)
    from muslist import MusList
    mus = MusList([mu],[w,h],[1/3,1/3],[2/3,2/3],[w_bar,h_bar])
    mus.make_ranges(10)

    omegas = list(sp.Function("O" + str(i),nargs = 2) for i in range(len(polygons)))
    polylist = PolygonList(pts_arr,polygons,omegas)
    polylist.compute_intersections(mus)
    polylist.compute_gamma(mus)

    x,y = sp.symbols("x y")
    f = sp.Function("f",nargs = 2)
    u,v = sp.Function("u",nargs = 2),sp.Function("v",nargs = 2)
    # eps = sp.Dummy("eps")
    grad_u = sp.Matrix(2,1,[sp.diff(u(x,y),x),sp.diff(u(x,y),y)])
    grad_v = sp.Matrix(2,1,[sp.diff(v(x,y),x),sp.diff(v(x,y),y)])
    def dot(a,b) : return (a.T * b)[0,0]
    form = sp.Integral(dot(grad_u,grad_v) * (omegas[0](x,y) + omegas[1](x,y)) - mu * f(x,y) * v(x,y) * omegas[1](x,y),x,y)

    with open("test.geo","w+") as file :
        from exporter import export_polygons
        export_polygons(file,mus,polylist)

    final_polygons,form,new_coords = construct_triangulation(mus,polylist,form,[u,v,f],[x,y])
    poly_symbols,poly_lines = zip(*final_polygons)

    polylist.polygons = list(map(lambda x : [x],poly_lines))
    polylist.symbols  = list(poly_symbols)
    with open("test2.geo","w+") as file :
        from exporter import export_polygons
        export_polygons(file,mus,polylist)

    from merge_polygons import merge_polygons_from_form
    merge_polygons_from_form(mus,polylist,form,new_coords)

    transforms = compute_linear_transforms(mus,polylist,[x,y])

    with open("test3.geo","w+") as file :
        from exporter import export_polygons
        export_polygons(file,mus,polylist)

    print("Points ",len(pts_arr))
    for k,(x,y) in enumerate(pts_arr) :
        print("  ",k,"->",(x,y))
    print()

    print("Polygons ",len(polylist.polygons),": ")
    for omega,triangle,(tfx,tfy) in zip(polylist.symbols,polylist.polygons,transforms) :
        print("  ",omega,"->",triangle,"\t#  x_ref =",tfx,"\t y_ref =",tfy)
    print()

    print("Form ",len(form),": ")
    for cst,mus,xys in form :
        print("  Cst :",cst)
        print("  Mus :",mus)
        print("  Xys :",xys)
        print()



# construct()
construct_2()


