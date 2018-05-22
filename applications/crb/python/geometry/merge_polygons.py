from muslist import MusList
from polygon import PolygonList

import sympy as sp


def replace_addition(expression,symbol1,symbol2,replaced) :
    if len(expression.args) == 0 : return expression
    if expression.func is not sp.Add :
        return expression.func(*[replace_addition(arg,symbol1,symbol2,replaced) for arg in expression.args])

    if symbol1 not in expression.args or symbol2 not in expression.args :
        return expression.func(*[replace_addition(arg,symbol1,symbol2,replaced) for arg in expression.args])

    args = list(expression.args)
    args.remove(symbol1)
    args.remove(symbol2)
    args.append(replaced)

    return expression.func(*[replace_addition(arg,symbol1,symbol2,replaced) for arg in args])


def merge_polygons_from_form(mus : MusList,polylist : PolygonList,form,coord) :
    i = 0
    dummy = sp.Dummy("_d")
    while i < len(polylist.polygons) :
        j = i + 1
        while j < len(polylist.polygons) :
            stop = False
            for _,_,x in form :
                # print(polylist.symbols[i],x.atoms(sp.Function))
                if ((polylist.symbols[i](*coord) in x.atoms(sp.Function))
                  ^ (polylist.symbols[j](*coord) in x.atoms(sp.Function))) :
                    stop = True
                    break
            if stop : j += 1 ; continue

            list_collected = []
            for _,_,x in form :
                replaced = replace_addition(x,polylist.symbols[i](*coord),polylist.symbols[j](*coord),dummy)
                if ((polylist.symbols[i](*coord) in replaced.atoms(sp.Function))
                 or (polylist.symbols[j](*coord) in replaced.atoms(sp.Function))) :
                    break

                list_collected.append(replaced)

            if len(list_collected) != len(form) : j += 1 ; continue
            if not polylist.merge_polygons(mus,i,j) : j += 1 ; continue
            # j -= 1

            for k in range(len(form)) :
                form[k] = (form[k][0],form[k][1],list_collected[k].subs(dummy,polylist.symbols[i](*coord)))
            # j += 1

        i += 1



