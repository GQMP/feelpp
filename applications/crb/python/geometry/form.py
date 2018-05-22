
# Form list
#   (
from collections import defaultdict
from itertools import chain

import sympy as sp


def merge_integrals(expression) :
    result,exprs = sp.expand(expression).as_coeff_add()

    all_integrals = defaultdict(lambda : sp.Integer(0))
    for x in exprs :
        a,b = sp.factor(x).as_coeff_mul()
        integral = None
        for e in b :
            if e.func is not sp.Integral :
                a *= e
            elif integral is None :
                integral = e
            else :
                # todo : correctly do integrals in Mul
                raise NotImplementedError("Encounter multiplications of integrals, but not supported.")

        if integral is None :
            result += a
        else :
            all_integrals[tuple(integral.args[1:])] += a * integral.args[0]

    for variables,element in all_integrals.items() :
        a,b = sp.factor(element).as_coeff_mul()
        new_expr = sp.Integer(1)
        for e in b :
            if any(v[0] in e.free_symbols for v in variables) :
                new_expr *= e
            else :
                a *= e

        if new_expr != 0 :
            result += a * sp.Integral(new_expr,*variables)

    return result


def simplify_form(form_list,left,right) :
    def reduce_left(form_list,left,right) :
        for i in range(len(form_list)) :
            for j in range(i+1,len(form_list)) :
                if form_list[i][2] == form_list[j][2] :
                    m = form_list[i][0] * form_list[i][1] + form_list[j][0] * form_list[j][1]
                    x = form_list[i][2]

                    c,factors = sp.factor(m).as_coeff_mul()
                    m = sp.Integer(1)
                    for f in factors :
                        if any(v in f.free_symbols for v in left) :
                            m *= f
                        else :
                            c *= f

                    del form_list[j]
                    del form_list[i]
                    form_list.append((c,sp.simplify(m),x))
                    return True
        return False

    def reduce_right(form_list,left,right) :
        for i in range(len(form_list)) :
            for j in range(i+1,len(form_list)) :
                if form_list[i][1] == form_list[j][1] :
                    m = form_list[i][1]
                    x = form_list[i][0] * form_list[i][2] + form_list[j][0] * form_list[j][2]

                    c,factors = merge_integrals(x).as_coeff_mul()
                    x = sp.Integer(1)
                    for f in factors :
                        if any(v in f.free_symbols for v in right) :
                            x *= f
                        else :
                            c *= f

                    del form_list[j]
                    del form_list[i]
                    form_list.append((c,m,x))
                    return True
        return False

    while reduce_left(form_list,left,right) or reduce_right(form_list,left,right) : pass

    i = 0
    while i < len(form_list) :
        c,m,x = form_list[i]
        if m == 0 : del form_list[i]
        elif x == 0 : del form_list[i]
        else : i += 1

    return form_list


def separate_form(form,left,right) :
    """
    Separate variables in form
    :param form: Form to separate
    :param left: Symbols to keep at left (mus)
    :param right: Symbols to keep at right (derivatives/functions,...)
    :return: List of couples, Sum[ Mul(constant,left variables,right variables) ]
    """
    list_terms = [] # = defaultdict(lambda : Integer(0))

    form = sp.expand(form)
    cst,exprs = form.as_coeff_add()
    # cst = sp.simplify(cst)
    # if cst != 0 :
    #     list_terms.append((sp.Integer(cst),sp.Integer(1),sp.Integer(1)))

    for x in chain([cst],exprs) :
        # Try factor before to expand the mus on integrals but
        # it seems to work only if the integral is alone...
        xys,mus = sp.Integer(1),sp.Integer(1)
        cst,factors = sp.factor(x).as_coeff_mul()
        for f in factors :
            if any(x in f.free_symbols for x in right) and any(x in f.free_symbols for x in left) :
                print("Warning : seems not to be separable :",f,"by",(left,right))
                xys *= f
            elif any(x in f.free_symbols for x in right) :
                xys *= f
            elif any(x in f.free_symbols for x in left) :
                mus *= f
            else :
                cst *= f
        if cst != 0 and mus != 0 :
            list_terms.append((cst,mus,xys))

    return simplify_form(list_terms,left,right)


def replace_func_in_derivatives(expression,functions,derivatives) :
    if len(expression.args) == 0 : return expression
    if expression.func is not sp.Derivative or all(x not in derivatives for x in expression.variables) :
        return expression.func(*[replace_func_in_derivatives(arg,functions,derivatives) for arg in expression.args])

    expr = expression.args[0]
    for (f,r) in functions :
        expr = expr.replace(f,lambda *args : r)
    return expression.func(expr,*expression.args[1:])


def apply_transform_opt(expression,functions,xys,uvs) :
    if len(expression.args) == 0 : return expression
    if expression.func is not sp.Integral or any(x not in expression.variables for (x,e) in xys) :
        return expression.func(*[apply_transform_opt(arg,functions,xys,uvs) for arg in expression.args])

    xys_symbols = [x for (x,e) in xys]
    xys_exprs   = [sp.factor(e) for (x,e) in xys]
    uvs_symbols = [x for (x,e) in uvs]
    uvs_exprs   = [sp.factor(e) for (x,e) in uvs]

    # substitute funcs(x,y) by funcs(r(x,y),t(x,y)) then doit(integrals = False)
    # func now are expressed in functions of new_coordinates with all derivatives needed...
    change_vars_funcs = [sp.Function("_T_" + str(u))(*xys_symbols) for u in uvs_symbols]
    expr_funcs_uvs_by_xys = expression.args[0].subs(
        zip([f(*xys_symbols) for f in functions],
            [f(*change_vars_funcs) for f in functions])
    ).doit(integrals = False)

    # replace functions of new_coordinates only where needed (/!\ derivatives only at the moment)
    expr_uvs_by_xys = replace_func_in_derivatives(expr_funcs_uvs_by_xys,list(zip(change_vars_funcs,uvs_exprs)),xys_symbols)

    # Change functions of new_coordinates by uvs then xys by their expressions
    expr_funcs_uvs = sp.simplify(expr_uvs_by_xys.doit()) \
        .subs(zip(change_vars_funcs,uvs_symbols)).subs(xys)

    # substitute r(x(r,t),y(r,t)),t(x(r,t),y(r,t)) by r,t in case of simplification failure
    expr_funcs_uvs_substituded = sp.simplify(sp.expand(expr_funcs_uvs)).subs(
        zip([sp.simplify(uvs_expr) for uvs_expr in uvs_exprs],uvs_symbols)
    ).doit()

    # don't forget the jacobian !
    # Todo : Check a better way to compute the jacobian ...
    jacobian = sp.Abs(sp.det(sp.Matrix(len(uvs_symbols),len(xys_exprs),[x.diff(u) for x in xys_exprs for u in uvs_symbols]))).doit() # +
    expr_integral = sp.simplify(expr_funcs_uvs_substituded * jacobian)

    if not any(x not in expr_integral.free_symbols for x in xys) :
        raise NotImplementedError("Not able to transform the integral : " + sp.srepr(expression))

    new_expr = sp.Integral(expr_integral,*uvs_symbols)
    return new_expr


# todo : optimize because it take really too many seconds... ; and too many hours for complicated changes...
# todo : try to suppress the uvs_exprs, like a classic integral change...
def apply_transform(expression,functions,xys,uvs) :
    if len(expression.args) == 0 : return expression
    if expression.func is not sp.Integral or any(x not in expression.variables for (x,e) in xys) :
        return expression.func(*[apply_transform(arg,functions,xys,uvs) for arg in expression.args])

    xys_symbols = [x for (x,e) in xys]
    xys_exprs   = [sp.factor(e) for (x,e) in xys]
    uvs_symbols = [x for (x,e) in uvs]
    uvs_exprs   = [sp.factor(e) for (x,e) in uvs]

    # substitute funcs(x,y) by funcs(r(x,y),t(x,y)) then doit(integrals = False)
    # func now are expressed in new_coordinates with all derivatives needed...
    # change_vars_funcs = [sp.Function("_T_" + str(u))(*xys_symbols) for u in uvs_symbols]
    expr_funcs_uvs_by_xys = expression.args[0].subs(
        zip([f(*xys_symbols) for f in functions],
            [f(*uvs_exprs) for f in functions])
    ).doit(integrals = False)
    # expr_funcs_uvs_by_xys = expr_funcs_uvs_by_xys.subs(zip(change_vars_funcs,uvs_exprs)).doit()
    expr_funcs_uvs = sp.simplify(expr_funcs_uvs_by_xys).subs(zip(xys_symbols,xys_exprs))

    # substitute r(x(r,t),y(r,t)),t(x(r,t),y(r,t)) by r,t in case of simplification failure
    expr_funcs_uvs = sp.simplify(sp.expand(expr_funcs_uvs)).subs(
        zip([sp.expand(sp.simplify(uvs_expr.subs(xys))) for uvs_expr in uvs_exprs],uvs_symbols)
    ).doit() # ++++

    # don't forget the jacobian !
    jacobian = sp.det(sp.Matrix(len(uvs_symbols),len(xys_exprs),[x.diff(u) for x in xys_exprs for u in uvs_symbols])).doit() # +
    expr_integral = sp.simplify(expr_funcs_uvs * jacobian)

    if not any(x not in expr_integral.free_symbols for x in xys) :
        raise NotImplementedError("Not able to transform the integral : " + sp.srepr(expression))

    new_expr = sp.Integral(expr_integral,*uvs_symbols)
    return new_expr

#
# # todo : optimize because it take really too many seconds... ; and too many hours for complicated changes...
# # todo : try to suppress the uvs_exprs, like a classic integral change...
# def apply_transform(expression,functions,xys,uvs) :
#     if len(expression.args) == 0 : return expression
#     if expression.func is not sp.Integral or any(x not in expression.variables for (x,e) in xys) :
#         return expression.func(*[apply_transform(arg,functions,xys,uvs) for arg in expression.args])
#
#     xys_symbols = [x for (x,e) in xys]
#     xys_exprs   = [e for (x,e) in xys]
#     uvs_symbols = [x for (x,e) in uvs]
#     uvs_exprs   = [e for (x,e) in uvs]
#
#     # substitute funcs(x,y) by funcs(r(x,y),t(x,y)) then doit(integrals = False)
#     # func now are expressed in new_coordinates with all derivatives needed...
#     expr_funcs_uvs_by_xys = expression.args[0].subs(
#         zip([f(*xys_symbols) for f in functions],
#             [f(*uvs_exprs) for f in functions])
#     ).doit(integrals = False)
#     expr_funcs_uvs = expr_funcs_uvs_by_xys.subs(zip(xys_symbols,xys_exprs))
#
#     # substitute r(x(r,t),y(r,t)),t(x(r,t),y(r,t)) by r,t in case of simplification failure
#     expr_funcs_uvs = sp.simplify(sp.expand(expr_funcs_uvs)).subs(
#         zip([sp.expand(sp.simplify(uvs_expr.subs(xys))) for uvs_expr in uvs_exprs],uvs_symbols)
#     ).doit() # ++++
#     # don't forget the jacobian !
#     jacobian = sp.det(sp.Matrix(len(uvs_symbols),len(xys_exprs),[x.diff(u) for x in xys_exprs for u in uvs_symbols])).doit() # +
#     expr_integral = sp.simplify(expr_funcs_uvs * jacobian)
#
#     if not any(x not in expr_integral.free_symbols for x in xys) :
#         raise NotImplementedError("Not able to transform the integral : " + sp.srepr(expression))
#
#     new_expr = sp.Integral(expr_integral,*uvs_symbols)
#     return new_expr
#

def mul_ind_integral(expression,ind) :
    if len(expression.args) == 0 : return expression
    if expression.func is not sp.Integral :
        return expression.func(*[mul_ind_integral(arg,ind) for arg in expression.args])

    new_expr = sp.Integral(expression.args[0] * ind(*[arg[0] for arg in expression.args[1:]]) ,*expression.args[1:])
    return new_expr

