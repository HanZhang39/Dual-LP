# A single function module for formulating dual LP
from scipy.optimize import linprog
import numpy as np

def dual(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None):

    c_new = b_ub + (b_eq + [-i for i in b_eq] if b_eq is not None else [])

    if A_eq is None:
        A_ub_t = list(zip(*A_ub))

        A_eq_new = [list(a_ub)  for a_ub in A_ub_t]
    else:
        A_eq_t = list(zip(*A_eq))
        A_ub_t = list(zip(*A_ub))

        A_eq_new = [list(a_ub) + list(a_eq) +[-i for i in a_eq] for a_ub, a_eq in zip(A_ub_t, A_eq_t)]
    b_eq_new = [-c_ for c_ in c]

    l = 2 * (0 if A_eq is None else len(A_eq)) + (0 if A_ub is None else len(A_ub))
    A_ub_new, b_ub_new = zip(*[(list(np.zeros(i)) + [-1] + list(np.zeros(l - 1 - i)), 0)
                               for i in range(l)])
    return linprog(c_new, A_ub=A_ub_new, b_ub=b_ub_new, A_eq=A_eq_new, b_eq=b_eq_new)

if __name__ == "__main__":
    c = [-1, 4]
    A = [[-3, 1], [1, 2], [0, -1]]
    b = [6, 4, 3]
    res = linprog(c, A_ub=A, b_ub=b, bounds=((None, None), (None, None)), options={"disp": True})
    print(f"Primal: \n{res}")
    print(f"Dual: \n{dual(c, A, b)}")
