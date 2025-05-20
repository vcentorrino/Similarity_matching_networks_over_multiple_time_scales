# Code to run the simulation in Appendix C, paragraph "Empirical Evidence for Conjectures in Section 5.3"

import numpy as np
import matplotlib
from matplotlib import rc
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
import random

# PLOT SETTINGS
font = {'size': 16}
plt.rc('font', **font)
rc('text', usetex=True)
rc('font', family='serif')

save = False
scalar = False
###
matrix = False
run_matrix = False
run_C = False
plot = True
###
cost_S = False
# ============================================================================
#                           Functions
# ============================================================================
def svd_eigenvalues(matrix, full_decomposition):
    """
    Computes the singular values of a matrix using Singular Value Decomposition (SVD).

    Parameters:
        matrix (ndarray): The input matrix to decompose.
        full_decomposition (bool): If True, returns (U, S, VT); otherwise, returns only S.

    Output:
        ndarray or tuple: Singular values if full_decomposition is False;
                          (U, S, VT) otherwise, where:
                           -  U = Unitary matrix with left singular vectors as columns.
                           -  S = 1-D array of singular values.
                           - VT = Unitary matrix with right singular vectors as rows.
    """
    U, S, VT = np.linalg.svd(matrix)
    return (U, S, VT) if full_decomposition else S


def simulate_dynamics_3rd_layer(W_0, C_X, t_span, dt):
    """
    Simulates the dynamics \dot W = -4W + 4(WC_X W^T)^(-1/3) WC_X

    Parameters:
        W_0 (ndarray): Initial condition for W.
        C_X (ndarray): Covariance matrix.
        t_span (tuple): Time range for simulation (start, end).
        dt (float): Time step size for simulation.
    """
    m, n = W_0.shape

    def system(t, W_flat):
        W = W_flat.reshape(m, n)
        WCX = W @ C_X
        W_CX_WT = WCX @ W.T
        W_CX_WT_inv_13 = fractional_matrix_power(W_CX_WT, -1/3)
        dW_dt = -4 * W + 4 * W_CX_WT_inv_13 @ WCX
        return dW_dt.ravel()

    z0 = W_0.ravel()
    t_eval = np.arange(t_span[0], t_span[1], dt)

    sol = solve_ivp(system, t_span, z0, t_eval=t_eval, method='RK45')

    t = sol.t
    W = sol.y.reshape(m, n, -1)

    rank_WWT = []
    for i in range(W.shape[2]):
        WWT = W[:, :, i] @ W[:, :, i].T
        rank_WWT.append(np.linalg.matrix_rank(WWT))
    U_W, eigenvalues_W, VT_W = svd_eigenvalues(W[:, :, -1], True)
    print(f"optimal eigenvalues: {eigenvalues_W}")
    return t, W, rank_WWT

# ============================================================================
#                             Scalar case
# ============================================================================
if scalar:
    def func_w(w, x):
        return 2*w**2 - 3 * (x**4)**(1/3) * (w**4)**(1/3)
    def scalar_ode(t, w, x):
        return -4*w + 4* x**(4/3) * np.sign(w)*np.abs(w)**(1/3)
    x = 2
    tf = 6
    w_values = np.linspace(-tf, tf, 1000)
    if 0 not in w_values:
        w_values = np.sort(np.append(w_values, 0))
    y_values = func_w(w_values, x)
    time_span = (0, tf)
    time_eval = np.linspace(*time_span, 100)
    # Plot Figure 3 in the manuscript
    plt.figure(figsize=(10, 4))#(14, 4)
    plt.scatter([tf], [x**2], color='red', label=rf'$\pm x^2$', zorder=3)
    plt.scatter([tf], [-x**2], color='red', zorder=3)
    for w0 in w_values:
        solution = solve_ivp(scalar_ode, time_span, [w0], args=(x,), t_eval=time_eval)
        plt.plot(solution.t, solution.y[0], alpha=0.3, color='royalblue')
    plt.xlabel(r'$t$', fontsize=20)
    plt.ylabel(r'$w(t)$', fontsize=20)
    plt.grid(True, alpha=0.7)
    plt.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    plt.legend()
    plt.xlim(0, tf+0.05)
    if save:
        plt.savefig(f'dyn_w_scalar.eps', bbox_inches='tight')
    plt.show()

# ============================================================================
#                             Matrix Case
# ============================================================================
elif matrix:
    n, m, k = 30, 3, 100
    t_span = (0, 30)
    dt = 0.01
    num_runs = 10000

    simulation_path = 'simulation_rank_3_30_100.npz'    
    if not run_C:
        np.random.seed(20)
        X = 10*np.random.randn(n, k)  # Random matrix from standard normal distribution
        C_X = X @ X.T / k + np.eye(n)
        U_C, eigenvalues_C, VT_C = svd_eigenvalues(C_X, True)
        print(f"eigenvalues C_X: {eigenvalues_C}")
    if run_matrix:
        all_ranks = []
        W0_list = []
        W_opt_list = []
        if run_C:
            C_list = []
        for run in range(num_runs):
            print(f" run = {run}")
            if run_C:
                X = 10*np.random.randn(n, k)  # Random matrix from standard normal distribution
                C_X = X @ X.T / k + 2*np.eye(n)
                C_list.append(C_X)
            W_0 = np.random.randn(m, n)
            W0_list.append(W_0)
            t_dyn, W_dyn, rank_WWT = simulate_dynamics_3rd_layer(W_0, C_X, t_span, dt)
            all_ranks.append(rank_WWT)
            W_opt = W_dyn[:, :, -1]
            W_opt_list.append(W_opt)
        np.savez_compressed(simulation_path, C_X = C_X,
                            t_dyn = t_dyn, W_opt_list=W_opt_list, all_ranks=all_ranks, W0_list = W0_list)
    else:
        load_both = np.load(simulation_path)
        t_dyn = load_both['t_dyn']
        W_opt_list = load_both['W_opt_list']
        W0_list = load_both['W0_list']
        all_ranks = load_both['all_ranks']
if plot:
    def average_top_m(W_opt_list):
        m = W_opt_list.shape[0]
        all_top_m_eigenvalues = []
        all_top_m_eigenvectors = []
        for W in W_opt_list:
            U_W, top_m_eigenvalues, top_m_eigenvectors = svd_eigenvalues(W, True)
            all_top_m_eigenvalues.append(top_m_eigenvalues[:m])
            all_top_m_eigenvectors.append(np.abs(top_m_eigenvectors.T))
            #print(np.around(top_m_eigenvectors.T,4))
        all_top_m_eigenvalues = np.array(all_top_m_eigenvalues)
        all_top_m_eigenvectors = np.array(all_top_m_eigenvectors)

        avg_eigenvalues = np.mean(all_top_m_eigenvalues, axis=0)
        avg_eigenvectors = np.mean(all_top_m_eigenvectors, axis=0)
        return avg_eigenvalues, avg_eigenvectors

    num_runs = 10000
    
    simulation_path_1 = 'simulation_rank_2_10_100.npz'
    simulation_path_2 = 'simulation_rank_3_30_100.npz'
    simulation_path_3 = 'simulation_rank_5_50_500.npz'

    ###### 1
    load_both_1 = np.load(simulation_path_1)
    t_dyn_1 = load_both_1['t_dyn']
    all_ranks_1 = load_both_1['all_ranks']
    ###### 2
    load_both_2 = np.load(simulation_path_2)
    t_dyn_2 = load_both_2['t_dyn']
    all_ranks_2 = load_both_2['all_ranks']
    ###### 2
    load_both_3 = np.load(simulation_path_3)
    t_dyn_3 = load_both_3['t_dyn']
    all_ranks_3 = load_both_3['all_ranks']

    # Plot Figure 8 in the manuscript
    plt.figure(figsize=(14, 4))
    for i in range(num_runs):
        plt.plot(t_dyn_1, all_ranks_1[i], label=rf"$m = 2$, $n = 10$" if i == 0 else "", color ='royalblue')
        plt.plot(t_dyn_2, all_ranks_2[i], label=rf"$m = 3$, $n = 30$" if i == 0 else "", color ='red')
        plt.plot(t_dyn_3, all_ranks_3[i], label=rf"$m = 5$, $n = 50$" if i == 0 else "", color ='green')
    plt.xlabel(r"$t$")
    plt.ylabel(r"Rank of $W(t) W(t)^\top$")
    plt.xlim([0, 30])
    plt.legend(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', labelsize=16)
    if save:
        plt.savefig(f'rank_dyn_w_.eps', bbox_inches='tight')
    plt.show()

    table1 = False
    table2 = False
    def latex_row_safe(name, values):
            formatted_values = [
                f"{v:.3f}" if isinstance(v, (int, float, np.floating)) else str(v)
                for v in values
            ]
            row = f"{name} & " + " & ".join(formatted_values) + r" \\"
            return row
    if table1:
        latex_table_CX = r"""\begin{tabular}{lcccccccccc}
        \toprule
        Matrix & $\lambda_1^C$ & $\lambda_2^C$ & $\lambda_3^C$ & $\lambda_4^C$ & $\lambda_5^C$ & $\lambda_6^C$ & $\lambda_7^C$ \\
        \midrule
        """
        latex_table_CX += latex_row_safe("C$_X$ 1", eigenvalues_Cx_1[:7]) + "\n"
        latex_table_CX += latex_row_safe("C$_X$ 2", eigenvalues_Cx_2[:7]) + "\n"
        latex_table_CX += latex_row_safe("C$_X$ 3", eigenvalues_Cx_3[:7]) + "\n"
        latex_table_CX += r"\bottomrule" + "\n\end{tabular}"

        # -------------------------------
        # LaTeX Table 2: average_top_m
        # -------------------------------
        latex_table_avg = r"""\begin{tabular}{lccccc}
        \toprule
        Dataset & $\hat{\lambda}_1$ & $\hat{\lambda}_2$ & $\hat{\lambda}_3$ & $\hat{\lambda}_4$ & $\hat{\lambda}_5$ \\
        \midrule
        """
        latex_table_avg += latex_row_safe("avg$_1$", avg_eigenvalues_1.tolist() + [""] * (5 - len(avg_eigenvalues_1))) + "\n"
        latex_table_avg += latex_row_safe("avg$_2$", avg_eigenvalues_2.tolist() + [""] * (5 - len(avg_eigenvalues_2))) + "\n"
        latex_table_avg += latex_row_safe("avg$_3$", avg_eigenvalues_3.tolist()) + "\n"
        latex_table_avg += r"\bottomrule" + "\n\end{tabular}"

        print("Top 6 Eigenvalues of C_X:\n")
        print(latex_table_CX)
        print("\nAverage Top-m Eigenvalues:")
        print(latex_table_avg)
    if table2:
        print("First Block:\n")
        print("C_X:")
        print(np.around(np.abs(U_Cx_1.T[:2, :]),4))
        print("\nAverage Top-m Right Eigenvectors:\n")
        print(np.around(avg_eigenvectors_1[:, :2],5))

        print("Second Block:\n")
        print("C_X:")
        print(np.around(np.abs(U_Cx_2.T[:2, :]),4))
        print("\nAverage Top-m Right Eigenvectors:")
        print(np.around(avg_eigenvectors_2[:, :2],5))

        print("Third Block:\n")
        print("C_X:")
        print(np.around(np.abs(U_Cx_3.T[:2, :]),4))
        print("\nAverage Top-m Right Eigenvectors:")
        print(np.around(avg_eigenvectors_3[:, :2],5))

elif cost_S:
    n, m, k = 5, 3, 10
    t_span = (0, 30)
    dt = 0.01
    np.random.seed(20)

    X = np.random.randn(n, k)  # Random matrix from standard normal distribution
    C_X = X @ X.T / k + np.eye(n)
    U_C, eigenvalues_C, VT_C = svd_eigenvalues(C_X, True)
    Cx = U_C @ np.diag(eigenvalues_C) @ VT_C
    V_perm = U_C.T.copy()
    V_perm[[0, 3]] = V_perm[[3, 0]]

    varying_values = np.linspace(-10, 10, 201)
    costs1 = []
    costs1_p = []
    costs2 = []
    costs3 = []
    varying_values = np.append(varying_values, eigenvalues_C[-1])
    varying_values = np.append(varying_values, eigenvalues_C[0])
    varying_values = np.append(varying_values, eigenvalues_C[1])
    varying_values = np.append(varying_values, eigenvalues_C[2])
    varying_values = np.append(varying_values, eigenvalues_C[3])
    varying_values = np.sort(varying_values)
    
    for lambda_m in varying_values:
        eigenvalues_W1 = eigenvalues_C[:m]
        VT_W1 = U_C.T
        eig_W1 = np.zeros((m, n))
        eig_W1[:m, :m] = np.diag(eigenvalues_W1)
        eig_W1[0, 0] = lambda_m
        W1_p = eig_W1 @ V_perm
        W1 = eig_W1 @ VT_W1 
        eig_W2 = np.zeros((m, n))
        eig_W2[:m, :m] = np.diag(eigenvalues_W1)
        eig_W2[1, 1] = lambda_m
        W2 = eig_W2 @ VT_W1
        eig_W3 = np.zeros((m, n))
        eig_W3[:m, :m] = np.diag(eigenvalues_W1)
        eig_W3[2, 2] = lambda_m
        W3 = eig_W3 @ VT_W1

        # Cost function components
        norm_term1_1 = np.linalg.norm(W1, 'fro')
        term1_1 = 2 * norm_term1_1**2
        WCWT_root1 = fractional_matrix_power(W1 @ C_X @ W1.T, 1/3)
        norm_term2_1 = np.linalg.norm(WCWT_root1, 'fro')
        term2_1 = 3 * norm_term2_1**2
        cost1 = term1_1 - term2_1
        costs1.append(cost1)

        norm_term1_1_p = np.linalg.norm(W1_p, 'fro')
        term1_1_p = 2 * norm_term1_1_p**2
        WCWT_root1_p = fractional_matrix_power(W1_p @ C_X @ W1_p.T, 1/3)
        norm_term2_1_p = np.linalg.norm(WCWT_root1_p, 'fro')
        term2_1_p = 3 * norm_term2_1_p**2
        cost1_p = term1_1_p - term2_1_p
        costs1_p.append(cost1_p)

        norm_term1_2 = np.linalg.norm(W2, 'fro')
        term1_2 = 2 * norm_term1_2**2
        WCWT_root2 = fractional_matrix_power(W2 @ C_X @ W2.T, 1/3)
        norm_term2_2 = np.linalg.norm(WCWT_root2, 'fro')
        term2_2 = 3 * norm_term2_2**2
        cost2 = term1_2 - term2_2
        costs2.append(cost2)

        norm_term1_3 = np.linalg.norm(W3, 'fro')
        term1_3 = 2 * norm_term1_3**2
        WCWT_root3 = fractional_matrix_power(W3 @ C_X @ W3.T, 1/3)
        norm_term2_3 = np.linalg.norm(WCWT_root3, 'fro')
        term2_3 = 3 * norm_term2_3**2
        cost3 = term1_3 - term2_3
        costs3.append(cost3)

    # Plot Figure 9 in the manuscript
    plt.figure(figsize=(11, 4.5))
    plt.plot(varying_values, costs1, label=r"$W$ as a function of the $x=1$st Eigenvalue")
    plt.plot(varying_values, costs2,   label=r"$W$ as a function of the $x=2$nd Eigenvalue")
    plt.plot(varying_values, costs3,   label=r"$W$ as a function of the $x=3$rd Eigenvalue")
    plt.plot(varying_values, costs1_p,   label=r"$W$ as a function of the $x=1$st Eigenvalue -- $V$ Permuted")
    plt.xlabel(r"$x$-th eigenvalue of $W$")
    plt.ylabel(r"$S_3(W)$")
    #plt.title(fr"$\Lambda_X = [{eigenvalues_C[0]:.2f}, {eigenvalues_C[1]:.2f}, {eigenvalues_C[2]:.2f}, {eigenvalues_C[3]:.2f}, {eigenvalues_C[4]:.2f}]$")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xlim(-10, 10)
    plt.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    if save:
        plt.savefig(f'costs_w_.eps', bbox_inches='tight')
    plt.show()
