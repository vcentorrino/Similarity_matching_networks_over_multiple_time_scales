import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

# PLOT SETTINGS
font = {'size': 15}
plt.rc('font', **font)
rc('text', usetex=True)
rc('font', family='serif')


# =========================================================
#            Functions to generate the matrices
# =========================================================

def generate_Lambda1(m, n):
    """
    Generate an mxn diagonal matrix \Lambda_1:

    Input:
    m = Row dimension.
    n = Column dimension.

    Output:
    \Lambda_1 = Randomly generated mxn diagonal matrix.
     The diagonal entries are uniformly distributed in [-1, 1].

    """
    Lambda_m = np.diag(np.random.uniform(low=-1.0, high=1.0, size=(m))[::-1])
    Lambda1 = np.hstack([Lambda_m, np.zeros((m, n - m))])
    #print("Lambda1 = ")
    #print(np.array_str(Lambda1, precision=2, suppress_small=True))
    return Lambda1

def generate_Lambda2(n):
    """
    Generate an nxn definite positive diagonal matrix \Lambda_2:

    Input:
    n = Matrix dimension.

    Output:
    \Lambda_2: Randomly generated nxn definite positive diagonal matrix.
    """
    Lambda2 = np.diag(np.sort(np.random.rand(n))[::-1])
    #print("Lambda2 = ")
    #print(np.array_str(Lambda2, precision=2, suppress_small=True))
    return Lambda2

def compute_Lambda3(U, Lambda1, Lambda2):
    """
    Compute the matrix \Lambda_3
    
    Input:
          U = Orthogonal nxn matrix
    Lambda1 = Diagonal mxn matrix
    Lambda2 = Diagonal nxn matrix

    Output:
    \Lambda3
    """
    middle_term = Lambda1 @ U @ Lambda2 @ U.T @ Lambda1.T
    #print("middle_term =", middle_term)
    eigvals, eigvecs = np.linalg.eigh(middle_term)
    middle_term_inv_cbrt = eigvecs @ np.diag(eigvals**(-1/3)) @ eigvecs.T
    Lambda3 = U.T @ Lambda1.T @ middle_term_inv_cbrt @ Lambda1 @ U
    #print("Lambda3 =", Lambda3)
    return Lambda3


###########################################################
####################### CASE n = 2  #######################
###########################################################

def rotation_matrix(theta):
    """
    Generate a 2D rotation matrix for a given angle theta.

    Input:
    theta = Angle in radians.

    Output:
    U: 2x2 rotation matrix.
    """
    U = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return U

# =========================================================
#                        Simulation
# =========================================================
# Parameters
theta_vals = np.linspace(0, 2 * np.pi, 10000)
theta_additional_degrees = [90, 180, 270]
theta_additional_radians = np.radians(theta_additional_degrees)
theta_vals_combined = np.concatenate((theta_vals, theta_additional_radians))
theta_vals = np.sort(theta_vals_combined)

off_diag_vals_01 = []
off_diag_vals_10 = []
theta_zeros = []

m, n = 2, 2

# Generate matrices
Lambda1 = np.abs(generate_Lambda1(m, n))
Lambda2 = generate_Lambda2(n)
#np.savez_compressed(path, Lambda1=Lambda1, Lambda2=Lambda2)
#path = 'lambda_matrices_n=2.npz'
#loaded = np.load(path)
#Lambda1 = loaded['Lambda1']
#Lambda2 = loaded['Lambda2']

# Compute the off-diagonal elements of Lambda3 for each theta
for theta in theta_vals:
    U = rotation_matrix(theta)
    Lambda3 = compute_Lambda3(U, Lambda1, Lambda2)
    if Lambda3.shape[0] > 1:
        off_diag_01 = Lambda3[0, 1]
        off_diag_10 = Lambda3[1, 0]
        off_diag_vals_01.append(off_diag_01)
        off_diag_vals_10.append(off_diag_10)
    if np.isclose(off_diag_01, 0, atol=1e-6) and np.isclose(off_diag_10, 0, atol=1e-6):
        theta_zeros.append(theta)

print(f"theta zeros = {theta_zeros}")


save = False
pi_ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
pi_labels = [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]

plt.figure(figsize=(8, 5.5))
plt.plot(theta_vals, off_diag_vals_01, 'r', label=r'$\Lambda_3[0,1]$', linewidth=2)
plt.plot(theta_vals, off_diag_vals_10, '--b', label=r'$\Lambda_3[1, 0]$', linewidth=2)
plt.ylabel(r'$\Lambda_3(\theta)$')
plt.xlabel(r'$\theta$ (radians)')
plt.scatter(theta_zeros, 
            [off_diag_vals_01[np.argmin(np.abs(theta_vals - d))] for d in theta_zeros], 
            color='black', s=50)
plt.xticks(pi_ticks, pi_labels)
plt.xlim([0, 2 * np.pi])
plt.grid(True)
plt.legend(fontsize=13)
if save:
    plt.savefig('plot_for_check_lemma_n=2_radians.pdf', bbox_inches='tight')
plt.tight_layout()
plt.show()



###########################################################
####################### CASE n > 2  #######################
###########################################################
def gram_schmidt_process(matrix):
    """
    Gram-Schmidt orthogonalization process to the columns of a matrix.

    Input:
    matrix = matrix with columns to be orthogonalized.

    Output:
    orthogonal_matrix = A matrix with orthogonalized columns.
    """
    # Number of columns
    n = matrix.shape[1]
    # Initialize an empty array to hold the orthogonal vectors
    orthogonal_matrix = np.zeros_like(matrix)

    for i in range(n):
        # Start with the i-th column of the input matrix
        vec = matrix[:, i]

        # Subtract projections onto all previously orthogonalized vectors
        for j in range(i):
            proj = np.dot(orthogonal_matrix[:, j], vec) * orthogonal_matrix[:, j]
            vec = vec - proj

        # Normalize the vector
        orthogonal_matrix[:, i] = vec / np.linalg.norm(vec)
    return orthogonal_matrix

def is_permutation_matrix(matrix):
    """
    Check if a matrix is a permutation matrix.

    Input:
    matrix = n x n matrix to check.

    Output:
    True if matrix is a permutation matrix, False otherwise.
    """
    # Check if all elements are either 0 or 1
    if not np.all((matrix == 0) | (matrix == 1)):
        return False

    # Check if each row and each column has exactly one '1'
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)

    return np.all(row_sums == 1) and np.all(col_sums == 1)

def generate_orthogonal_matrix(n):
    """
    Generate an nxn orthogonal matrix and check if it is a permutation matrix.

    Input:
    n = Dimension of the matrix.

    Output:
    orthogonal_matrix = n x n orthogonal matrix.
    """
    random_matrix = np.random.uniform(-1, 1, (n, n))
    orthogonal_matrix = gram_schmidt_process(random_matrix)

    # Check if the generated orthogonal matrix is a permutation matrix
    if is_permutation_matrix(orthogonal_matrix):
        print("The generated orthogonal matrix is a permutation matrix.")
        print("Orthogonal Matrix = ")
        print(np.array_str(orthogonal_matrix, precision=2, suppress_small=True))  
    return orthogonal_matrix

def generate_permutation_matrix(n):
    """
    Generate an n x n permutation matrix.

    Input:
    n = Dimension of the matrix.

    Output:
    permutation_matrix = n x n permutation matrix.
    """
    permutation_matrix = np.eye(n)
    np.random.shuffle(permutation_matrix)
    #print("Permutation Matrix = ")
    #print(np.array_str(permutation_matrix, precision=2, suppress_small=True))
    return permutation_matrix

def is_diagonal(matrix, tol=1e-5):
    """
    Check if a matrix is diagonal

    Input:
    matrix = The matrix to check.
       tol = Tolerance for off-diagonal elements to be considered zero.

    Output:
    True if matrix is diagonal, False otherwise.
    """
    return np.all(np.abs(matrix - np.diag(np.diagonal(matrix))) < tol)

# =========================================================
#                        Simulation
# =========================================================
# (m, n) pairs to test
configurations = [(50, 500)]#(5, 10), (10, 20), (10, 50), (50, 100), (50, 500), (50, 1000), (100, 1000)]
number_of_trials = 100000

results = []

#  Run simulations for each (m, n)
for m, n in configurations:
    print(f"(m, n) = ({m}, {n})")
    Lambda1 = generate_Lambda1(m, n)
    Lambda2 = generate_Lambda2(n)

    # Counter
    diagonal_count_orthogonal = 0

    # Trials for orthogonal matrices
    for _ in range(number_of_trials):
        print(f'Number of trials = {_}')
        U = generate_orthogonal_matrix(n)
        Lambda3 = compute_Lambda3(U, Lambda1, Lambda2)
        if is_diagonal(Lambda3):
            diagonal_count_orthogonal += 1
            print(Lambda3) #if it finds a diagonal lambda, printing it to check if it is a permutation matrix

    results.append({
        'm': m,
        'n': n,
        'Number of Simulations': number_of_trials,
        'Diagonal Count (Orthogonal)': diagonal_count_orthogonal
    })
    print(f'Diagonal Count (Orthogonal) = {diagonal_count_orthogonal}')

# Convert results to a DataFrame and generate LaTeX table
results_df = pd.DataFrame(results)
latex_table = results_df.to_latex(index=False, caption="Simulation Results for Different (m, n) Configurations", label="tab:simulation_results")
print(latex_table)

# Directly to a .tex file
#with open("results_table.tex", "w") as file:
#    file.write(latex_table)
