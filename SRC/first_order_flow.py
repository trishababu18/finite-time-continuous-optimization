import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def cost_function(x, p):
    return (1/p) * np.abs(x)**p

def gradient(x, p):
    return np.abs(x)**(p-1) * np.sign(x)

# Analytical solution for scalar q-RGF
def analytical_solution_scalar(t, x0, c, p, q):
    
    if q <= p:
        # Asymptotic convergence case (no finite-time)
        # For q = p, solution is x(t) = x0 * exp(-c*t)
        if np.isclose(q, p):
            return x0 * np.exp(-c * t)
        else:
            # For q < p, ε < 0
            eps = (q - p) / (q - 1)
            return np.sign(x0) * (np.abs(x0)**eps - eps * c * t)**(1/eps)
    else:
        # Finite-time convergence case (q > p)
        eps = (q - p) / (q - 1)
        inner = np.abs(x0)**eps - eps * c * t
        if inner <= 0:
            return 0.0
        else:
            return np.sign(x0) * inner**(1/eps)

def compute_settling_time(x0, c, p, q):
    if q <= p:
        return np.inf
    eps = (q - p) / (q - 1)
    return (1 / (c * eps)) * np.abs(x0)**eps

# q-RGF ODE for numerical integration (scalar case)
def q_rgf_ode(t, x, c, p, q):
    if np.abs(x) < 1e-15:
        return 0.0
    eps = (q - p) / (q - 1)
    return -c * np.abs(x)**(-eps) * x

def plot_figure1():
    x0 = 3/4
    c = 2
    p = 3
    q_values = [3.3, 4, 10, 100]
    colors = ['blue', 'orange', 'green', 'red']
    
    t_end = 5
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, 1000)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes[0]
    for q, color in zip(q_values, colors):
        # Compute analytical solution
        x_analytical = np.array([analytical_solution_scalar(t, x0, c, p, q) for t in t_eval])
        ax1.plot(t_eval, x_analytical, color=color, linewidth=2, label=f'q = {q}')
        
        # Mark settling time if finite
        t_star = compute_settling_time(x0, c, p, q)
        if t_star < t_end:
            ax1.axvline(x=t_star, color=color, linestyle='--', alpha=0.5)
            eps = (q - p) / (q - 1)
            ax1.annotate(f'$\\frac{{1}}{{c\\varepsilon}}|x_0|^\\varepsilon_{{q={q}}} \\approx {t_star:.3f}$', 
                        xy=(t_star, 0.1), fontsize=8, color=color)
    ax1.set_xlabel('t', fontsize=12)
    ax1.set_ylabel('x(t)', fontsize=12)
    ax1.set_title(f'Figure 1 (linear): q-RGF solutions, x₀ = {x0}, c = {c}, p = {p}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, t_end])
    ax1.set_ylim([0, 0.8])
    
    ax2 = axes[1]
    for q, color in zip(q_values, colors):
        x_analytical = np.array([analytical_solution_scalar(t, x0, c, p, q) for t in t_eval])
        # Add small epsilon to avoid log(0)
        x_plot = np.maximum(np.abs(x_analytical), 1e-50)
        ax2.semilogy(t_eval, x_plot, color=color, linewidth=2, label=f'q = {q}')
    ax2.set_xlabel('t', fontsize=12)
    ax2.set_ylabel('x(t)', fontsize=12)
    ax2.set_title(f'Figure 1 (log): q-RGF solutions, x₀ = {x0}, c = {c}, p = {p}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, t_end])
    ax2.set_ylim([1e-40, 1])
    
    plt.tight_layout()
    plt.savefig('figure1_q_rgf_varying_q.png', dpi=150, bbox_inches='tight')
    
    print("\n--- Figure 1: Settling Times ---")
    print(f"Parameters: x0 = {x0}, c = {c}, p = {p}")
    for q in q_values:
        t_star = compute_settling_time(x0, c, p, q)
        eps = (q - p) / (q - 1)
        print(f"q = {q}: ε = {eps:.4f}, t* = {t_star:.4f}")
    return fig

def plot_figure2():
    c = 2
    p = 3
    q = 10
    x0_values = [0.75, 0.2, -0.5]
    colors = ['blue', 'orange', 'green']
    
    eps = (q - p) / (q - 1)
    t_end = 0.7
    t_eval = np.linspace(0, t_end, 1000)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for x0, color in zip(x0_values, colors):
        # Compute analytical solution
        x_analytical = np.array([analytical_solution_scalar(t, x0, c, p, q) for t in t_eval])
        ax.plot(t_eval, x_analytical, color=color, linewidth=2, label=f'x₀ = {x0}')
        
        # Mark settling time
        t_star = compute_settling_time(x0, c, p, q)
        ax.axvline(x=t_star, color=color, linestyle='--', alpha=0.5)
    
    ax.set_xlabel('t', fontsize=12)
    ax.set_ylabel('x(t)', fontsize=12)
    ax.set_title(f'Figure 2: q-RGF solutions, c = {c}, q = {q}, p = {p}, ε = {eps:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, t_end])
    
    for x0, color in zip(x0_values, colors):
        t_star = compute_settling_time(x0, c, p, q)
        ax.annotate(f'$\\frac{{1}}{{c\\varepsilon}}|x_0|^\\varepsilon = {t_star:.3f}$', 
                   xy=(t_star + 0.02, 0.7), fontsize=9, color=color)
    
    plt.tight_layout()
    plt.savefig('figure2_q_rgf_varying_x0.png', dpi=150, bbox_inches='tight')

    
    print("\n--- Figure 2: Settling Times ---")
    print(f"Parameters: c = {c}, q = {q}, p = {p}, ε = (q-p)/(q-1) = {eps:.4f}")
    for x0 in x0_values:
        t_star = compute_settling_time(x0, c, p, q)
        print(f"x0 = {x0}: t* = {t_star:.4f}")
    
    return fig

def verify_theorem1():
    
    print("\n" + "="*60)
    print("Verification of Theorem 1 (Finite-Time Convergence)")
    print("="*60)
    
    x0 = 3/4
    c = 2
    p = 3
    
    print(f"\nCost function: f(x) = (1/{p})|x|^{p}")
    print(f"Initial condition: x0 = {x0}")
    print(f"Coefficient: c = {c}")
    print(f"Gradient dominance order: p = {p}")
    
    print("\n" + "-"*40)
    print("Testing q-RGF for various q > p:")
    print("-"*40)
    
    for q in [3.5, 4, 5, 10, 100]:
        eps = (q - p) / (q - 1)
        t_star_bound = (1 / (c * eps)) * np.abs(x0)**eps
        
        # Verify by numerical integration
        t_span = (0, t_star_bound * 1.5)
        sol = solve_ivp(lambda t, x: q_rgf_ode(t, x[0], c, p, q), 
                       t_span, [x0], dense_output=True, max_step=0.001)
        
        # Find when |x(t)| < tolerance
        tol = 1e-10
        t_converged = None
        for t in np.linspace(0, t_star_bound * 1.5, 10000):
            if np.abs(sol.sol(t)[0]) < tol:
                t_converged = t
                break
        
        print(f"q = {q:5.1f}: ε = {eps:.4f}, Upper bound t* = {t_star_bound:.6f}, "
              f"Actual (numerical) ≈ {t_converged:.6f}" if t_converged else 
              f"q = {q:5.1f}: ε = {eps:.4f}, Upper bound t* = {t_star_bound:.6f}")

if __name__ == "__main__":
    print("="*60)
    print("Replicating Section 5.1: First-Order Flow (q-RGF)")
    print("From: 'Finite-Time Convergence in Continuous-Time Optimization'")
    print("      Romero & Benosman, ICML 2020")
    print("="*60)
    
    # Create output directory
    import os
    os.makedirs('figures', exist_ok=True)
    os.chdir('figures')

    plot_figure1()
    plot_figure2()
    
    verify_theorem1()
    plt.show()
