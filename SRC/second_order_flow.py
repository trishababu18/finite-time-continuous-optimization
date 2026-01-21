import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import fractional_matrix_power
import warnings
warnings.filterwarnings('ignore')

class RosenbrockFunction:
    def __init__(self, a=3, b=100):
        self.a = a
        self.b = b
        self.x_star = np.array([a, a**2])
    
    def f(self, x):
        x1, x2 = x[0], x[1]
        return (self.a - x1)**2 + self.b * (x2 - x1**2)**2
    
    def grad(self, x):
        x1, x2 = x[0], x[1]
        df_dx1 = -2*(self.a - x1) - 4*self.b*x1*(x2 - x1**2)
        df_dx2 = 2*self.b*(x2 - x1**2)
        return np.array([df_dx1, df_dx2])
    
    def hessian(self, x):
        x1, x2 = x[0], x[1]
        d2f_dx1dx1 = 2 - 4*self.b*(x2 - 3*x1**2)
        d2f_dx1dx2 = -4*self.b*x1
        d2f_dx2dx2 = 2*self.b
        return np.array([[d2f_dx1dx1, d2f_dx1dx2],
                        [d2f_dx1dx2, d2f_dx2dx2]])

def second_order_flow_eq27(t, x, rosenbrock, c, alpha, r):
    grad = rosenbrock.grad(x)
    grad_norm = np.linalg.norm(grad)
    
    if grad_norm < 1e-12:
        return np.zeros_like(x)
    
    H = rosenbrock.hessian(x)
    
    # Computing matrix powers using eigendecomposition for better stability
    try:
        # H^r
        eigvals, eigvecs = np.linalg.eigh(H)
        eigvals = np.maximum(eigvals, 1e-10)  
        H_r = eigvecs @ np.diag(eigvals**r) @ eigvecs.T
        
        # H^(r+1)
        H_r1 = eigvecs @ np.diag(eigvals**(r+1)) @ eigvecs.T
    except:
        return np.zeros_like(x)
    
    numerator = H_r @ grad
    denominator = grad.T @ H_r1 @ grad
    
    if np.abs(denominator) < 1e-12:
        return np.zeros_like(x)
    
    dx = -c * (grad_norm**(2*alpha)) * numerator / denominator
    return dx

def rescaled_newton_flow_eq30(t, x, rosenbrock, grad_norm_x0, T):
    grad = rosenbrock.grad(x)
    grad_norm = np.linalg.norm(grad)
    
    if grad_norm < 1e-12:
        return np.zeros_like(x)
    
    H = rosenbrock.hessian(x)
    
    try:
        H_inv = np.linalg.inv(H)
    except:
        # Use pseudo-inverse if singular
        H_inv = np.linalg.pinv(H)
    
    dx = -(grad_norm_x0 / T) * (H_inv @ grad) / grad_norm
    return dx

def plot_figure3():
    rosenbrock = RosenbrockFunction(a=3, b=100)
    T = 1.0  # Prescribed settling time
    
    initial_conditions = [
        np.array([0.0, 8.0]),   
        np.array([1.0, 8.5]),     
        np.array([3.5, 2.0]),   
        np.array([-2.0, 4.0])   
    ]
    
    colors = ['blue', 'orange', 'green', 'red']
    
    fig = plt.figure(figsize=(16, 12))
    
    # Main trajectory plot
    ax_main = fig.add_subplot(2, 2, 1)
    
    x1_range = np.linspace(-3, 5, 200)
    x2_range = np.linspace(-1, 11, 200)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = rosenbrock.f(np.array([X1[i, j], X2[i, j]]))
    
    # Log-scale contours
    levels = np.logspace(0, 4, 20)
    ax_main.contour(X1, X2, Z, levels=levels, colors='gray', alpha=0.5)
    
    # Plot trajectories
    trajectories = []
    
    for x0, color in zip(initial_conditions, colors):
        grad_norm_x0 = np.linalg.norm(rosenbrock.grad(x0))
        
        # Solve ODE
        t_span = (0, T * 1.5)
        t_eval = np.linspace(0, T * 1.5, 500)
        
        sol = solve_ivp(
            lambda t, x: rescaled_newton_flow_eq30(t, x, rosenbrock, grad_norm_x0, T),
            t_span, x0,
            t_eval=t_eval,
            method='RK45',
            max_step=0.01
        )
        
        trajectories.append(sol)
        
        # Plot trajectory
        ax_main.plot(sol.y[0], sol.y[1], color=color, linewidth=2, label=f'x₀ = ({x0[0]:.1f}, {x0[1]:.1f})')
        ax_main.scatter(x0[0], x0[1], color=color, s=100, marker='o', edgecolors='black', zorder=5)
        ax_main.scatter(sol.y[0, -1], sol.y[1, -1], color=color, s=100, marker='*', edgecolors='black', zorder=5)
    
    # Mark minimum
    ax_main.scatter(rosenbrock.x_star[0], rosenbrock.x_star[1], 
                   color='black', s=200, marker='*', label=f'x* = ({rosenbrock.a}, {rosenbrock.a**2})', zorder=10)
    
    ax_main.set_xlabel('$x_1$', fontsize=12)
    ax_main.set_ylabel('$x_2$', fontsize=12)
    ax_main.set_title('Figure 3: Trajectories on Rosenbrock Function\n(c, α, r) = (||∇f(x₀)||, 1/2, -1), T = 1')
    ax_main.legend(loc='upper left')
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlim([-3, 5])
    ax_main.set_ylim([-1, 11])
    
    # Subplot: ||x(t) - x*||
    ax_dist = fig.add_subplot(2, 2, 2)
    for sol, color, x0 in zip(trajectories, colors, initial_conditions):
        dist = np.sqrt((sol.y[0] - rosenbrock.x_star[0])**2 + 
                      (sol.y[1] - rosenbrock.x_star[1])**2)
        ax_dist.plot(sol.t, dist, color=color, linewidth=2, 
                    label=f'x₀ = ({x0[0]:.1f}, {x0[1]:.1f})')
    ax_dist.axvline(x=T, color='black', linestyle='--', alpha=0.5, label='T = 1')
    ax_dist.set_xlabel('t', fontsize=12)
    ax_dist.set_ylabel('||x(t) - x*||', fontsize=12)
    ax_dist.set_title('Distance to Optimum')
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.3)
    ax_dist.set_xlim([0, T * 1.5])
    
    # Subplot: ||∇f(x(t))||
    ax_grad = fig.add_subplot(2, 2, 3)
    for sol, color, x0 in zip(trajectories, colors, initial_conditions):
        grad_norms = np.array([np.linalg.norm(rosenbrock.grad(sol.y[:, i])) 
                               for i in range(sol.y.shape[1])])
        ax_grad.semilogy(sol.t, grad_norms + 1e-16, color=color, linewidth=2,
                        label=f'x₀ = ({x0[0]:.1f}, {x0[1]:.1f})')
    ax_grad.axvline(x=T, color='black', linestyle='--', alpha=0.5, label='T = 1')
    ax_grad.set_xlabel('t', fontsize=12)
    ax_grad.set_ylabel('||∇f(x(t))||', fontsize=12)
    ax_grad.set_title('Gradient Norm (log scale)')
    ax_grad.legend()
    ax_grad.grid(True, alpha=0.3)
    ax_grad.set_xlim([0, T * 1.5])
    
    # Subplot: f(x(t)) - f*
    ax_cost = fig.add_subplot(2, 2, 4)
    f_star = rosenbrock.f(rosenbrock.x_star)
    for sol, color, x0 in zip(trajectories, colors, initial_conditions):
        costs = np.array([rosenbrock.f(sol.y[:, i]) - f_star 
                         for i in range(sol.y.shape[1])])
        ax_cost.semilogy(sol.t, costs + 1e-16, color=color, linewidth=2,
                        label=f'x₀ = ({x0[0]:.1f}, {x0[1]:.1f})')
    ax_cost.axvline(x=T, color='black', linestyle='--', alpha=0.5, label='T = 1')
    ax_cost.set_xlabel('t', fontsize=12)
    ax_cost.set_ylabel('f(x(t)) - f*', fontsize=12)
    ax_cost.set_title('Cost Function Gap (log scale)')
    ax_cost.legend()
    ax_cost.grid(True, alpha=0.3)
    ax_cost.set_xlim([0, T * 1.5])
    
    plt.tight_layout()
    plt.savefig('figure3_second_order_rosenbrock.png', dpi=150, bbox_inches='tight')
    
    # Print convergence details
    print("\n--- Figure 3: Convergence Analysis ---")
    print(f"Rosenbrock parameters: a = {rosenbrock.a}, b = {rosenbrock.b}")
    print(f"Minimum: x* = ({rosenbrock.a}, {rosenbrock.a**2})")
    print(f"Prescribed settling time: T = {T}")
    print("\nFinal states at t = T:")
    for sol, x0 in zip(trajectories, initial_conditions):
        # Find state at t = T
        idx = np.argmin(np.abs(sol.t - T))
        x_T = sol.y[:, idx]
        grad_norm_T = np.linalg.norm(rosenbrock.grad(x_T))
        dist_T = np.linalg.norm(x_T - rosenbrock.x_star)
        print(f"  x₀ = ({x0[0]:5.1f}, {x0[1]:5.1f}) → x(T) = ({x_T[0]:8.5f}, {x_T[1]:8.5f}), "
              f"||x(T) - x*|| = {dist_T:.2e}, ||∇f|| = {grad_norm_T:.2e}")
    
    return fig

def verify_theorem2():
    print("\n" + "="*60)
    print("Verification of Theorem 2 (Second-Order Finite-Time Convergence)")
    print("="*60)
    
    rosenbrock = RosenbrockFunction(a=3, b=100)
    
    print(f"\nRosenbrock function: f(x1, x2) = ({rosenbrock.a} - x1)² + {rosenbrock.b}(x2 - x1²)²")
    print(f"Minimum at x* = ({rosenbrock.a}, {rosenbrock.a**2})")
    
    x0 = np.array([1.0, 8.0])
    grad_norm_x0 = np.linalg.norm(rosenbrock.grad(x0))
    
    print(f"\nInitial condition: x0 = ({x0[0]}, {x0[1]})")
    print(f"||∇f(x0)|| = {grad_norm_x0:.6f}")
    
    # Test for different prescribed settling times
    for T in [0.5, 1.0, 2.0]:
        print(f"\n--- Prescribed T = {T} ---")
        
        # The parameters are (c, α, r) = (||∇f(x0)||/T, 1/2, -1)
        c = grad_norm_x0 / T
        alpha = 0.5
        
        # Theoretical settling time from eq (29):
        # t* = ||∇f(x0)||^(2(1-α)) / (2c(1-α))
        #    = ||∇f(x0)||^1 / (2 * (||∇f(x0)||/T) * 0.5)
        #    = ||∇f(x0)|| * T / ||∇f(x0)||
        #    = T
        t_star_theory = (grad_norm_x0**(2*(1-alpha))) / (2*c*(1-alpha))
        
        print(f"  c = ||∇f(x0)||/T = {c:.6f}")
        print(f"  α = {alpha}")
        print(f"  Theoretical t* = {t_star_theory:.6f} (should equal T)")
        
        # Numerical verification
        t_span = (0, T * 1.5)
        sol = solve_ivp(
            lambda t, x: rescaled_newton_flow_eq30(t, x, rosenbrock, grad_norm_x0, T),
            t_span, x0,
            method='RK45',
            max_step=0.001
        )
        
        # Find convergence time (when gradient norm drops below threshold)
        tol = 1e-6
        t_converged = None
        for i in range(len(sol.t)):
            grad_norm = np.linalg.norm(rosenbrock.grad(sol.y[:, i]))
            if grad_norm < tol:
                t_converged = sol.t[i]
                break
        
        if t_converged:
            print(f"  Numerical convergence time ≈ {t_converged:.6f}")
        else:
            # Get gradient norm at t = T
            idx_T = np.argmin(np.abs(sol.t - T))
            grad_at_T = np.linalg.norm(rosenbrock.grad(sol.y[:, idx_T]))
            print(f"  ||∇f(x(T))|| at t = T: {grad_at_T:.2e}")

if __name__ == "__main__":
    print("="*60)
    print("Replicating Section 5.2: Second-Order Flow")
    print("From: 'Finite-Time Convergence in Continuous-Time Optimization'")
    print("      Romero & Benosman, ICML 2020")
    print("="*60)
    
    # Create output directory
    import os
    os.makedirs('figures', exist_ok=True)
    os.chdir('figures')
    
    # Generate figure
    plot_figure3()
    
    # Verify theoretical results
    verify_theorem2()
    
    plt.show()
