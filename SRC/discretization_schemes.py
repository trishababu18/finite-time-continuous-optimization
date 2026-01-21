import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class LogSumExp:
    def __init__(self, n=20, m=50, rho=5, seed=42):
        np.random.seed(seed)
        self.n = n
        self.m = m
        self.rho = rho
        self.A = np.random.randn(m, n)  
        self.b = np.random.randn(m)
        
        self._compute_optimal()
    
    def _compute_optimal(self):
        
        x0 = np.zeros(self.n)
        result = minimize(self.f, x0, method='BFGS', jac=self.grad, 
                         options={'gtol': 1e-12, 'maxiter': 10000})
        self.x_star = result.x
        self.f_star = result.fun
    
    def f(self, x):
        """Evaluate f(x) = ρ * log(Σᵢ exp((aᵢᵀx - bᵢ)/ρ))"""
        z = (self.A @ x - self.b) / self.rho
        z_max = np.max(z)
        return self.rho * (z_max + np.log(np.sum(np.exp(z - z_max))))
    
    def grad(self, x):
        z = (self.A @ x - self.b) / self.rho
        # Softmax weights
        z_max = np.max(z)
        exp_z = np.exp(z - z_max)
        weights = exp_z / np.sum(exp_z)
        # ∇f = Σᵢ wᵢ aᵢ = Aᵀ w
        return self.A.T @ weights
    
    def hessian(self, x):
        z = (self.A @ x - self.b) / self.rho
        z_max = np.max(z)
        exp_z = np.exp(z - z_max)
        weights = exp_z / np.sum(exp_z)
        
        Aw = self.A.T @ weights  # n x 1
        weighted_A = self.A.T * weights  # n x m, each column scaled by wᵢ
        H = (1/self.rho) * (weighted_A @ self.A - np.outer(Aw, Aw))
        return H

# First-Order Discrete Algorithms

def gradient_descent(func, x0, eta, max_iter):
    """Standard Gradient Descent: xₖ₊₁ = xₖ - η∇f(xₖ)"""
    x = x0.copy()
    history = [func.f(x) - func.f_star]
    
    for k in range(max_iter):
        grad = func.grad(x)
        x = x - eta * grad
        history.append(func.f(x) - func.f_star)
    
    return np.array(history)

def q_rgf_forward_euler(func, x0, eta, q, max_iter):
    """
    Forward Euler discretization of q-RGF:
    xₖ₊₁ = xₖ - η * ∇f(xₖ) / ||∇f(xₖ)||^((q-2)/(q-1))
    """
    x = x0.copy()
    history = [func.f(x) - func.f_star]
    
    for k in range(max_iter):
        grad = func.grad(x)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        
        exponent = (q - 2) / (q - 1)
        F = -grad / (grad_norm ** exponent)
        x = x + eta * F
        history.append(func.f(x) - func.f_star)
    
    return np.array(history)

def nesterov_agd(func, x0, eta, max_iter, beta_type='standard'):
    """
    Nesterov's Accelerated Gradient Descent:
    yₖ = xₖ + βₖ(xₖ - xₖ₋₁)
    xₖ₊₁ = yₖ - η∇f(yₖ)
    """
    x = x0.copy()
    x_prev = x0.copy()
    history = [func.f(x) - func.f_star]
    
    for k in range(max_iter):
        if beta_type == 'standard':
            beta = (k - 1) / (k + 2) if k > 0 else 0
        else:
            beta = 0.9  # Fixed momentum
        
        y = x + beta * (x - x_prev)
        grad = func.grad(y)
        x_prev = x.copy()
        x = y - eta * grad
        history.append(func.f(x) - func.f_star)
    
    return np.array(history)

def q_rgf_nesterov(func, x0, eta, q, max_iter, beta_type='standard'):
    """
    Proposed Nesterov-style discretization of q-RGF:
    yₖ = xₖ + βₖ(xₖ - xₖ₋₁)
    xₖ₊₁ = yₖ + η F(yₖ)  where F is the q-RGF direction
    """
    x = x0.copy()
    x_prev = x0.copy()
    history = [func.f(x) - func.f_star]
    
    for k in range(max_iter):
        if beta_type == 'standard':
            beta = (k - 1) / (k + 2) if k > 0 else 0
        else:
            beta = 0.9
        
        y = x + beta * (x - x_prev)
        grad = func.grad(y)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-15:
            break
        
        exponent = (q - 2) / (q - 1)
        F = -grad / (grad_norm ** exponent)
        
        x_prev = x.copy()
        x = y + eta * F
        history.append(func.f(x) - func.f_star)
    
    return np.array(history)

# Second-Order Discrete Algorithms

def newton_method(func, x0, max_iter, damping=1.0):
    """Standard Newton's Method: xₖ₊₁ = xₖ - [∇²f(xₖ)]⁻¹ ∇f(xₖ)"""
    x = x0.copy()
    history = [func.f(x) - func.f_star]
    
    for k in range(max_iter):
        grad = func.grad(x)
        H = func.hessian(x)
        
        try:
            # Add regularization for stability
            H_reg = H + 1e-8 * np.eye(len(x))
            direction = np.linalg.solve(H_reg, grad)
        except:
            direction = grad
        
        x = x - damping * direction
        history.append(func.f(x) - func.f_star)
    
    return np.array(history)

def rnf_forward_euler(func, x0, eta, max_iter):
    """
    Forward Euler discretization of RNF (Rescaled Newton Flow):
    xₖ₊₁ = xₖ - η * ||∇f(x0)|| * [∇²f(xₖ)]⁻¹ ∇f(xₖ) / ||∇f(xₖ)||
    """
    x = x0.copy()
    grad_norm_x0 = np.linalg.norm(func.grad(x0))
    history = [func.f(x) - func.f_star]
    
    for k in range(max_iter):
        grad = func.grad(x)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-15:
            break
        
        H = func.hessian(x)
        try:
            H_reg = H + 1e-8 * np.eye(len(x))
            H_inv_grad = np.linalg.solve(H_reg, grad)
        except:
            H_inv_grad = grad
        
        F = -grad_norm_x0 * H_inv_grad / grad_norm
        x = x + eta * F
        history.append(func.f(x) - func.f_star)
    
    return np.array(history)

def nf_nesterov(func, x0, eta, max_iter, beta_type='standard'):
    """
    Nesterov-style discretization of Newton Flow:
    yₖ = xₖ + βₖ(xₖ - xₖ₋₁)
    xₖ₊₁ = yₖ - η [∇²f(yₖ)]⁻¹ ∇f(yₖ)
    """
    x = x0.copy()
    x_prev = x0.copy()
    history = [func.f(x) - func.f_star]
    
    for k in range(max_iter):
        if beta_type == 'standard':
            beta = (k - 1) / (k + 2) if k > 0 else 0
        else:
            beta = 0.9
        
        y = x + beta * (x - x_prev)
        grad = func.grad(y)
        H = func.hessian(y)
        
        try:
            H_reg = H + 1e-8 * np.eye(len(y))
            direction = np.linalg.solve(H_reg, grad)
        except:
            direction = grad
        
        x_prev = x.copy()
        x = y - eta * direction
        history.append(func.f(x) - func.f_star)
    
    return np.array(history)

def rnf_nesterov(func, x0, eta, max_iter, beta_type='standard'):
    """
    Nesterov-style discretization of RNF (Rescaled Newton Flow):
    yₖ = xₖ + βₖ(xₖ - xₖ₋₁)
    xₖ₊₁ = yₖ + η F(yₖ)  where F is the RNF direction
    """
    x = x0.copy()
    x_prev = x0.copy()
    grad_norm_x0 = np.linalg.norm(func.grad(x0))
    history = [func.f(x) - func.f_star]
    
    for k in range(max_iter):
        if beta_type == 'standard':
            beta = (k - 1) / (k + 2) if k > 0 else 0
        else:
            beta = 0.9
        
        y = x + beta * (x - x_prev)
        grad = func.grad(y)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-15:
            break
        
        H = func.hessian(y)
        try:
            H_reg = H + 1e-8 * np.eye(len(y))
            H_inv_grad = np.linalg.solve(H_reg, grad)
        except:
            H_inv_grad = grad
        
        F = -grad_norm_x0 * H_inv_grad / grad_norm
        
        x_prev = x.copy()
        x = y + eta * F
        history.append(func.f(x) - func.f_star)
    
    return np.array(history)

def accelerated_backtracking(func, x0, max_iter, flow_type='newton', q=10, d=0.5, u=1.5):
    """
    Accelerated backtracking line search from Almeida et al. (1997)
    
    ηₖ = u * ηₖ₋₁ if f(yₖ + u*ηₖ₋₁*F(yₖ)) ≤ min{f(xₖ), f(yₖ + ηₖ₋₁*F(yₖ))}
    Otherwise, ηₖ = d^rₖ * ηₖ₋₁ where rₖ is smallest r such that 
    f(yₖ + d^r * ηₖ₋₁ * F(yₖ)) ≤ f(xₖ)
    """
    x = x0.copy()
    x_prev = x0.copy()
    eta = 1.0
    grad_norm_x0 = np.linalg.norm(func.grad(x0))
    history = [func.f(x) - func.f_star]
    
    for k in range(max_iter):
        beta = (k - 1) / (k + 2) if k > 0 else 0
        y = x + beta * (x - x_prev)
        
        grad = func.grad(y)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-15:
            break
        
        # Compute flow direction F
        if flow_type == 'newton':
            H = func.hessian(y)
            try:
                H_reg = H + 1e-8 * np.eye(len(y))
                F = -np.linalg.solve(H_reg, grad)
            except:
                F = -grad
        elif flow_type == 'rnf':
            H = func.hessian(y)
            try:
                H_reg = H + 1e-8 * np.eye(len(y))
                H_inv_grad = np.linalg.solve(H_reg, grad)
            except:
                H_inv_grad = grad
            F = -grad_norm_x0 * H_inv_grad / grad_norm
        else:  # gradient
            F = -grad
        
        f_x = func.f(x)
        f_y_base = func.f(y + eta * F)
        f_y_up = func.f(y + u * eta * F)
        
        # Try to increase step size
        if f_y_up <= min(f_x, f_y_base):
            eta = u * eta
            x_new = y + eta * F
        else:
            # Backtrack
            eta_try = eta
            for r in range(50):
                x_try = y + eta_try * F
                if func.f(x_try) <= f_x:
                    break
                eta_try = d * eta_try
            eta = eta_try
            x_new = y + eta * F
        
        x_prev = x.copy()
        x = x_new
        history.append(func.f(x) - func.f_star)
    
    return np.array(history)

def plot_figure4(func, n_trials=50, max_iter=500):

    np.random.seed(42)
    
    # Run multiple trials with different initial conditions
    histories = {
        'GD': [],
        'RGF (FE)': [],
        'NAGD': [],
        'RGF (prop)': []
    }
    
    # Tuned hyperparameters 
    eta_gd = 0.1
    eta_rgf = 0.05
    eta_nagd = 0.1
    eta_rgf_prop = 0.03
    q = 4.0
    
    for trial in range(n_trials):
        x0 = np.random.randn(func.n)
        
        histories['GD'].append(gradient_descent(func, x0, eta_gd, max_iter))
        histories['RGF (FE)'].append(q_rgf_forward_euler(func, x0, eta_rgf, q, max_iter))
        histories['NAGD'].append(nesterov_agd(func, x0, eta_nagd, max_iter))
        histories['RGF (prop)'].append(q_rgf_nesterov(func, x0, eta_rgf_prop, q, max_iter))
    
    # Compute averages
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['blue', 'orange', 'green', 'red']
    labels = ['GD (fixed stepsizes)', 
              'RGF (forward-Euler discretization w/ fixed stepsizes)',
              'NAGD (fixed stepsizes)', 
              'RGF (proposed discretization w/ fixed stepsizes)']
    
    for (name, hist_list), color, label in zip(histories.items(), colors, labels):
        max_len = max(len(h) for h in hist_list)
        padded = []
        for h in hist_list:
            if len(h) < max_len:
                h = np.concatenate([h, np.full(max_len - len(h), h[-1])])
            padded.append(h)
        
        avg_hist = np.mean(padded, axis=0)
        ax.semilogy(range(len(avg_hist)), avg_hist, color=color, linewidth=2, label=label)
    
    ax.set_xlabel('k (iteration)', fontsize=12)
    ax.set_ylabel('f(xₖ) - f*', fontsize=12)
    ax.set_title('Figure 4: First-Order Discrete Algorithms (log-sum-exp)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max_iter])
    ax.set_ylim([1e-15, 1e5])
    
    plt.tight_layout()
    plt.savefig('figure4_first_order_discrete.png', dpi=150, bbox_inches='tight')
    
    return fig

def plot_figure5(func, n_trials=50, max_iter=500):
   
    np.random.seed(42)
    
    histories = {
        'RNF (FE)': [],
        'Newton': [],
        'NF (prop)': [],
        'NF (prop+bt)': [],
        'RNF (prop)': [],
        'Newton (bt)': [],
        'RNF (prop+bt)': []
    }
    
    # Tuned hyperparameters
    eta_rnf = 0.05
    eta_newton = 1.0
    eta_nf_prop = 0.5
    eta_rnf_prop = 0.02
    
    for trial in range(n_trials):
        x0 = np.random.randn(func.n)
        
        histories['RNF (FE)'].append(rnf_forward_euler(func, x0, eta_rnf, max_iter))
        histories['Newton'].append(newton_method(func, x0, max_iter))
        histories['NF (prop)'].append(nf_nesterov(func, x0, eta_nf_prop, max_iter))
        histories['NF (prop+bt)'].append(accelerated_backtracking(func, x0, max_iter, 'newton'))
        histories['RNF (prop)'].append(rnf_nesterov(func, x0, eta_rnf_prop, max_iter))
        histories['Newton (bt)'].append(accelerated_backtracking(func, x0, max_iter, 'newton'))
        histories['RNF (prop+bt)'].append(accelerated_backtracking(func, x0, max_iter, 'rnf'))
    
    # Compute averages
    fig, ax = plt.subplots(figsize=(10, 6))
    
    styles = [
        ('RNF (FE)', 'blue', '-'),
        ('Newton', 'orange', '-'),
        ('NF (prop)', 'green', '-'),
        ('NF (prop+bt)', 'green', '--'),
        ('RNF (prop)', 'red', '-'),
        ('Newton (bt)', 'orange', '--'),
        ('RNF (prop+bt)', 'red', '--')
    ]
    
    labels_full = [
        'RNF (forward-Euler w/ fixed stepsizes)',
        'Newton',
        'NF (proposed discretization w/ fixed stepsizes)',
        'NF (proposed discretization w/ accelerated backtracking)',
        'RNF (proposed discretization w/ fixed stepsizes)',
        'Newton (accelerated backtracking)',
        'RNF (proposed discretization w/ accelerated backtracking)'
    ]
    
    for (name, color, style), label in zip(styles, labels_full):
        hist_list = histories[name]
        max_len = max(len(h) for h in hist_list)
        padded = []
        for h in hist_list:
            if len(h) < max_len:
                h = np.concatenate([h, np.full(max_len - len(h), h[-1])])
            padded.append(h)
        
        avg_hist = np.mean(padded, axis=0)
        # Add small epsilon to avoid log(0)
        avg_hist = np.maximum(avg_hist, 1e-16)
        ax.semilogy(range(len(avg_hist)), avg_hist, color=color, linestyle=style, 
                   linewidth=2, label=label)
    
    ax.set_xlabel('k (iteration)', fontsize=12)
    ax.set_ylabel('f(xₖ) - f* (+ 1e⁻¹³)', fontsize=12)
    ax.set_title('Figure 5: Second-Order Discrete Algorithms (log-sum-exp)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max_iter])
    ax.set_ylim([1e-15, 1e5])
    
    plt.tight_layout()
    plt.savefig('figure5_second_order_discrete.png', dpi=150, bbox_inches='tight')
    
    return fig

if __name__ == "__main__":
    print("="*60)
    print("Replicating Section 5.3: Discretization Schemes")
    print("From: 'Finite-Time Convergence in Continuous-Time Optimization'")
    print("      Romero & Benosman, ICML 2020")
    print("="*60)
    
    # Create log-sum-exp function
    print("\nInitializing log-sum-exp function...")
    print("Parameters: n = 20, m = 50, ρ = 5")
    func = LogSumExp(n=20, m=50, rho=5, seed=42)
    print(f"Optimal value f* = {func.f_star:.6f}")
    
    # Create output directory
    import os
    os.makedirs('figures', exist_ok=True)
    os.chdir('figures')
    
    # Generate figures
    print("\n--- Generating Figure 4 (First-Order Algorithms) ---")
    plot_figure4(func, n_trials=50, max_iter=500)
    
    print("\n--- Generating Figure 5 (Second-Order Algorithms) ---")
    plot_figure5(func, n_trials=50, max_iter=500)
    
    plt.show()
