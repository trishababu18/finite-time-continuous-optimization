# Finite-Time Convergence in Continuous-Time Optimization
A Replication Study (Romero & Benosman, ICML 2020)

This repository contains a replication and validation of the finite-time convergent optimization flows proposed by Romero & Benosman (ICML 2020). The project was completed as part of the *Optimization for Engineers* course at Utah State University.

## Overview
Classical gradient flows converge asymptotically to an optimum. In contrast, the methods studied here achieve **exact convergence in finite, bounded time**. This project implements and validates both first-order and second-order continuous-time optimization flows, reproducing all experimental figures from the original paper.

## Algorithms Implemented
### First-Order Method
- **q-Rescaled Gradient Flow (q-RGF)**
- Finite-time convergence for gradient-dominated functions when q > p
- Verified theoretical settling-time bounds

### Second-Order Method
- **Rescaled Newton Flow (RNF)**
- Prescribed convergence time for strongly convex functions
- Uses Hessian information with time-rescaling

## Discretization Schemes
- Nesterov-style momentum discretization
- Accelerated backtracking line search
- Comparison against Gradient Descent, Nesterov Accelerated GD, and Newton’s method

## Implementation Details
- Language: Python
- Numerical integration: SciPy ODE solvers (RK45)
- Libraries: NumPy, SciPy, Matplotlib
- Test functions:
  - p-norm function (gradient-dominated)
  - Rosenbrock function (strongly convex)
  - Log-sum-exp function (high-dimensional convex)

## Results
- Successfully reproduced all five figures from Section 5 of the original paper
- Verified finite-time convergence bounds for q-RGF
- Demonstrated prescribed-time convergence behavior for RNF
- Showed that Nesterov-style discretization provides strong practical performance

## Repository Structure
- `SRC/` – continuous-time optimization flow implementations
- `figures/` – generated plots (Figures 1–5)
- `docs/` – final project report

## Reference
Romero, O., & Benosman, M. (2020). *Finite-Time Convergence in Continuous-Time Optimization*. ICML.
