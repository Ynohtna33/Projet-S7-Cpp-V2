#include "TimeScheme.h"
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace std;
using namespace Eigen;

// ============================================================================
// EULER EXPLICITE
// ============================================================================
// Schéma explicite: V^{n+1} = V^n + dτ * F(V^n, τ^n)
// Simple mais nécessite une condition CFL pour la stabilité
VectorXd EulerExplicit::step(ODESystem* system, 
                              const VectorXd& V,
                              double tau, 
                              double dtau) {
    int n = V.size();
    VectorXd V_new(n);
    
    // Évalue F(V^n, τ^n)
    VectorXd F = system->evaluate(V, tau);
    
    // Mise à jour explicite: V^{n+1} = V^n + dτ * F
    V_new = V + dtau * F;
    
    // Applique les conditions aux limites au nouveau temps
    BlackScholesODE* bs_system = dynamic_cast<BlackScholesODE*>(system);
    if (bs_system) {
        bs_system->applyBoundaryConditions(V_new, tau + dtau);
    }
    
    return V_new;
}

// ============================================================================
// EULER IMPLICITE
// ============================================================================
// Constructeur: stocke les paramètres nécessaires pour construire la matrice
EulerImplicit::EulerImplicit(double r, double sigma, const VectorXd& S_grid)
    : r_(r), sigma_(sigma), S_grid_(S_grid) {
    
    if (S_grid_.size() > 1) {
        dS_ = S_grid_(1) - S_grid_(0);
    } else {
        dS_ = 1.0;
    }
}

// Algorithme de Thomas pour résoudre un système tridiagonal
// a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
VectorXd EulerImplicit::solveTridiagonal(const VectorXd& a,
                                         const VectorXd& b,
                                         const VectorXd& c,
                                         const VectorXd& d) const {
    int n = b.size();
    VectorXd c_star(n);
    VectorXd d_star(n);
    VectorXd x(n);
    
    // Forward sweep: élimine la sous-diagonale
    c_star(0) = c(0) / b(0);
    d_star(0) = d(0) / b(0);
    
    for (int i = 1; i < n; ++i) {
        double m = b(i) - a(i) * c_star(i - 1);
        c_star(i) = c(i) / m;
        d_star(i) = (d(i) - a(i) * d_star(i - 1)) / m;
    }
    
    // Backward substitution: résout pour x
    x(n - 1) = d_star(n - 1);
    for (int i = n - 2; i >= 0; --i) {
        x(i) = d_star(i) - c_star(i) * x(i + 1);
    }
    
    return x;
}

// Schéma implicite: (I - dτ*J)*V^{n+1} = V^n
// où J est le Jacobien de F (la matrice de discrétisation)
VectorXd EulerImplicit::step(ODESystem* system, 
                              const VectorXd& V,
                              double tau, 
                              double dtau) {
    int n = V.size();
    
    // Construction de la matrice tridiagonale (I - dτ*J)
    // J_{i,j} représente la discrétisation de l'opérateur de Black-Scholes
    VectorXd a = VectorXd::Zero(n);  // Sous-diagonale
    VectorXd b = VectorXd::Zero(n);  // Diagonale
    VectorXd c = VectorXd::Zero(n);  // Sur-diagonale
    VectorXd rhs = V;                // Second membre = V^n
    
    // Conditions aux limites (bords fixés)
    b(0) = 1.0;
    c(0) = 0.0;
    b(n - 1) = 1.0;
    a(n - 1) = 0.0;
    
    BlackScholesODE* bs_system = dynamic_cast<BlackScholesODE*>(system);
    if (bs_system) {
        rhs(0) = 0.0; // V(0, τ) = 0
        rhs(n - 1) = S_grid_(n - 1) - bs_system->getGrid()(0) * exp(-r_ * (tau + dtau));
    }
    
    // Points intérieurs: discrétise l'opérateur Black-Scholes
    for (int i = 1; i < n - 1; ++i) {
        double S = S_grid_(i);
        
        // Coefficients de l'opérateur discrétisé
        // ∂²V/∂S²: (V_{i-1} - 2V_i + V_{i+1}) / dS²
        double alpha = 0.5 * sigma_ * sigma_ * S * S / (dS_ * dS_);
        
        // ∂V/∂S: (V_{i+1} - V_{i-1}) / (2*dS)
        double beta = 0.5 * r_ * S / dS_;
        
        // Coefficient de V_i
        double gamma = r_;
        
        // Construction de (I - dτ*J)
        // J*V = alpha*(V_{i-1} - 2V_i + V_{i+1}) + beta*(V_{i+1} - V_{i-1}) - gamma*V_i
        a(i) = -dtau * (alpha - beta);           // Coefficient de V_{i-1}
        b(i) = 1.0 + dtau * (2.0 * alpha + gamma); // Coefficient de V_i
        c(i) = -dtau * (alpha + beta);           // Coefficient de V_{i+1}
    }
    
    // Résout le système tridiagonal
    VectorXd V_new = solveTridiagonal(a, b, c, rhs);
    
    // Applique les conditions aux limites
    if (bs_system) {
        bs_system->applyBoundaryConditions(V_new, tau + dtau);
    }
    
    return V_new;
}

// ============================================================================
// RUNGE-KUTTA 2
// ============================================================================
// Schéma RK2 (méthode du point milieu):
// k1 = F(V^n, τ^n)
// k2 = F(V^n + dτ/2 * k1, τ^n + dτ/2)
// V^{n+1} = V^n + dτ * k2
VectorXd RungeKutta2::step(ODESystem* system, 
                           const VectorXd& V,
                           double tau, 
                           double dtau) {
    int n = V.size();
    
    // Étape 1: calcule k1 = F(V^n)
    VectorXd k1 = system->evaluate(V, tau);
    
    // Étape 2: calcule V_temp = V^n + dτ/2 * k1
    VectorXd V_temp = V + 0.5 * dtau * k1;
    
    // Applique les conditions aux limites à V_temp
    BlackScholesODE* bs_system = dynamic_cast<BlackScholesODE*>(system);
    if (bs_system) {
        bs_system->applyBoundaryConditions(V_temp, tau + 0.5 * dtau);
    }
    
    // Étape 3: calcule k2 = F(V_temp)
    VectorXd k2 = system->evaluate(V_temp, tau + 0.5 * dtau);
    
    // Étape 4: mise à jour finale V^{n+1} = V^n + dτ * k2
    VectorXd V_new = V + dtau * k2;
    
    // Applique les conditions aux limites
    if (bs_system) {
        bs_system->applyBoundaryConditions(V_new, tau + dtau);
    }
    
    return V_new;
}

// ============================================================================
// RUNGE-KUTTA 3
// ============================================================================
// Schéma RK3:
// k1 = F(V^n)
// k2 = F(V^n + dτ/2 * k1)
// k3 = F(V^n - dτ*k1 + 2*dτ*k2)
// V^{n+1} = V^n + dτ/6 * (k1 + 4*k2 + k3)
VectorXd RungeKutta3::step(ODESystem* system, 
                           const VectorXd& V,
                           double tau, 
                           double dtau) {
    int n = V.size();
    BlackScholesODE* bs_system = dynamic_cast<BlackScholesODE*>(system);
    
    // k1 = F(V^n)
    VectorXd k1 = system->evaluate(V, tau);
    
    // V_temp1 = V^n + dτ/2 * k1
    VectorXd V_temp1 = V + 0.5 * dtau * k1;
    if (bs_system) bs_system->applyBoundaryConditions(V_temp1, tau + 0.5 * dtau);
    
    // k2 = F(V_temp1)
    VectorXd k2 = system->evaluate(V_temp1, tau + 0.5 * dtau);
    
    // V_temp2 = V^n - dτ*k1 + 2*dτ*k2
    VectorXd V_temp2 = V - dtau * k1 + 2.0 * dtau * k2;
    if (bs_system) bs_system->applyBoundaryConditions(V_temp2, tau + dtau);
    
    // k3 = F(V_temp2)
    VectorXd k3 = system->evaluate(V_temp2, tau + dtau);
    
    // V^{n+1} = V^n + dτ/6 * (k1 + 4*k2 + k3)
    VectorXd V_new = V + dtau / 6.0 * (k1 + 4.0 * k2 + k3);
    
    if (bs_system) bs_system->applyBoundaryConditions(V_new, tau + dtau);
    
    return V_new;
}

// ============================================================================
// RUNGE-KUTTA 4
// ============================================================================
// Schéma RK4 classique:
// k1 = F(V^n)
// k2 = F(V^n + dτ/2 * k1)
// k3 = F(V^n + dτ/2 * k2)
// k4 = F(V^n + dτ * k3)
// V^{n+1} = V^n + dτ/6 * (k1 + 2*k2 + 2*k3 + k4)
VectorXd RungeKutta4::step(ODESystem* system, 
                           const VectorXd& V,
                           double tau, 
                           double dtau) {
    int n = V.size();
    BlackScholesODE* bs_system = dynamic_cast<BlackScholesODE*>(system);
    
    // Étape 1: k1 = F(V^n)
    VectorXd k1 = system->evaluate(V, tau);
    
    // Étape 2: k2 = F(V^n + dτ/2 * k1)
    VectorXd V_temp = V + 0.5 * dtau * k1;
    if (bs_system) bs_system->applyBoundaryConditions(V_temp, tau + 0.5 * dtau);
    VectorXd k2 = system->evaluate(V_temp, tau + 0.5 * dtau);
    
    // Étape 3: k3 = F(V^n + dτ/2 * k2)
    V_temp = V + 0.5 * dtau * k2;
    if (bs_system) bs_system->applyBoundaryConditions(V_temp, tau + 0.5 * dtau);
    VectorXd k3 = system->evaluate(V_temp, tau + 0.5 * dtau);
    
    // Étape 4: k4 = F(V^n + dτ * k3)
    V_temp = V + dtau * k3;
    if (bs_system) bs_system->applyBoundaryConditions(V_temp, tau + dtau);
    VectorXd k4 = system->evaluate(V_temp, tau + dtau);
    
    // Étape 5: Combinaison finale
    // V^{n+1} = V^n + dτ/6 * (k1 + 2*k2 + 2*k3 + k4)
    VectorXd V_new = V + dtau / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    
    // Applique les conditions aux limites
    if (bs_system) {
        bs_system->applyBoundaryConditions(V_new, tau + dtau);
    }
    
    return V_new;
}

// ============================================================================
// FONCTION UTILITAIRE POUR CRÉER LE SCHÉMA TEMPOREL
// ============================================================================
// Fonction pour créer le schéma temporel selon le nom
unique_ptr<TimeScheme> createTimeScheme(const string& name,
                                        double r, double sigma,
                                        const VectorXd& S_grid) {
    if (name == "euler_explicit") {
        return make_unique<EulerExplicit>();
    } else if (name == "euler_implicit") {
        return make_unique<EulerImplicit>(r, sigma, S_grid);
    } else if (name == "rk2") {
        return make_unique<RungeKutta2>();
    } else if (name == "rk3") {
        return make_unique<RungeKutta3>();
    } else if (name == "rk4") {
        return make_unique<RungeKutta4>();
    } else {
        cerr << "Schéma temporel inconnu: " << name << endl;
        cerr << "Utilisation du schéma Euler implicite par défaut" << endl;
        return make_unique<EulerImplicit>(r, sigma, S_grid);
    }
}