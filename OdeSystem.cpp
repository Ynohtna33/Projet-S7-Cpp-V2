#include "OdeSystem.h"
#include "FiniteDifference.h"
#include <cmath>
#include <algorithm>

using namespace std;
using namespace Eigen;

// Constructeur: initialise les paramètres et la grille
BlackScholesODE::BlackScholesODE(double r, double sigma, double K, double T,
                                 const VectorXd& S_grid)
    : r_(r), sigma_(sigma), K_(K), T_(T), S_grid_(S_grid), n_(S_grid.size()) {
    
    // Calcule le pas d'espace (supposé uniforme)
    if (n_ > 1) {
        dS_ = S_grid_(1) - S_grid_(0);
    } else {
        dS_ = 1.0;
    }
}

// Condition au bord pour S → ∞: V(S_max, τ) ≈ S_max - K*exp(-r*τ)
// Cette approximation vient du fait que pour S >> K, N(d1) ≈ 1 et N(d2) ≈ 1
double BlackScholesODE::boundaryConditionLarge(double tau) const {
    double S_max = S_grid_(n_ - 1); // Dernier point de la grille
    
    // Formule asymptotique pour un call européen quand S >> K
    return max(S_max - K_ * exp(-r_ * tau), 0.0);
}

// Applique les conditions aux limites de Dirichlet
void BlackScholesODE::applyBoundaryConditions(VectorXd& V, double tau) const {
    // Condition au bord S = 0: V(0, τ) = 0
    // Un call ne vaut rien si le spot est nul
    V(0) = 0.0;
    
    // Condition au bord S = S_max: V(S_max, τ) = S_max - K*exp(-r*τ)
    // Pour S grand, le call se comporte comme un forward
    V(n_ - 1) = boundaryConditionLarge(tau);
}

// Évalue F(V) = 0.5σ²S²∂²V/∂S² + rS∂V/∂S - rV
// C'est le membre de droite de l'équation de Black-Scholes en temps backward
VectorXd BlackScholesODE::evaluate(const VectorXd& V, double tau) {
    VectorXd F = VectorXd::Zero(n_);
    
    // Boucle sur les points intérieurs (les bords sont fixés par conditions aux limites)
    for (int i = 1; i < n_ - 1; ++i) {
        double S = S_grid_(i); // Prix spot au point i
        
        // Calcule ∂V/∂S avec différences centrées
        // ∂V/∂S ≈ (V_{i+1} - V_{i-1}) / (2*dS)
        double dV_dS = (V(i + 1) - V(i - 1)) / (2.0 * dS_);
        
        // Calcule ∂²V/∂S² avec différences centrées
        // ∂²V/∂S² ≈ (V_{i+1} - 2*V_i + V_{i-1}) / dS²
        double d2V_dS2 = (V(i + 1) - 2.0 * V(i) + V(i - 1)) / (dS_ * dS_);
        
        // Terme de diffusion: 0.5 * σ² * S² * ∂²V/∂S²
        double diffusion = 0.5 * sigma_ * sigma_ * S * S * d2V_dS2;
        
        // Terme de convection: r * S * ∂V/∂S
        double convection = r_ * S * dV_dS;
        
        // Terme de réaction: -r * V
        double reaction = -r_ * V(i);
        
        // Assemble l'équation: dV/dτ = diffusion + convection + reaction
        F(i) = diffusion + convection + reaction;
    }
    
    // Les bords restent à 0 car ils sont gérés par les conditions aux limites
    F(0) = 0.0;
    F(n_ - 1) = 0.0;
    
    return F;
}