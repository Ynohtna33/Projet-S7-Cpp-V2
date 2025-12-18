#ifndef ODESYSTEM_H
#define ODESYSTEM_H

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Classe abstraite pour un système d'EDO: dV/dτ = F(V)
class ODESystem {
public:
    virtual ~ODESystem() = default;
    
    // Évalue F(V) pour le système d'EDO
    virtual VectorXd evaluate(const VectorXd& V, double tau) = 0;
    
    // Retourne la taille du système
    virtual int size() const = 0;
};

// Système d'EDO pour Black-Scholes discrétisé en espace
// dV/dτ = 0.5σ²S²∂²V/∂S² + rS∂V/∂S - rV
// où τ = T - t (temps backward)
class BlackScholesODE : public ODESystem {
public:
    // Constructeur avec paramètres du modèle et grille spatiale
    BlackScholesODE(double r, double sigma, double K, double T,
                    const VectorXd& S_grid);
    
    // Évalue le membre de droite de l'EDO
    VectorXd evaluate(const VectorXd& V, double tau) override;
    
    // Retourne la taille du système
    int size() const override { return n_; }
    
    // Applique les conditions aux limites de Dirichlet
    void applyBoundaryConditions(VectorXd& V, double tau) const;
    
    // Retourne la grille spatiale
    const VectorXd& getGrid() const { return S_grid_; }

private:
    double r_;          // Taux sans risque
    double sigma_;      // Volatilité
    double K_;          // Strike
    double T_;          // Maturité
    VectorXd S_grid_;   // Grille des prix spots
    int n_;             // Nombre de points de grille
    double dS_;         // Pas d'espace
    
    // Calcule la condition au bord S → ∞
    double boundaryConditionLarge(double tau) const;
};

#endif // ODESYSTEM_H