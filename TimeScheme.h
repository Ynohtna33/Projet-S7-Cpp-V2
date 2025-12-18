#ifndef TIMESCHEME_H
#define TIMESCHEME_H

#include "OdeSystem.h"
#include <Eigen/Dense>
#include <string>
#include <memory>

using namespace std;
using namespace Eigen;

// Classe abstraite pour les schémas d'intégration temporelle
class TimeScheme {
public:
    virtual ~TimeScheme() = default;
    
    // Avance la solution d'un pas de temps: V^{n+1} = TimeScheme(V^n, τ^n, dτ)
    virtual VectorXd step(ODESystem* system, 
                          const VectorXd& V,
                          double tau, 
                          double dtau) = 0;
    
    // Retourne le nom du schéma
    virtual string name() const = 0;
};

// Schéma d'Euler explicite: V^{n+1} = V^n + dτ * F(V^n)
// Ordre 1 en temps, conditionnellement stable
// Condition CFL: dτ ≤ min(dS²/(σ²S_max²), 1/r) pour la stabilité
class EulerExplicit : public TimeScheme {
public:
    VectorXd step(ODESystem* system, 
                  const VectorXd& V,
                  double tau, 
                  double dtau) override;
    
    string name() const override { return "Euler Explicit"; }
};

// Schéma d'Euler implicite: V^{n+1} = V^n + dτ * F(V^{n+1})
// Ordre 1 en temps, inconditionnellement stable
// Nécessite la résolution d'un système linéaire à chaque pas
class EulerImplicit : public TimeScheme {
public:
    EulerImplicit(double r, double sigma, const VectorXd& S_grid);
    
    VectorXd step(ODESystem* system, 
                  const VectorXd& V,
                  double tau, 
                  double dtau) override;
    
    string name() const override { return "Euler Implicit"; }

private:
    double r_;
    double sigma_;
    VectorXd S_grid_;
    double dS_;
    
    // Résout le système linéaire (I - dτ*J)*V^{n+1} = V^n par Thomas (tridiagonal)
    VectorXd solveTridiagonal(const VectorXd& a,
                              const VectorXd& b,
                              const VectorXd& c,
                              const VectorXd& d) const;
};

// Schéma de Runge-Kutta d'ordre 2 (RK2)
// V^{n+1} = V^n + dτ/2 * (k1 + k2)
// où k1 = F(V^n), k2 = F(V^n + dτ*k1)
// Ordre 2 en temps
class RungeKutta2 : public TimeScheme {
public:
    VectorXd step(ODESystem* system, 
                  const VectorXd& V,
                  double tau, 
                  double dtau) override;
    
    string name() const override { return "Runge-Kutta 2"; }
};

// Schéma de Runge-Kutta d'ordre 3 (RK3)
// Ordre 3 en temps
class RungeKutta3 : public TimeScheme {
public:
    VectorXd step(ODESystem* system, 
                  const VectorXd& V,
                  double tau, 
                  double dtau) override;
    
    string name() const override { return "Runge-Kutta 3"; }
};

// Schéma de Runge-Kutta d'ordre 4 (RK4) - schéma classique
// V^{n+1} = V^n + dτ/6 * (k1 + 2*k2 + 2*k3 + k4)
// Ordre 4 en temps
class RungeKutta4 : public TimeScheme {
public:
    VectorXd step(ODESystem* system, 
                  const VectorXd& V,
                  double tau, 
                  double dtau) override;
    
    string name() const override { return "Runge-Kutta 4"; }
};

// Fonction utilitaire pour créer le schéma temporel selon le nom
unique_ptr<TimeScheme> createTimeScheme(const string& name,
                                        double r, double sigma,
                                        const VectorXd& S_grid);

#endif // TIMESCHEME_H