#ifndef FINITEDIFFERENCE_H
#define FINITEDIFFERENCE_H

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Classe pour les approximations par différences finies
class FiniteDifference {
public:
    // Dérivée première: schéma centré d'ordre 2
    // f'(x_i) ≈ (f_{i+1} - f_{i-1}) / (2*h)
    static double firstDerivativeCentered(const VectorXd& f, int i, double h);
    
    // Dérivée première: schéma décentré avant (forward)
    // f'(x_i) ≈ (f_{i+1} - f_i) / h
    static double firstDerivativeForward(const VectorXd& f, int i, double h);
    
    // Dérivée première: schéma décentré arrière (backward)
    // f'(x_i) ≈ (f_i - f_{i-1}) / h
    static double firstDerivativeBackward(const VectorXd& f, int i, double h);
    
    // Dérivée seconde: schéma centré d'ordre 2
    // f''(x_i) ≈ (f_{i+1} - 2*f_i + f_{i-1}) / h²
    static double secondDerivativeCentered(const VectorXd& f, int i, double h);
    
    // Applique la dérivée première centrée à tout le vecteur (sauf bords)
    static VectorXd applyFirstDerivative(const VectorXd& f, double h);
    
    // Applique la dérivée seconde centrée à tout le vecteur (sauf bords)
    static VectorXd applySecondDerivative(const VectorXd& f, double h);
};

#endif // FINITEDIFFERENCE_H