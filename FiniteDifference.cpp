#include "FiniteDifference.h"
#include <stdexcept>

using namespace std;
using namespace Eigen;

// Dérivée première centrée d'ordre 2: f'(x_i) ≈ (f_{i+1} - f_{i-1}) / (2h)
// Précision: O(h²)
double FiniteDifference::firstDerivativeCentered(const VectorXd& f, int i, double h) {
    // Vérifie que l'indice est valide pour le schéma centré
    if (i < 1 || i >= f.size() - 1) {
        throw out_of_range("Index out of range for centered difference");
    }
    
    // Formule des différences centrées
    return (f(i + 1) - f(i - 1)) / (2.0 * h);
}

// Dérivée première décentrée avant: f'(x_i) ≈ (f_{i+1} - f_i) / h
// Précision: O(h)
double FiniteDifference::firstDerivativeForward(const VectorXd& f, int i, double h) {
    // Vérifie que l'indice est valide
    if (i < 0 || i >= f.size() - 1) {
        throw out_of_range("Index out of range for forward difference");
    }
    
    // Formule des différences avant
    return (f(i + 1) - f(i)) / h;
}

// Dérivée première décentrée arrière: f'(x_i) ≈ (f_i - f_{i-1}) / h
// Précision: O(h)
double FiniteDifference::firstDerivativeBackward(const VectorXd& f, int i, double h) {
    // Vérifie que l'indice est valide
    if (i < 1 || i >= f.size()) {
        throw out_of_range("Index out of range for backward difference");
    }
    
    // Formule des différences arrière
    return (f(i) - f(i - 1)) / h;
}

// Dérivée seconde centrée: f''(x_i) ≈ (f_{i+1} - 2f_i + f_{i-1}) / h²
// Précision: O(h²)
double FiniteDifference::secondDerivativeCentered(const VectorXd& f, int i, double h) {
    // Vérifie que l'indice est valide pour le schéma centré
    if (i < 1 || i >= f.size() - 1) {
        throw out_of_range("Index out of range for centered second difference");
    }
    
    // Formule des différences centrées pour la dérivée seconde
    return (f(i + 1) - 2.0 * f(i) + f(i - 1)) / (h * h);
}

// Applique la dérivée première à tout le vecteur (utilise schémas décentrés aux bords)
VectorXd FiniteDifference::applyFirstDerivative(const VectorXd& f, double h) {
    int n = f.size();
    VectorXd df(n);
    
    // Au bord gauche (i=0): utilise le schéma décentré avant
    df(0) = firstDerivativeForward(f, 0, h);
    
    // Points intérieurs: utilise le schéma centré (plus précis)
    for (int i = 1; i < n - 1; ++i) {
        df(i) = firstDerivativeCentered(f, i, h);
    }
    
    // Au bord droit (i=n-1): utilise le schéma décentré arrière
    df(n - 1) = firstDerivativeBackward(f, n - 1, h);
    
    return df;
}

// Applique la dérivée seconde à tout le vecteur (valeurs nulles aux bords)
VectorXd FiniteDifference::applySecondDerivative(const VectorXd& f, double h) {
    int n = f.size();
    VectorXd d2f(n);
    
    // Aux bords, on met 0 (sera géré par les conditions aux limites)
    d2f(0) = 0.0;
    d2f(n - 1) = 0.0;
    
    // Points intérieurs: utilise le schéma centré
    for (int i = 1; i < n - 1; ++i) {
        d2f(i) = secondDerivativeCentered(f, i, h);
    }
    
    return d2f;
}