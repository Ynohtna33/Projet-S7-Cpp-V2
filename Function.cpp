#include "Function.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;
using namespace Eigen;

// ============================================================================
// FONCTIONS POUR BLACK-SCHOLES EXACT
// ============================================================================

// Constructeur: initialise les paramètres du modèle
BlackScholesExact::BlackScholesExact(double r, double sigma, double K, double T)
    : r_(r), sigma_(sigma), K_(K), T_(T) {}

// Fonction de distribution cumulative normale standard (approximation)
double BlackScholesExact::normalCDF(double x) const {
    // Utilise l'approximation de la fonction d'erreur
    // CDF(x) = 0.5 * (1 + erf(x/sqrt(2)))
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}

// Calcule d1 = [ln(S/K) + (r + σ²/2)(T-t)] / (σ√(T-t))
double BlackScholesExact::d1(double S, double t) const {
    double tau = T_ - t; // Temps jusqu'à maturité
    
    // Cas particulier: à maturité
    if (tau < 1e-10) {
        return (S > K_) ? 1e10 : -1e10;
    }
    
    // Formule de d1
    double numerator = log(S / K_) + (r_ + 0.5 * sigma_ * sigma_) * tau;
    double denominator = sigma_ * sqrt(tau);
    
    return numerator / denominator;
}

// Calcule d2 = d1 - σ√(T-t)
double BlackScholesExact::d2(double S, double t) const {
    double tau = T_ - t; // Temps jusqu'à maturité
    
    // Cas particulier: à maturité
    if (tau < 1e-10) {
        return (S > K_) ? 1e10 : -1e10;
    }
    
    // d2 = d1 - σ√τ
    return d1(S, t) - sigma_ * sqrt(tau);
}

// Prix d'un call européen: V(S,t) = S*N(d1) - K*exp(-r(T-t))*N(d2)
double BlackScholesExact::callPrice(double S, double t) const {
    // Cas spécial: S = 0
    if (S < 1e-10) {
        return 0.0;
    }
    
    double tau = T_ - t; // Temps jusqu'à maturité
    
    // À maturité, on retourne le payoff
    if (tau < 1e-10) {
        return max(S - K_, 0.0);
    }
    
    // Calcule d1 et d2
    double d1_val = d1(S, t);
    double d2_val = d2(S, t);
    
    // Formule de Black-Scholes
    // V = S * N(d1) - K * exp(-r*τ) * N(d2)
    double price = S * normalCDF(d1_val) - K_ * exp(-r_ * tau) * normalCDF(d2_val);
    
    return max(price, 0.0); // Garantit un prix non-négatif
}

// Calcule le prix pour un vecteur de prix spots
VectorXd BlackScholesExact::callPriceVector(const VectorXd& S, double t) const {
    VectorXd result(S.size());
    
    // Applique callPrice à chaque élément du vecteur
    for (int i = 0; i < S.size(); ++i) {
        result(i) = callPrice(S(i), t);
    }
    
    return result;
}

// Payoff d'un call européen: max(S - K, 0)
double BlackScholesExact::payoff(double S) const {
    return max(S - K_, 0.0);
}

// ============================================================================
// FONCTIONS UTILITAIRES
// ============================================================================

// Fonction pour lire les paramètres depuis un fichier
Parameters readParameters(const string& filename) {
    Parameters params;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Erreur: impossible d'ouvrir " << filename << endl;
        exit(1);
    }
    
    string line;
    while (getline(file, line)) {
        // Ignore les lignes de commentaire (commençant par #)
        if (line.empty() || line[0] == '#') continue;
        
        istringstream iss(line);
        string key;
        iss >> key;
        
        // Lit la valeur correspondant à chaque clé
        if (key == "S0") iss >> params.S0;
        else if (key == "K") iss >> params.K;
        else if (key == "r") iss >> params.r;
        else if (key == "sigma") iss >> params.sigma;
        else if (key == "T") iss >> params.T;
        else if (key == "Smax") iss >> params.Smax;
        else if (key == "Ns") iss >> params.Ns;
        else if (key == "Nt") iss >> params.Nt;
        else if (key == "time_scheme") iss >> params.time_scheme;
    }
    
    file.close();
    return params;
}

// Fonction pour calculer l'erreur L2 entre deux vecteurs
double computeL2Error(const VectorXd& numerical, const VectorXd& exact) {
    // Calcule la norme L2 de la différence
    VectorXd diff = numerical - exact;
    return diff.norm() / exact.norm();
    //return diff.norm() / sqrt(diff.size());
}

// Fonction pour calculer l'erreur maximale (norme infinie)
double computeMaxError(const VectorXd& numerical, const VectorXd& exact) {
    // Calcule la norme infinie de la différence
    VectorXd diff = numerical - exact;
    return diff.cwiseAbs().maxCoeff();
}