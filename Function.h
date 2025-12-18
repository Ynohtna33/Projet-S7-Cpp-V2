#ifndef FUNCTION_H
#define FUNCTION_H

#include <Eigen/Dense>
#include <string>

using namespace std;
using namespace Eigen;

// Structure pour stocker les paramètres lus depuis le fichier
struct Parameters {
    double S0;          // Prix spot initial
    double K;           // Strike
    double r;           // Taux sans risque
    double sigma;       // Volatilité
    double T;           // Maturité
    double Smax;        // Prix maximum du domaine
    int Ns;             // Nombre de points en espace
    int Nt;             // Nombre de pas de temps
    string time_scheme; // Nom du schéma temporel
};

// Classe pour calculer la solution exacte de Black-Scholes
class BlackScholesExact {
public:
    // Constructeur avec les paramètres du modèle
    BlackScholesExact(double r, double sigma, double K, double T);
    
    // Calcule le prix d'un call européen au temps t pour un prix spot S
    double callPrice(double S, double t) const;
    
    // Calcule le prix d'un call pour un vecteur de prix spots au temps t
    VectorXd callPriceVector(const VectorXd& S, double t) const;
    
    // Fonction de payoff pour un call européen
    double payoff(double S) const;
    
private:
    double r_;      // Taux sans risque
    double sigma_;  // Volatilité
    double K_;      // Strike
    double T_;      // Maturité
    
    // Fonction de distribution cumulative normale standard
    double normalCDF(double x) const;
    
    // Calcule d1 dans la formule de Black-Scholes
    double d1(double S, double t) const;
    
    // Calcule d2 dans la formule de Black-Scholes
    double d2(double S, double t) const;
};

// Fonctions utilitaires pour le main
// Fonction pour lire les paramètres depuis un fichier
Parameters readParameters(const string& filename);

// Fonction pour calculer l'erreur L2 entre deux vecteurs
double computeL2Error(const VectorXd& numerical, const VectorXd& exact);

// Fonction pour calculer l'erreur maximale (norme infinie)
double computeMaxError(const VectorXd& numerical, const VectorXd& exact);

#endif // FUNCTION_H