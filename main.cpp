#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <Eigen/Dense>

#include "Function.h"
#include "OdeSystem.h"
#include "TimeScheme.h"
#include "FiniteDifference.h"

using namespace std;
using namespace Eigen;

int main() {
    cout << "========================================" << endl;
    cout << "  Solveur Black-Scholes par Différences Finies" << endl;
    cout << "========================================\n" << endl;
    
    // ========================================================================
    // LECTURE DES PARAMÈTRES
    // ========================================================================
    cout << "Lecture des paramètres depuis parameters.dat..." << endl;
    Parameters params = readParameters("parameters.dat");
    
    // Affiche les paramètres lus
    cout << "\nParamètres du modèle:" << endl;
    cout << "  S0     = " << params.S0 << endl;
    cout << "  K      = " << params.K << endl;
    cout << "  r      = " << params.r << endl;
    cout << "  sigma  = " << params.sigma << endl;
    cout << "  T      = " << params.T << endl;
    cout << "  Smax   = " << params.Smax << endl;
    cout << "  Ns     = " << params.Ns << endl;
    cout << "  Nt     = " << params.Nt << endl;
    cout << "  Schéma = " << params.time_scheme << endl;
    
    // ========================================================================
    // CONSTRUCTION DE LA GRILLE SPATIALE
    // ========================================================================
    cout << "\nConstruction de la grille spatiale..." << endl;
    
    // Grille uniforme de S = 0 à S = Smax
    VectorXd S_grid(params.Ns);
    double dS = params.Smax / (params.Ns - 1);
    for (int i = 0; i < params.Ns; ++i) {
        S_grid(i) = i * dS;
    }
    
    cout << "  Grille: [0, " << params.Smax << "] avec " << params.Ns 
              << " points" << endl;
    cout << "  dS = " << dS << endl;
    
    // ========================================================================
    // CALCUL DU PAS DE TEMPS ET VÉRIFICATION CFL
    // ========================================================================
    double dtau = params.T / params.Nt;
    cout << "\nPas de temps:" << endl;
    cout << "  dτ = " << dtau << endl;
    
    // Condition CFL pour la stabilité d'Euler explicite
    // Pour l'équation de Black-Scholes: dτ ≤ dS² / (σ² * Smax²)
    double CFL_diffusion = dS * dS / (params.sigma * params.sigma * params.Smax * params.Smax);
    double CFL_convection = dS / (params.r * params.Smax);
    double CFL_limit = min(CFL_diffusion, CFL_convection);
    
    cout << "\nAnalyse de stabilité (pour Euler explicite):" << endl;
    cout << "  CFL diffusion  = " << CFL_diffusion << endl;
    cout << "  CFL convection = " << CFL_convection << endl;
    cout << "  CFL limite     = " << CFL_limit << endl;
    cout << "  Ratio dτ/CFL   = " << dtau / CFL_limit << endl;
    
    if (params.time_scheme == "euler_explicit" && dtau > CFL_limit) {
        cout << "  ⚠ ATTENTION: Le pas de temps dépasse la limite CFL!" << endl;
        cout << "  Le schéma Euler explicite peut être instable." << endl;
        cout << "  Considérez l'utilisation d'un schéma implicite ou réduisez Nt." << endl;
    } else if (params.time_scheme == "euler_implicit") {
        cout << "  ✓ Schéma implicite: inconditionnellement stable" << endl;
    } else {
        cout << "  ✓ Schéma RK: généralement stable pour ce pas de temps" << endl;
    }
    
    // ========================================================================
    // INITIALISATION DU SYSTÈME ET DU SCHÉMA TEMPOREL
    // ========================================================================
    cout << "\nInitialisation du solveur..." << endl;
    
    // Crée le système d'EDO pour Black-Scholes
    BlackScholesODE system(params.r, params.sigma, params.K, params.T, S_grid);
    
    // Crée le schéma temporel
    auto time_scheme = createTimeScheme(params.time_scheme, params.r, params.sigma, S_grid);
    cout << "  Schéma temporel: " << time_scheme->name() << endl;
    
    // ========================================================================
    // CONDITION INITIALE (PAYOFF À MATURITÉ)
    // ========================================================================
    cout << "\nApplication de la condition initiale (payoff)..." << endl;
    
    // Au temps τ = 0 (qui correspond à t = T), on a V = max(S - K, 0)
    BlackScholesExact exact_solver(params.r, params.sigma, params.K, params.T);
    VectorXd V(params.Ns);
    for (int i = 0; i < params.Ns; ++i) {
        V(i) = exact_solver.payoff(S_grid(i));
    }
    
    // ========================================================================
    // INTÉGRATION TEMPORELLE BACKWARD
    // ========================================================================
    cout << "\nIntégration temporelle (backward)..." << endl;
    cout << "  Nombre de pas de temps: " << params.Nt << endl;
    
    // On résout de τ = 0 (t = T) vers τ = T (t = 0)
    double tau = 0.0;
    
    // Affiche la progression tous les 10%
    int progress_step = params.Nt / 10;
    if (progress_step == 0) progress_step = 1;
    
    for (int n = 0; n < params.Nt; ++n) {
        // Affiche la progression
        if (n % progress_step == 0) {
            cout << "  Progression: " << setw(3) 
                      << (100 * n / params.Nt) << "%" << endl;
        }
        
        // Avance d'un pas de temps
        V = time_scheme->step(&system, V, tau, dtau);
        tau += dtau;
    }
    
    cout << "  Progression: 100%" << endl;
    cout << "Intégration terminée!" << endl;
    
    // ========================================================================
    // CALCUL DE LA SOLUTION EXACTE À t = 0
    // ========================================================================
    cout << "\nCalcul de la solution exacte..." << endl;
    VectorXd V_exact = exact_solver.callPriceVector(S_grid, 0.0);
    
    // ========================================================================
    // CALCUL DES ERREURS
    // ========================================================================
    cout << "\nAnalyse des erreurs:" << endl;
    
    double l2_error = computeL2Error(V, V_exact);
    double max_error = computeMaxError(V, V_exact);
    
    cout << "  Erreur L2   = " << scientific << l2_error << endl;
    cout << "  Erreur max  = " << scientific << max_error << endl;
    
    // ========================================================================
    // AFFICHAGE DU PRIX À S0
    // ========================================================================
    cout << "\n========================================" << endl;
    cout << "RÉSULTATS POUR S = S0 = " << params.S0 << endl;
    cout << "========================================" << endl;
    
    // Trouve l'indice le plus proche de S0
    int i0 = static_cast<int>(params.S0 / dS);
    if (i0 >= params.Ns) i0 = params.Ns - 1;
    
    cout << fixed << setprecision(6);
    cout << "Prix numérique: " << V(i0) << endl;
    cout << "Prix exact:     " << V_exact(i0) << endl;
    cout << "Erreur:         " << abs(V(i0) - V_exact(i0)) << endl;
    cout << "Erreur relative: " << setprecision(4) 
              << 100.0 * abs(V(i0) - V_exact(i0)) / V_exact(i0) << "%" << endl;
    
    // ========================================================================
    // SAUVEGARDE DES RÉSULTATS
    // ========================================================================
    cout << "\nSauvegarde des résultats dans 'results.dat'..." << endl;
    
    ofstream outfile("results.dat");
    outfile << "# S  V_numerical  V_exact  Error\n";
    outfile << scientific << setprecision(8);
    
    for (int i = 0; i < params.Ns; ++i) {
        outfile << S_grid(i) << "  " 
                << V(i) << "  " 
                << V_exact(i) << "  " 
                << abs(V(i) - V_exact(i)) << "\n";
    }
    
    outfile.close();
    cout << "Résultats sauvegardés!" << endl;
    
    cout << "\n========================================" << endl;
    cout << "Simulation terminée avec succès!" << endl;
    cout << "========================================" << endl;
    
    return 0;
}