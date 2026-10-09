package fr.minillm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * ACP maison : projette des vecteurs de dimension D sur un plan, sans
 * dépendance externe.
 *
 * La base est calculée UNE FOIS sur l'ensemble des points qu'on lui donne, puis
 * réutilisée telle quelle. C'est ce qui rend les cartes comparables : deux
 * instantanés projetés dans la même base peuvent être superposés, et un point
 * qui bouge d'une carte à l'autre a vraiment bougé.
 *
 * Les phases 1 et 2 partagent la même instance : les embeddings d'entrée et les
 * vecteurs de sortie vivent dans le même espace, donc dans le même repère.
 */
final class Projection {

    private final int dimension;
    private final double[] centre;
    private final double[][] base;   // 2 x D : les deux axes principaux
    private final double[] parts;    // part de variance portée par chaque axe
    private final double echelle;    // rayon de la carte, le point le plus éloigné

    private Projection(int dimension, double[] centre, double[][] base, double[] parts, double echelle) {
        this.dimension = dimension;
        this.centre = centre;
        this.base = base;
        this.parts = parts;
        this.echelle = echelle;
    }

    /** Ajuste une projection centrée sur le nuage : adaptée à des positions. */
    static Projection ajuster(List<double[]> points, int dimension) {
        return ajuster(points, dimension, true);
    }

    /**
     * Ajuste une projection sur un nuage de points.
     *
     * @param centrer {@code true} centre le repère sur le nuage, ce qui convient
     *     pour comparer des POSITIONS. {@code false} garde l'origine de l'espace
     *     comme origine du repère : indispensable dès qu'on veut lire des
     *     DIRECTIONS, car un produit scalaire se mesure depuis l'origine, pas
     *     depuis le centre de gravité du nuage.
     */
    static Projection ajuster(List<double[]> points, int dimension, boolean centrer) {
        double[] centre = new double[dimension];
        if (centrer) {
            for (double[] v : points) {
                for (int d = 0; d < dimension; d++) centre[d] += v[d];
            }
            for (int d = 0; d < dimension; d++) centre[d] /= Math.max(1, points.size());
        }

        double[][] covariance = new double[dimension][dimension];
        for (double[] v : points) {
            for (int a = 0; a < dimension; a++) {
                for (int b = 0; b < dimension; b++) {
                    covariance[a][b] += (v[a] - centre[a]) * (v[b] - centre[b]);
                }
            }
        }
        for (int a = 0; a < dimension; a++) {
            for (int b = 0; b < dimension; b++) covariance[a][b] /= Math.max(1, points.size());
        }

        double[] u1 = axePrincipal(covariance, dimension);
        double lambda1 = quadratique(covariance, u1, dimension);
        double[][] deflate = new double[dimension][dimension];
        for (int a = 0; a < dimension; a++) {
            for (int b = 0; b < dimension; b++) {
                deflate[a][b] = covariance[a][b] - lambda1 * u1[a] * u1[b];
            }
        }
        double[] u2 = axePrincipal(deflate, dimension);
        orthogonaliser(u2, u1, dimension);

        return construire(dimension, centre, u1, u2, points);
    }

    /**
     * Assemble la projection : échelle = le point le plus éloigné, parts = la
     * fraction de la dispersion totale que chaque axe retient.
     */
    private static Projection construire(int dimension, double[] centre,
                                         double[] u1, double[] u2, List<double[]> points) {
        double max = 0;
        double sommeX = 0;
        double sommeY = 0;
        double sommeTotale = 0;
        for (double[] v : points) {
            double x = 0;
            double y = 0;
            double carre = 0;
            for (int d = 0; d < dimension; d++) {
                double ecart = v[d] - centre[d];
                x += ecart * u1[d];
                y += ecart * u2[d];
                carre += ecart * ecart;
            }
            max = Math.max(max, Math.max(Math.abs(x), Math.abs(y)));
            sommeX += x * x;
            sommeY += y * y;
            sommeTotale += carre;
        }
        double[] parts = sommeTotale < 1e-12
                ? new double[] {0, 0}
                : new double[] {sommeX / sommeTotale, sommeY / sommeTotale};
        return new Projection(dimension, centre, new double[][] {u1, u2}, parts,
                max < 1e-9 ? 1 : max * 1.05);
    }

    /** Covariance sans retrait de moyenne : la dispersion autour de l'origine. */
    private static double[][] covarianceBrute(List<double[]> points, int dimension) {
        double[][] c = new double[dimension][dimension];
        for (double[] v : points) {
            for (int a = 0; a < dimension; a++) {
                for (int b = 0; b < dimension; b++) c[a][b] += v[a] * v[b];
            }
        }
        for (int a = 0; a < dimension; a++) {
            for (int b = 0; b < dimension; b++) c[a][b] /= Math.max(1, points.size());
        }
        return c;
    }

    /**
     * Projection DIRIGÉE : le premier axe est imposé, pris dans la direction de
     * {@code direction} ; le second est la direction principale de ce qu'il
     * reste une fois cette composante retirée. Le repère n'est pas centré.
     *
     * Intérêt par rapport à une ACP : l'abscisse d'un vecteur v vaut exactement
     * v · direction / ‖direction‖, donc le produit scalaire v · direction se lit
     * sans approximation sur l'axe horizontal. Une ACP, elle, mélange les axes
     * et peut faire paraître alignés deux vecteurs orthogonaux — inutilisable
     * dès qu'on veut montrer un alignement.
     */
    static Projection selonDirection(double[] direction, List<double[]> points, int dimension) {
        double normeDirection = 0;
        for (double x : direction) normeDirection += x * x;
        normeDirection = Math.sqrt(normeDirection);

        double[] u1 = new double[dimension];
        if (normeDirection > 1e-12) {
            for (int d = 0; d < dimension; d++) u1[d] = direction[d] / normeDirection;
        } else {
            u1[0] = 1;
        }

        // Second axe : ce que les points font perpendiculairement à u1.
        List<double[]> residus = new ArrayList<>();
        for (double[] v : points) {
            double produit = 0;
            for (int d = 0; d < dimension; d++) produit += v[d] * u1[d];
            double[] residu = new double[dimension];
            for (int d = 0; d < dimension; d++) residu[d] = v[d] - produit * u1[d];
            residus.add(residu);
        }
        double[][] covariance = covarianceBrute(residus, dimension);
        double[] u2 = axePrincipal(covariance, dimension);
        orthogonaliser(u2, u1, dimension);

        return construire(dimension, new double[dimension], u1, u2, points);
    }

    /** Projette un vecteur sur les deux axes principaux. */
    double[] projeter(double[] v) {
        double x = 0;
        double y = 0;
        for (int d = 0; d < dimension; d++) {
            x += (v[d] - centre[d]) * base[0][d];
            y += (v[d] - centre[d]) * base[1][d];
        }
        return new double[] {x, y};
    }

    /** Rayon du nuage formé par ces vecteurs, dans le plan de projection. */
    double rayon(double[][] vecteurs) {
        double max = 0;
        for (double[] v : vecteurs) {
            double[] p = projeter(v);
            max = Math.max(max, Math.hypot(p[0], p[1]));
        }
        return max;
    }

    double echelle() {
        return echelle;
    }

    double part(int axe) {
        return parts[axe];
    }

    /** Vecteur propre dominant, par itération de la puissance. */
    private static double[] axePrincipal(double[][] matrice, int dimension) {
        double[] u = new double[dimension];
        Arrays.fill(u, 1.0 / Math.sqrt(dimension));
        for (int iteration = 0; iteration < 500; iteration++) {
            double[] suivant = new double[dimension];
            for (int a = 0; a < dimension; a++) {
                for (int b = 0; b < dimension; b++) suivant[a] += matrice[a][b] * u[b];
            }
            double norme = 0;
            for (double x : suivant) norme += x * x;
            norme = Math.sqrt(norme);
            if (norme < 1e-12) return u;
            for (int a = 0; a < dimension; a++) u[a] = suivant[a] / norme;
        }
        return u;
    }

    private static double quadratique(double[][] matrice, double[] u, int dimension) {
        double somme = 0;
        for (int a = 0; a < dimension; a++) {
            for (int b = 0; b < dimension; b++) somme += u[a] * matrice[a][b] * u[b];
        }
        return somme;
    }

    private static void orthogonaliser(double[] v, double[] reference, int dimension) {
        double produit = 0;
        for (int d = 0; d < dimension; d++) produit += v[d] * reference[d];
        for (int d = 0; d < dimension; d++) v[d] -= produit * reference[d];
        double norme = 0;
        for (double x : v) norme += x * x;
        norme = Math.sqrt(norme);
        if (norme > 1e-12) {
            for (int d = 0; d < dimension; d++) v[d] /= norme;
        }
    }
}
