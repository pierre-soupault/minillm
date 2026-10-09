package fr.minillm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Journal de l'espace vectoriel : suit le déplacement des embeddings pendant
 * l'entraînement.
 *
 * Deux niveaux de lecture :
 *
 *   1. Le TABLEAU (journaliserDernier) : coordonnées brutes en dimension D,
 *      distance parcourue pendant l'époque, distance cumulée depuis le départ.
 *      C'est la mesure exacte, sans perte d'information.
 *
 *   2. Les CARTES (rejouer, carteTrajectoires) : une projection 2D de l'espace
 *      en D dimensions, calculée par ACP (analyse en composantes principales).
 *      La base de projection et l'échelle sont calculées UNE FOIS sur tout
 *      l'historique : les cartes successives sont donc comparables, un mot qui
 *      bouge d'une carte à l'autre a vraiment bougé.
 *
 * Limite à garder en tête : la carte écrase D dimensions sur 2. Le pourcentage
 * de variance expliquée affiché dit quelle part de la vraie géométrie survit à
 * la projection ; le reste est invisible sur la carte mais bien présent dans le
 * tableau.
 */
final class EspaceVectoriel {

    /** Taille de la carte ASCII, en caractères. */
    private static final int LARGEUR = 69;
    private static final int HAUTEUR = 21;

    /** Caractère marquant une position passée (la trace laissée par un mot). */
    private static final char TRACE = '·';

    /** Les embeddings de tous les mots à une époque donnée. */
    record Instantane(int epoque, double perte, double[][] embeddings) { }

    private final String[] vocab;
    private final int dimension;
    private final List<Instantane> historique = new ArrayList<>();
    private final double[] distanceCumulee;

    EspaceVectoriel(String[] vocab, int dimension) {
        this.vocab = vocab;
        this.dimension = dimension;
        this.distanceCumulee = new double[vocab.length];
    }

    // =====================================================================
    // Enregistrement
    // =====================================================================

    /**
     * Copie l'état courant des embeddings et met à jour la longueur du chemin
     * parcouru par chaque mot. Appelable à chaque époque : un instantané coûte
     * V * D doubles.
     */
    void enregistrer(int epoque, double perte, double[][] embeddings) {
        double[][] copie = new double[vocab.length][];
        for (int i = 0; i < vocab.length; i++) copie[i] = embeddings[i].clone();

        if (!historique.isEmpty()) {
            double[][] precedent = historique.get(historique.size() - 1).embeddings();
            for (int i = 0; i < vocab.length; i++) {
                distanceCumulee[i] += distance(precedent[i], copie[i]);
            }
        }
        historique.add(new Instantane(epoque, perte, copie));
    }

    // =====================================================================
    // Niveau 1 : le tableau des déplacements
    // =====================================================================

    /**
     * Affiche la position de chaque mot au dernier instantané et son
     * déplacement depuis l'instantané précédent. |pas| et cumul sont mesurés
     * dans l'espace complet (D dimensions), pas dans la projection 2D.
     */
    void journaliserDernier() {
        int n = historique.size();
        double[][] courant = historique.get(n - 1).embeddings();
        double[][] precedent = n > 1 ? historique.get(n - 2).embeddings() : null;

        System.out.printf("  %-7s %-30s %9s %9s %6s %7s%n",
                "mot", "vecteur (D=" + dimension + ")", "|pas|", "chemin", "axe", "norme");
        for (int i = 0; i < vocab.length; i++) {
            double pas = precedent == null ? 0 : distance(precedent[i], courant[i]);
            String axe = precedent == null ? "-" : axeDominant(precedent[i], courant[i]);
            System.out.printf("  %-7s %-30s %9.4f %9.4f %6s %7.3f%n",
                    vocab[i], vecteur(courant[i]), pas, distanceCumulee[i], axe, norme(courant[i]));
        }
    }

    /** La dimension dans laquelle le mot a le plus bougé, avec le signe. */
    private String axeDominant(double[] avant, double[] apres) {
        int meilleur = 0;
        for (int d = 1; d < dimension; d++) {
            if (Math.abs(apres[d] - avant[d]) > Math.abs(apres[meilleur] - avant[meilleur])) meilleur = d;
        }
        double delta = apres[meilleur] - avant[meilleur];
        if (Math.abs(delta) < 1e-9) return "-";
        return (delta > 0 ? "+d" : "-d") + meilleur;
    }

    // =====================================================================
    // Niveau 2 : les cartes
    // =====================================================================

    /**
     * Rejoue l'historique sous forme de cartes 2D successives : étiquette =
     * position courante du mot, '·' = positions passées.
     *
     * @param cartes nombre de cartes à afficher, réparties sur l'historique
     */
    void rejouer(int cartes) {
        if (historique.size() < 2 || cartes < 2) return;

        Projection vue = projection();
        double echelle = vue.echelle();
        double part1 = vue.part(0);
        double part2 = vue.part(1);

        titre("DÉPLACEMENT DANS L'ESPACE VECTORIEL (projection ACP 2D)");
        System.out.printf("  Base de projection et échelle fixes, calculées sur les %d instantanés :%n",
                historique.size());
        System.out.printf("  les cartes sont donc comparables entre elles.%n");
        System.out.printf("  Variance expliquée : axe 1 = %.0f%%, axe 2 = %.0f%% (total %.0f%% de la géométrie en D=%d).%n",
                100 * part1, 100 * part2, 100 * (part1 + part2), dimension);
        System.out.printf("  Bords de la carte = +/- %.2f ; '%c' = position passée.%n", echelle, TRACE);
        System.out.printf("  Époques espacées de façon logarithmique : presque tout le mouvement%n");
        System.out.printf("  a lieu dans les premières époques, ensuite le nuage se fige.%n");

        int dernier = historique.size() - 1;
        int precedent = -1;
        for (int c = 0; c < cartes; c++) {
            // Espacement cubique plutôt que régulier : sans ça, les 4 dernières
            // cartes sont identiques et on rate la phase intéressante.
            double fraction = c / (double) (cartes - 1);
            int s = (int) Math.round(dernier * Math.pow(fraction, 3));
            if (s <= precedent) s = precedent + 1;
            if (s > dernier) break;
            precedent = s;
            Instantane instantane = historique.get(s);

            Carte carte = new Carte(vue, echelle);
            for (int t = 0; t < s; t++) {                       // la trace du passé
                for (double[] v : historique.get(t).embeddings()) carte.tracer(v);
            }
            for (int i = 0; i < vocab.length; i++) {            // puis les positions actuelles
                carte.etiqueter(instantane.embeddings()[i], vocab[i]);
            }

            System.out.printf("%n--- Époque %3d", instantane.epoque());
            if (!Double.isNaN(instantane.perte())) System.out.printf(" | perte = %.4f", instantane.perte());
            System.out.printf(" | chemin cumulé = %.2f | rayon du nuage = %.2f ---%n",
                    cheminTotal(s), vue.rayon(instantane.embeddings()));
            carte.afficher();
        }
    }

    /**
     * Une seule carte avec la trajectoire complète de chaque mot : le chiffre
     * marque les positions traversées, l'étiquette la position finale.
     */
    void carteTrajectoires() {
        if (historique.size() < 2) return;

        Projection vue = projection();
        double echelle = vue.echelle();

        titre("TRAJECTOIRES COMPLÈTES");
        Carte carte = new Carte(vue, echelle);
        for (int i = 0; i < vocab.length; i++) {
            String marque = String.valueOf((char) ('1' + i % 9));
            for (int t = 0; t < historique.size() - 1; t++) {
                carte.marquer(historique.get(t).embeddings()[i], marque);
            }
        }
        double[][] fin = historique.get(historique.size() - 1).embeddings();
        for (int i = 0; i < vocab.length; i++) carte.etiqueter(fin[i], vocab[i]);
        carte.afficher();

        StringBuilder legende = new StringBuilder("  Légende : ");
        for (int i = 0; i < vocab.length; i++) {
            legende.append((char) ('1' + i % 9)).append("='").append(vocab[i]).append("'  ");
        }
        System.out.println(legende);
    }

    /** Bilan : qui a le plus bougé, et quels mots se sont rapprochés. */
    void resume() {
        if (historique.size() < 2) return;
        double[][] debut = historique.get(0).embeddings();
        double[][] fin = historique.get(historique.size() - 1).embeddings();

        titre("BILAN DU DÉPLACEMENT");
        System.out.printf("  %-7s %12s %12s %12s%n", "mot", "chemin", "deplacement", "detour");
        System.out.printf("  %-7s %12s %12s %12s%n", "", "parcouru", "net", "(chemin/net)");
        Integer[] ordre = new Integer[vocab.length];
        for (int i = 0; i < vocab.length; i++) ordre[i] = i;
        Arrays.sort(ordre, (a, b) -> Double.compare(distanceCumulee[b], distanceCumulee[a]));
        for (int i : ordre) {
            double net = distance(debut[i], fin[i]);
            System.out.printf("  %-7s %12.3f %12.3f %12s%n", vocab[i], distanceCumulee[i], net,
                    net < 1e-9 ? "-" : String.format("%.2fx", distanceCumulee[i] / net));
        }
        System.out.println("  (chemin >> net = le mot a fait des allers-retours avant de se stabiliser)");

        System.out.println("\n  Similarité cosinus début -> fin, paires qui ont le plus changé :");
        List<int[]> paires = new ArrayList<>();
        for (int i = 0; i < vocab.length; i++) {
            for (int j = i + 1; j < vocab.length; j++) paires.add(new int[] {i, j});
        }
        paires.sort((p, q) -> Double.compare(
                Math.abs(ecartCosinus(debut, fin, q)), Math.abs(ecartCosinus(debut, fin, p))));
        for (int k = 0; k < Math.min(5, paires.size()); k++) {
            int[] p = paires.get(k);
            double avant = MiniLLM.cosinus(debut[p[0]], debut[p[1]]);
            double apres = MiniLLM.cosinus(fin[p[0]], fin[p[1]]);
            System.out.printf("    %-7s / %-7s  %+.3f -> %+.3f  (%+.3f, %s)%n",
                    vocab[p[0]], vocab[p[1]], avant, apres, apres - avant,
                    apres > avant ? "rapprochement" : "eloignement");
        }
    }

    // =====================================================================
    // Accès pour l'exportateur HTML
    // =====================================================================

    List<Instantane> historique() {
        return historique;
    }

    int dimension() {
        return dimension;
    }

    String[] vocab() {
        return vocab;
    }

    /** Longueur du chemin parcouru par chaque mot sur tout l'historique. */
    double[] distancesCumulees() {
        return distanceCumulee.clone();
    }

    /** Les embeddings de l'instantané le plus récent. */
    double[][] embeddingsFinaux() {
        return historique.get(historique.size() - 1).embeddings();
    }

    private double ecartCosinus(double[][] debut, double[][] fin, int[] paire) {
        return MiniLLM.cosinus(fin[paire[0]], fin[paire[1]])
                - MiniLLM.cosinus(debut[paire[0]], debut[paire[1]]);
    }

    // =====================================================================
    // Projection : déléguée à Projection, partagée avec les autres phases
    // =====================================================================

    private Projection projection;
    private int tailleProjetee = -1;

    /** La projection ajustée sur tout l'historique, recalculée s'il a grandi. */
    Projection projection() {
        if (projection == null || tailleProjetee != historique.size()) {
            List<double[]> points = new ArrayList<>();
            for (Instantane instantane : historique) {
                points.addAll(Arrays.asList(instantane.embeddings()));
            }
            projection = Projection.ajuster(points, dimension);
            tailleProjetee = historique.size();
        }
        return projection;
    }

    // =====================================================================
    // Rendu ASCII
    // =====================================================================

    /** Une carte ASCII avec son repère et sa gestion des collisions d'étiquettes. */
    private final class Carte {

        private final char[][] cellules = new char[HAUTEUR][LARGEUR];
        private final boolean[][] prise = new boolean[HAUTEUR][LARGEUR];
        private final Projection vue;
        private final double echelle;

        Carte(Projection vue, double echelle) {
            this.vue = vue;
            this.echelle = echelle;
            for (char[] ligne : cellules) Arrays.fill(ligne, ' ');
            int cx = LARGEUR / 2;
            int cy = HAUTEUR / 2;
            for (int c = 0; c < LARGEUR; c++) cellules[cy][c] = '-';
            for (int r = 0; r < HAUTEUR; r++) cellules[r][cx] = '|';
            cellules[cy][cx] = '+';
        }

        /** Marque une position passée, sans écraser une étiquette déjà posée. */
        void tracer(double[] v) {
            marquer(v, String.valueOf(TRACE));
        }

        void marquer(double[] v, String marque) {
            int[] rc = cellule(v, marque.length());
            if (!prise[rc[0]][rc[1]]) cellules[rc[0]][rc[1]] = marque.charAt(0);
        }

        /**
         * Pose une étiquette, décalée verticalement si la place est déjà
         * occupée par une autre étiquette : sans ça, "le" et "chat" proches se
         * chevauchent et deviennent illisibles.
         */
        void etiqueter(double[] v, String texte) {
            int[] rc = cellule(v, texte.length());
            for (int decalage : new int[] {0, -1, 1, -2, 2, -3, 3, -4, 4}) {
                int ligne = rc[0] + decalage;
                if (ligne < 0 || ligne >= HAUTEUR) continue;
                if (libre(ligne, rc[1], texte.length())) {
                    ecrire(ligne, rc[1], texte);
                    return;
                }
            }
            ecrire(rc[0], rc[1], texte); // dernier recours : on écrase
        }

        private boolean libre(int ligne, int colonne, int longueur) {
            for (int k = 0; k < longueur; k++) {
                if (prise[ligne][colonne + k]) return false;
            }
            return true;
        }

        private void ecrire(int ligne, int colonne, String texte) {
            for (int k = 0; k < texte.length(); k++) {
                cellules[ligne][colonne + k] = texte.charAt(k);
                prise[ligne][colonne + k] = true;
            }
        }

        /** Coordonnées écran du vecteur v, pour un texte de la longueur donnée. */
        private int[] cellule(double[] v, int longueur) {
            double[] p = vue.projeter(v);
            int cx = LARGEUR / 2;
            int cy = HAUTEUR / 2;
            int colonne = cx + (int) Math.round(p[0] / echelle * (cx - 1));
            int ligne = cy - (int) Math.round(p[1] / echelle * (cy - 1));
            return new int[] {
                Math.clamp(ligne, 0, HAUTEUR - 1),
                Math.clamp(colonne, 0, LARGEUR - longueur)
            };
        }

        void afficher() {
            for (char[] ligne : cellules) System.out.println("  " + new String(ligne).stripTrailing());
        }
    }

    private void titre(String texte) {
        System.out.println("\n---------- " + texte + " ----------\n");
    }

    // =====================================================================
    // Utilitaires
    // =====================================================================

    /** Somme des chemins parcourus par tous les mots jusqu'à l'instantané donné. */
    private double cheminTotal(int instantane) {
        double somme = 0;
        for (int t = 1; t <= instantane; t++) {
            double[][] avant = historique.get(t - 1).embeddings();
            double[][] apres = historique.get(t).embeddings();
            for (int i = 0; i < vocab.length; i++) somme += distance(avant[i], apres[i]);
        }
        return somme;
    }

    private double distance(double[] a, double[] b) {
        double somme = 0;
        for (int d = 0; d < a.length; d++) somme += (a[d] - b[d]) * (a[d] - b[d]);
        return Math.sqrt(somme);
    }

    private double norme(double[] v) {
        double somme = 0;
        for (double x : v) somme += x * x;
        return Math.sqrt(somme);
    }

    private String vecteur(double[] v) {
        StringBuilder sb = new StringBuilder("[");
        for (double x : v) sb.append(String.format("%6.2f", x));
        return sb.append(" ]").toString();
    }
}
