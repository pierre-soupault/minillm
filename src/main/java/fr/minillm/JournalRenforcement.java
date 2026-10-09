package fr.minillm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Journal de la phase de renforcement.
 *
 * Point important : le renforcement ne touche PAS aux embeddings. Il ne met à
 * jour que W et b. Les mots ne se déplacent donc plus dans l'espace vectoriel —
 * ce qui bouge, ce sont les VECTEURS DE SORTIE, c'est-à-dire les colonnes de W.
 *
 * Or une colonne de W vit dans le même espace que les embeddings :
 *
 *   logit_j = h · W[:,j] + b_j     avec h = E[mot courant]
 *
 * Le vecteur de sortie d'un mot est donc un « détecteur » : plus il est aligné
 * avec l'embedding du mot courant, plus ce mot est probable ensuite. C'est la
 * même dualité que dans un vrai LLM entre la matrice d'embedding et celle de
 * dé-embedding. On peut les projeter dans le même repère et regarder le vecteur
 * de sortie récompensé venir s'aligner sur l'embedding du contexte.
 */
final class JournalRenforcement {

    /** Un instantané de la couche de sortie à un essai donné. */
    record Instantane(int essai, double[][] sorties, double[] biais, int action,
                      double recompense, double[] probabilites, double recompenseMoyenne) { }

    private final String[] vocab;
    private final int dimension;
    private final int motContexte;
    private final int motCible;
    private final double[] embeddingContexte;
    private final List<Instantane> historique = new ArrayList<>();
    private final double[] distanceCumulee;
    private double sommeRecompenses;
    private int essaisVus;

    JournalRenforcement(String[] vocab, int dimension, int motContexte, int motCible,
                        double[] embeddingContexte) {
        this.vocab = vocab;
        this.dimension = dimension;
        this.motContexte = motContexte;
        this.motCible = motCible;
        this.embeddingContexte = embeddingContexte.clone();
        this.distanceCumulee = new double[vocab.length];
    }

    /**
     * Enregistre l'état de la couche de sortie. {@code poids} est la matrice W
     * telle que le modèle la stocke, en D x V ; on en extrait les colonnes pour
     * obtenir un vecteur par mot.
     */
    void enregistrer(int essai, double[][] poids, double[] biais, int action,
                     double recompense, double[] probabilites) {
        double[][] sorties = new double[vocab.length][dimension];
        for (int m = 0; m < vocab.length; m++) {
            for (int d = 0; d < dimension; d++) sorties[m][d] = poids[d][m];
        }

        if (!historique.isEmpty()) {
            double[][] precedent = historique.get(historique.size() - 1).sorties();
            for (int m = 0; m < vocab.length; m++) {
                distanceCumulee[m] += distance(precedent[m], sorties[m]);
            }
        }
        if (!Double.isNaN(recompense)) {
            sommeRecompenses += recompense;
            essaisVus++;
        }
        double moyenne = essaisVus == 0 ? Double.NaN : sommeRecompenses / essaisVus;

        historique.add(new Instantane(essai, sorties, biais.clone(), action, recompense,
                probabilites.clone(), moyenne));
    }

    /** Tableau des vecteurs de sortie au dernier instantané. */
    void journaliserDernier() {
        int n = historique.size();
        Instantane courant = historique.get(n - 1);
        double[][] precedent = n > 1 ? historique.get(n - 2).sorties() : null;

        System.out.printf("  %-7s %-30s %9s %9s %8s %7s%n",
                "mot", "vecteur de sortie (D=" + dimension + ")", "|pas|", "chemin", "biais",
                "P(.|" + vocab[motContexte] + ")");
        for (int m = 0; m < vocab.length; m++) {
            double pas = precedent == null ? 0 : distance(precedent[m], courant.sorties()[m]);
            System.out.printf("  %-7s %-30s %9.4f %9.4f %8.3f %7.3f%s%n",
                    vocab[m], vecteur(courant.sorties()[m]), pas, distanceCumulee[m],
                    courant.biais()[m], courant.probabilites()[m],
                    m == motCible ? "  <- cible" : "");
        }
    }

    /** Bilan : la cible a-t-elle gagné, et au prix de quel déplacement ? */
    void resume() {
        if (historique.size() < 2) return;
        Instantane debut = historique.get(0);
        Instantane fin = historique.get(historique.size() - 1);

        System.out.println("\n---------- BILAN DU RENFORCEMENT ----------\n");
        System.out.printf("  Contexte : '%s'   cible récompensée : '%s'%n",
                vocab[motContexte], vocab[motCible]);
        System.out.printf("  P('%s' | '%s') : %.3f -> %.3f%n", vocab[motCible], vocab[motContexte],
                debut.probabilites()[motCible], fin.probabilites()[motCible]);
        System.out.printf("  Récompense moyenne : %+.3f%n", fin.recompenseMoyenne());

        // h . W[:,m] est le logit du mot m avant le biais : c'est la grandeur que
        // le renforcement manipule, l'alignement entre le contexte et chaque
        // vecteur de sortie.
        System.out.printf("%n  %-7s %10s %10s %20s%n", "mot", "chemin", "deplac.", "h . W[:,mot]");
        System.out.printf("  %-7s %10s %10s %20s%n", "", "parcouru", "net", "debut -> fin");
        for (int m = 0; m < vocab.length; m++) {
            System.out.printf("  %-7s %10.3f %10.3f %9.3f -> %7.3f%s%n", vocab[m], distanceCumulee[m],
                    distance(debut.sorties()[m], fin.sorties()[m]),
                    produitScalaire(embeddingContexte, debut.sorties()[m]),
                    produitScalaire(embeddingContexte, fin.sorties()[m]),
                    m == motCible ? "  <- cible" : "");
        }
        System.out.println("  (les embeddings, eux, n'ont pas bougé d'un pouce : le RL ne touche que W et b)");
    }

    List<Instantane> historique() {
        return historique;
    }

    int motContexte() {
        return motContexte;
    }

    int motCible() {
        return motCible;
    }

    /** Tous les vecteurs de sortie de l'historique, pour ajuster la projection. */
    List<double[]> tousLesVecteurs() {
        List<double[]> points = new ArrayList<>();
        for (Instantane instantane : historique) points.addAll(Arrays.asList(instantane.sorties()));
        return points;
    }

    private double produitScalaire(double[] a, double[] b) {
        double somme = 0;
        for (int d = 0; d < a.length; d++) somme += a[d] * b[d];
        return somme;
    }

    private double distance(double[] a, double[] b) {
        double somme = 0;
        for (int d = 0; d < a.length; d++) somme += (a[d] - b[d]) * (a[d] - b[d]);
        return Math.sqrt(somme);
    }

    private String vecteur(double[] v) {
        StringBuilder sb = new StringBuilder("[");
        for (double x : v) sb.append(String.format("%6.2f", x));
        return sb.append(" ]").toString();
    }
}
