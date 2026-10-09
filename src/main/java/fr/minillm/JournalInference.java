package fr.minillm;

import java.util.ArrayList;
import java.util.List;

/**
 * Journal de la phase d'inférence : le parcours d'un mot à travers le réseau.
 *
 * À chaque pas de génération, le mot courant traverse quatre étages :
 *
 *   mot --[table E]--> h (D nombres)
 *       --[h · W[:,j] + b_j]--> un logit par mot candidat
 *       --[softmax / température]--> une probabilité par mot candidat
 *       --[tirage]--> le mot suivant, qui redevient l'entrée du pas suivant
 *
 * On garde la décomposition du produit scalaire, dimension par dimension :
 * contributions[d][j] = h[d] * W[d][j]. C'est là qu'on voit *pourquoi* un mot
 * gagne — quelle dimension de l'embedding pousse quel candidat.
 */
final class JournalInference {

    /** Un pas de génération, avec tout ce qui a circulé dans le réseau. */
    record Pas(int motEntree, double[] h, double[][] contributions, double[] produits,
               double[] biais, double[] logits, double[] probabilites, int motChoisi) { }

    /** Une génération complète, à une température donnée. */
    record Sequence(double temperature, String amorce, List<Pas> pas, String texte) { }

    private final String[] vocab;
    private final List<Sequence> sequences = new ArrayList<>();
    private List<Pas> courants;
    private double temperatureCourante;
    private String amorceCourante;

    JournalInference(String[] vocab) {
        this.vocab = vocab;
    }

    void demarrerSequence(double temperature, String amorce) {
        courants = new ArrayList<>();
        temperatureCourante = temperature;
        amorceCourante = amorce;
    }

    /**
     * Enregistre un pas. {@code poids} est W en D x V, comme le modèle la
     * stocke ; les contributions et les produits scalaires en sont déduits.
     */
    void enregistrerPas(int motEntree, double[] h, double[][] poids, double[] biais,
                        double[] logits, double[] probabilites, int motChoisi) {
        int dimension = h.length;
        double[][] contributions = new double[dimension][vocab.length];
        double[] produits = new double[vocab.length];
        for (int d = 0; d < dimension; d++) {
            for (int j = 0; j < vocab.length; j++) {
                contributions[d][j] = h[d] * poids[d][j];
                produits[j] += contributions[d][j];
            }
        }
        courants.add(new Pas(motEntree, h.clone(), contributions, produits, biais.clone(),
                logits.clone(), probabilites.clone(), motChoisi));
    }

    void terminerSequence(String texte) {
        sequences.add(new Sequence(temperatureCourante, amorceCourante, List.copyOf(courants), texte));
        courants = null;
    }

    List<Sequence> sequences() {
        return sequences;
    }
}
