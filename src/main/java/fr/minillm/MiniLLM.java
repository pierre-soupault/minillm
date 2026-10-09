package fr.minillm;

import java.util.*;

/**
 * MiniLLM : un "modèle de langage" minuscule pour comprendre le principe.
 *
 * Tâche : prédire le MOT SUIVANT à partir du mot courant (comme un LLM, mais
 * avec un contexte d'1 seul mot au lieu de milliers).
 *
 * Architecture :
 *   mot -> [embedding E] -> vecteur h (D nombres) -> [couche W,b] -> logits (un score par mot)
 *       -> softmax -> probabilités du mot suivant
 *
 * 3 phases :
 *   1. PRÉ-ENTRAÎNEMENT : on apprend sur un petit corpus (descente de gradient)
 *   2. RENFORCEMENT     : on "récompense" certaines réponses (REINFORCE, idée de base du RLHF)
 *   3. INFÉRENCE        : on génère du texte mot par mot
 *
 * Compiler / lancer :  javac MiniLLM.java && java MiniLLM
 */
public class MiniLLM {

    // ---------- Vocabulaire et corpus ----------
    static String[] VOCAB = {"le", "chat", "chien", "mange", "dort", "court", "."};
    static String CORPUS = "le chat mange . le chien mange . le chat dort . le chien court . "
                         + "le chat mange . le chien court .";

    static int V = VOCAB.length; // taille du vocabulaire
    static int D = 4;            // taille des embeddings (vecteur qui "représente" un mot)

    // ---------- Paramètres du modèle (ce qu'on apprend) ----------
    static double[][] E = new double[V][D]; // embeddings : une ligne par mot
    static double[][] W = new double[D][V]; // poids de sortie
    static double[] b = new double[V];      // biais de sortie

    static Random rnd = new Random(42);

    // ---------- Traçage de l'espace vectoriel ----------
    static boolean TRACER_ESPACE = true; // mettre false pour une sortie sans les cartes
    static int EPOQUES = 300;            // nombre d'époques de pré-entraînement
    static int PERIODE_TRACE = 25;       // un instantané des embeddings toutes les N époques
    static int ESSAIS_RL = 200;          // nombre d'essais de renforcement
    static String SORTIE_HTML = "target/espace-vectoriel.html"; // page interactive, null pour ne rien écrire

    public static void main(String[] args) {
        // Initialisation aléatoire : au départ le modèle ne sait rien
        for (int i = 0; i < V; i++) for (int d = 0; d < D; d++) E[i][d] = rnd.nextGaussian() * 0.5;
        for (int d = 0; d < D; d++) for (int j = 0; j < V; j++) W[d][j] = rnd.nextGaussian() * 0.5;

        // Tokenisation : le texte devient une suite d'entiers
        int[] tokens = Arrays.stream(CORPUS.trim().split(" ")).mapToInt(MiniLLM::id).toArray();
        System.out.println("Tokens : " + Arrays.toString(tokens));

        // =====================================================================
        // PHASE 1 : PRÉ-ENTRAÎNEMENT (apprentissage supervisé "mot suivant")
        // =====================================================================
        titre("PHASE 1 : PRÉ-ENTRAÎNEMENT");
        System.out.println("Avant entraînement :");
        afficherPredictions("chat");

        // Le journal garde une copie des embeddings à chaque instantané : c'est ce
        // qui permet de mesurer le déplacement d'un mot entre deux époques.
        EspaceVectoriel espace = new EspaceVectoriel(VOCAB, D);
        espace.enregistrer(0, Double.NaN, E); // position de départ (aléatoire)
        System.out.println("\nPositions initiales dans l'espace vectoriel :");
        espace.journaliserDernier();

        double lr = 0.1; // taux d'apprentissage : taille des pas de correction
        for (int epoch = 1; epoch <= EPOQUES; epoch++) {
            double perteTotale = 0;
            for (int t = 0; t < tokens.length - 1; t++) {
                int x = tokens[t];     // mot courant
                int y = tokens[t + 1]; // mot suivant attendu (la "bonne réponse")

                // --- Forward : calcul de la prédiction ---
                double[] h = E[x].clone();
                double[] p = softmax(logits(h));

                // Perte (cross-entropy) : grande si le modèle donnait une faible proba à la bonne réponse
                perteTotale += -Math.log(p[y]);

                // --- Backward : gradient = p - onehot(y) ---
                // "j'ai trop parié sur les mauvais mots, pas assez sur le bon"
                double[] dLogits = p.clone();
                dLogits[y] -= 1;

                // --- Mise à jour des poids (descente de gradient) ---
                double[] dh = new double[D];
                for (int d = 0; d < D; d++)
                    for (int j = 0; j < V; j++) {
                        dh[d] += W[d][j] * dLogits[j];
                        W[d][j] -= lr * h[d] * dLogits[j];
                    }
                for (int j = 0; j < V; j++) b[j] -= lr * dLogits[j];
                for (int d = 0; d < D; d++) E[x][d] -= lr * dh[d];
            }
            double perteMoyenne = perteTotale / (tokens.length - 1);

            // On enregistre à CHAQUE époque (un instantané coûte V*D doubles) :
            // les trajectoires tracées plus bas sont ainsi complètes.
            if (TRACER_ESPACE) espace.enregistrer(epoch, perteMoyenne, E);

            boolean tracer = TRACER_ESPACE
                    && (epoch <= 5 || epoch % PERIODE_TRACE == 0 || epoch == EPOQUES);
            boolean predire = epoch <= 5 || epoch % 50 == 0;

            if (tracer || predire) {
                System.out.printf("%nÉpoque %3d | perte moyenne = %.4f%n", epoch, perteMoyenne);
            }
            if (predire) afficherPredictions("chat");
            if (tracer) {
                // Chaque mot vu pendant l'époque a été tiré vers ses contextes :
                // le tableau dit de combien (|pas|) et dans quelle direction (axe).
                espace.journaliserDernier();
            }
        }

        // Relecture de l'historique : les cartes partagent la même base de
        // projection, donc un mot qui se déplace d'une carte à l'autre s'est
        // vraiment déplacé dans l'espace en D dimensions.
        if (TRACER_ESPACE) {
            espace.rejouer(6);
            espace.carteTrajectoires();
            espace.resume();
        }

        // Avec un gros corpus, les mots utilisés dans des contextes proches finissent avec des
        // vecteurs proches. Ici le corpus est minuscule, donc ce n'est pas garanti : observe !
        System.out.println("\nEmbeddings appris (la 'représentation' de chaque mot) :");
        for (String m : new String[]{"chat", "chien", "mange", "dort"})
            System.out.printf("  %-6s %s%n", m, vec(E[id(m)]));
        System.out.printf("  similarité(chat, chien) = %.3f%n", cosinus(E[id("chat")], E[id("chien")]));
        System.out.printf("  similarité(chat, dort)  = %.3f%n", cosinus(E[id("chat")], E[id("dort")]));

        // =====================================================================
        // PHASE 2 : RENFORCEMENT (REINFORCE)
        // On ne donne plus "la bonne réponse" : le modèle génère, et on note.
        // Préférence humaine simulée : "on veut que le chat dorme".
        // =====================================================================
        titre("PHASE 2 : RENFORCEMENT");
        System.out.println("Avant renforcement :");
        afficherPredictions("chat");

        // Le renforcement ne touche pas à E : ce qui se déplace, ce sont les
        // colonnes de W, un vecteur de sortie par mot, dans le même espace que
        // les embeddings. C'est ce que suit ce journal.
        JournalRenforcement renfort = new JournalRenforcement(VOCAB, D, id("chat"), id("dort"), E[id("chat")]);
        renfort.enregistrer(0, W, b, -1, Double.NaN, softmax(logits(E[id("chat")])));

        double lrRL = 0.05;
        for (int essai = 1; essai <= ESSAIS_RL; essai++) {
            int x = id("chat");
            double[] h = E[x].clone();
            double[] p = softmax(logits(h));
            int a = echantillonner(p, 1.0);              // le modèle "agit" (choisit un mot)
            double recompense = (a == id("dort")) ? 1.0 : -0.3; // le "juge" note

            // Gradient de log p(a) = onehot(a) - p ; on le pousse dans le sens de la récompense
            double[] g = new double[V];
            for (int j = 0; j < V; j++) g[j] = ((j == a) ? 1 : 0) - p[j];
            for (int d = 0; d < D; d++)
                for (int j = 0; j < V; j++) W[d][j] += lrRL * recompense * h[d] * g[j];
            for (int j = 0; j < V; j++) b[j] += lrRL * recompense * g[j];

            // Après la mise à jour : on veut voir l'effet de l'essai.
            if (TRACER_ESPACE) renfort.enregistrer(essai, W, b, a, recompense, softmax(logits(E[x])));

            boolean tracer = TRACER_ESPACE && (essai <= 3 || essai % PERIODE_TRACE == 0 || essai == ESSAIS_RL);
            if (essai % 50 == 0 || tracer) {
                System.out.printf("%nEssai %3d (dernier choix : '%s', récompense %+.1f)%n", essai, VOCAB[a], recompense);
            }
            if (essai % 50 == 0) afficherPredictions("chat");
            if (tracer) renfort.journaliserDernier();
        }
        if (TRACER_ESPACE) renfort.resume();

        // =====================================================================
        // PHASE 3 : INFÉRENCE (génération)
        // =====================================================================
        titre("PHASE 3 : INFÉRENCE");
        JournalInference parcours = new JournalInference(VOCAB);
        JournalInference journal = TRACER_ESPACE ? parcours : null;

        // À température 0 le tirage est déterministe : répéter la même amorce
        // redonnerait trois fois le même texte. Ce qui varie utilement ici,
        // c'est le mot de départ.
        System.out.println("Température 0,0 — déterministe, donc une génération par amorce :");
        for (String amorce : VOCAB) {
            System.out.printf("  depuis '%s'%s[%s]%n", amorce,
                    " ".repeat(Math.max(1, 9 - amorce.length())), generer(amorce, 8, 0.0, journal));
        }

        // Au-dessus de 0 le tirage est aléatoire : là, répéter la même amorce
        // montre la dispersion.
        for (double temp : new double[]{1.0, 2.0}) {
            for (String amorce : new String[]{"le", "chat"}) {
                System.out.printf("%nTempérature %.1f depuis '%s' : ", temp, amorce);
                for (int k = 0; k < 2; k++) System.out.print("[" + generer(amorce, 8, temp, journal) + "]  ");
            }
        }
        System.out.println("\n\n(température 0 = toujours le mot le plus probable ; plus haut = plus créatif/aléatoire)");

        if (TRACER_ESPACE) {
            // Le détail d'un parcours : mot -> h -> logits -> probas -> tirage.
            System.out.println("\nParcours détaillés :");
            detailler(parcours, "le", 0.0);
            detailler(parcours, "chat", 0.0);
            detailler(parcours, "chat", 2.0);

            if (SORTIE_HTML != null) {
                try {
                    java.nio.file.Path page = java.nio.file.Path.of(SORTIE_HTML);
                    new ExportHtml(espace, renfort, parcours).ecrire(page);
                    System.out.println("\nPage interactive : " + page.toAbsolutePath());
                } catch (java.io.IOException e) {
                    System.out.println("\nExport HTML impossible : " + e.getMessage());
                }
            }
        }
    }

    // ---------- Briques du réseau ----------

    static double[] logits(double[] h) {
        double[] z = b.clone();
        for (int d = 0; d < D; d++) for (int j = 0; j < V; j++) z[j] += h[d] * W[d][j];
        return z;
    }

    static double[] softmax(double[] z) {
        double max = Arrays.stream(z).max().getAsDouble(), somme = 0;
        double[] p = new double[z.length];
        for (int i = 0; i < z.length; i++) { p[i] = Math.exp(z[i] - max); somme += p[i]; }
        for (int i = 0; i < z.length; i++) p[i] /= somme;
        return p;
    }

    static int echantillonner(double[] p, double temperature) {
        if (temperature == 0) { // greedy : on prend le max
            int best = 0;
            for (int i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
            return best;
        }
        // On "aplatit" ou "accentue" la distribution selon la température
        double[] q = new double[p.length];
        double s = 0;
        for (int i = 0; i < p.length; i++) { q[i] = Math.pow(p[i], 1.0 / temperature); s += q[i]; }
        double r = rnd.nextDouble() * s, cumul = 0;
        for (int i = 0; i < q.length; i++) { cumul += q[i]; if (r <= cumul) return i; }
        return q.length - 1;
    }

    static String generer(String debut, int maxMots, double temperature) {
        return generer(debut, maxMots, temperature, null);
    }

    /** Même génération, en notant au passage tout ce qui traverse le réseau. */
    static String generer(String debut, int maxMots, double temperature, JournalInference journal) {
        StringBuilder sb = new StringBuilder(debut);
        int x = id(debut);
        if (journal != null) journal.demarrerSequence(temperature, debut);
        for (int i = 0; i < maxMots; i++) {
            double[] z = logits(E[x]);
            double[] p = softmax(z);
            int suivant = echantillonner(p, temperature);
            if (journal != null) journal.enregistrerPas(x, E[x], W, b, z, p, suivant);
            x = suivant;
            sb.append(" ").append(VOCAB[x]);
            if (VOCAB[x].equals(".")) break;
        }
        if (journal != null) journal.terminerSequence(sb.toString());
        return sb.toString();
    }

    /** Affiche le parcours de la première génération correspondant à ces paramètres. */
    static void detailler(JournalInference parcours, String amorce, double temperature) {
        for (JournalInference.Sequence sequence : parcours.sequences()) {
            if (!sequence.amorce().equals(amorce) || sequence.temperature() != temperature) continue;
            System.out.printf("%n  Température %.1f, depuis '%s'  |  « %s »%n",
                    sequence.temperature(), amorce, sequence.texte());
            for (JournalInference.Pas pas : sequence.pas()) {
                int meilleur = argmax(pas.probabilites());
                System.out.printf("    '%s' -> le plus probable est '%s' (p = %.2f) -> tirage : '%s'%s%n",
                        VOCAB[pas.motEntree()], VOCAB[meilleur], pas.probabilites()[meilleur],
                        VOCAB[pas.motChoisi()],
                        pas.motChoisi() == meilleur ? "" : "   <- la température a tranché autrement");
            }
            return;
        }
    }

    static int argmax(double[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    // ---------- Affichage ----------

    static void afficherPredictions(String mot) {
        double[] p = softmax(logits(E[id(mot)]));
        StringBuilder sb = new StringBuilder("  P(suivant | '" + mot + "') : ");
        for (int j = 0; j < V; j++) sb.append(String.format("%s=%.2f  ", VOCAB[j], p[j]));
        System.out.println(sb);
        System.out.println("  " + barres(p));
    }

    static String barres(double[] p) {
        StringBuilder sb = new StringBuilder();
        for (int j = 0; j < V; j++)
            if (p[j] > 0.05) sb.append(String.format("%s %s  ", VOCAB[j], "#".repeat((int) Math.round(p[j] * 20))));
        return sb.toString();
    }

    static void titre(String t) { System.out.println("\n==================== " + t + " ====================\n"); }
    static int id(String mot) { return Arrays.asList(VOCAB).indexOf(mot); }
    static String vec(double[] v) {
        StringBuilder sb = new StringBuilder("[");
        for (double x : v) sb.append(String.format("%6.2f", x));
        return sb.append(" ]").toString();
    }
    static double cosinus(double[] a, double[] c) {
        double ab = 0, aa = 0, cc = 0;
        for (int i = 0; i < a.length; i++) { ab += a[i] * c[i]; aa += a[i] * a[i]; cc += c[i] * c[i]; }
        return ab / Math.sqrt(aa * cc);
    }
}
