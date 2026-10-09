package fr.minillm;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Écrit une page HTML autonome montrant les trois phases : le déplacement des
 * embeddings (pré-entraînement), celui des vecteurs de sortie (renforcement) et
 * le parcours d'un mot dans le réseau (inférence).
 *
 * Les données sont injectées en JSON dans le gabarit
 * {@code /espace-vectoriel.html} : la page s'ouvre depuis le disque, sans
 * serveur ni dépendance.
 *
 * Repères : la phase 1 compare des POSITIONS, son ACP est donc centrée sur le
 * nuage des embeddings. La phase 2 mesure un ALIGNEMENT avec l'embedding du
 * contexte ; elle n'utilise pas une ACP, qui mélangerait les axes et ferait
 * paraître alignés des vecteurs orthogonaux, mais une projection dirigée par h :
 * l'abscisse y vaut exactement h · W / ‖h‖. Chaque onglet annonce son repère.
 */
final class ExportHtml {

    private static final String MARQUEUR = "/*__DONNEES__*/";

    private final EspaceVectoriel phase1;
    private final JournalRenforcement phase2;
    private final JournalInference phase3;

    ExportHtml(EspaceVectoriel phase1, JournalRenforcement phase2, JournalInference phase3) {
        this.phase1 = phase1;
        this.phase2 = phase2;
        this.phase3 = phase3;
    }

    void ecrire(Path destination) throws IOException {
        String gabarit;
        try (InputStream flux = ExportHtml.class.getResourceAsStream("/espace-vectoriel.html")) {
            if (flux == null) {
                throw new IOException("gabarit /espace-vectoriel.html absent du classpath");
            }
            gabarit = new String(flux.readAllBytes(), StandardCharsets.UTF_8);
        }
        if (!gabarit.contains(MARQUEUR)) {
            throw new IOException("gabarit sans emplacement " + MARQUEUR);
        }
        if (destination.getParent() != null) Files.createDirectories(destination.getParent());
        Files.writeString(destination, gabarit.replace(MARQUEUR, json()), StandardCharsets.UTF_8);
    }

    private String json() {
        StringBuilder json = new StringBuilder(256 * 1024);
        json.append("{\n");
        json.append("\"dimension\":").append(phase1.dimension()).append(",\n");
        json.append("\"vocab\":[");
        String[] vocab = phase1.vocab();
        for (int m = 0; m < vocab.length; m++) {
            if (m > 0) json.append(',');
            json.append('"').append(echapper(vocab[m])).append('"');
        }
        json.append("],\n");
        json.append("\"phase1\":").append(jsonPhase1()).append(",\n");
        json.append("\"phase2\":").append(jsonPhase2()).append(",\n");
        json.append("\"phase3\":").append(jsonPhase3()).append("\n");
        json.append('}');
        return json.toString();
    }

    // =====================================================================
    // Phase 1 : déplacement des embeddings
    // =====================================================================

    private String jsonPhase1() {
        List<EspaceVectoriel.Instantane> historique = phase1.historique();
        int dimension = phase1.dimension();
        int mots = phase1.vocab().length;
        Projection vue = phase1.projection();

        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append(String.format(Locale.ROOT, "\"variance\":[%.4f,%.4f],%n", vue.part(0), vue.part(1)));
        json.append(String.format(Locale.ROOT, "\"echelle\":%.4f,%n", vue.echelle()));

        json.append("\"epoques\":[");
        for (int t = 0; t < historique.size(); t++) {
            if (t > 0) json.append(',');
            json.append(historique.get(t).epoque());
        }
        json.append("],\n\"pertes\":[");
        for (int t = 0; t < historique.size(); t++) {
            if (t > 0) json.append(',');
            json.append(nb(historique.get(t).perte(), 5));
        }
        json.append("],\n");

        List<double[][]> etats = new ArrayList<>();
        for (EspaceVectoriel.Instantane instantane : historique) etats.add(instantane.embeddings());
        json.append(jsonSuite(etats, vue, dimension, mots));

        json.append(",\n\"rayons\":[");
        for (int t = 0; t < historique.size(); t++) {
            if (t > 0) json.append(',');
            json.append(String.format(Locale.ROOT, "%.4f", vue.rayon(historique.get(t).embeddings())));
        }
        json.append("]\n}");
        return json.toString();
    }

    // =====================================================================
    // Phase 2 : déplacement des vecteurs de sortie
    // =====================================================================

    private String jsonPhase2() {
        List<JournalRenforcement.Instantane> historique = phase2.historique();
        int dimension = phase1.dimension();
        int mots = phase1.vocab().length;

        // Repère dirigé par h, l'embedding du contexte : l'abscisse d'un vecteur
        // de sortie y vaut exactement h · W / ‖h‖. Une ACP ordinaire mélangeait
        // les axes et faisait paraître alignés des vecteurs orthogonaux.
        double[] h = phase1.embeddingsFinaux()[phase2.motContexte()];
        List<double[]> points = new ArrayList<>(phase2.tousLesVecteurs());
        points.add(h);
        Projection vue = Projection.selonDirection(h, points, dimension);

        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append(String.format(Locale.ROOT, "\"variance\":[%.4f,%.4f],%n", vue.part(0), vue.part(1)));
        json.append(String.format(Locale.ROOT, "\"echelle\":%.4f,%n", vue.echelle()));
        json.append("\"contexte\":").append(phase2.motContexte()).append(",\n");
        json.append("\"cible\":").append(phase2.motCible()).append(",\n");

        json.append("\"essais\":[");
        for (int t = 0; t < historique.size(); t++) {
            if (t > 0) json.append(',');
            json.append(historique.get(t).essai());
        }
        json.append("],\n\"actions\":[");
        for (int t = 0; t < historique.size(); t++) {
            if (t > 0) json.append(',');
            json.append(historique.get(t).action());
        }
        json.append("],\n\"recompenses\":[");
        for (int t = 0; t < historique.size(); t++) {
            if (t > 0) json.append(',');
            json.append(nb(historique.get(t).recompense(), 3));
        }
        json.append("],\n\"recompenseMoyenne\":[");
        for (int t = 0; t < historique.size(); t++) {
            if (t > 0) json.append(',');
            json.append(nb(historique.get(t).recompenseMoyenne(), 4));
        }
        json.append("],\n\"probabilites\":[");
        for (int t = 0; t < historique.size(); t++) {
            if (t > 0) json.append(',');
            json.append(tableau(historique.get(t).probabilites(), 5));
        }
        json.append("],\n\"biais\":[");
        for (int t = 0; t < historique.size(); t++) {
            if (t > 0) json.append(',');
            json.append(tableau(historique.get(t).biais(), 4));
        }
        json.append("],\n");

        List<double[][]> etats = new ArrayList<>();
        for (JournalRenforcement.Instantane instantane : historique) etats.add(instantane.sorties());
        json.append(jsonSuite(etats, vue, dimension, mots));

        // Les embeddings, figés pendant toute la phase, projetés dans ce repère.
        json.append(",\n\"embeddings\":{\"positions\":[");
        double[][] figes = phase1.embeddingsFinaux();
        for (int m = 0; m < mots; m++) {
            if (m > 0) json.append(',');
            double[] p = vue.projeter(figes[m]);
            json.append(String.format(Locale.ROOT, "[%.4f,%.4f]", p[0], p[1]));
        }
        json.append("],\"vecteurs\":[");
        for (int m = 0; m < mots; m++) {
            if (m > 0) json.append(',');
            json.append(tableau(figes[m], 4));
        }
        json.append("]}\n}");
        return json.toString();
    }

    // =====================================================================
    // Phase 3 : parcours dans le réseau
    // =====================================================================

    private String jsonPhase3() {
        StringBuilder json = new StringBuilder();
        json.append("{\n\"sequences\":[");
        List<JournalInference.Sequence> sequences = phase3.sequences();
        for (int s = 0; s < sequences.size(); s++) {
            JournalInference.Sequence sequence = sequences.get(s);
            if (s > 0) json.append(",\n");
            json.append("{");
            json.append(String.format(Locale.ROOT, "\"temperature\":%.2f,", sequence.temperature()));
            json.append("\"amorce\":\"").append(echapper(sequence.amorce())).append("\",");
            json.append("\"texte\":\"").append(echapper(sequence.texte())).append("\",");
            json.append("\"pas\":[");
            for (int k = 0; k < sequence.pas().size(); k++) {
                JournalInference.Pas pas = sequence.pas().get(k);
                if (k > 0) json.append(',');
                json.append("{\"motEntree\":").append(pas.motEntree());
                json.append(",\"motChoisi\":").append(pas.motChoisi());
                json.append(",\"h\":").append(tableau(pas.h(), 4));
                json.append(",\"produits\":").append(tableau(pas.produits(), 4));
                json.append(",\"biais\":").append(tableau(pas.biais(), 4));
                json.append(",\"logits\":").append(tableau(pas.logits(), 4));
                json.append(",\"probabilites\":").append(tableau(pas.probabilites(), 5));
                json.append(",\"contributions\":[");
                for (int d = 0; d < pas.contributions().length; d++) {
                    if (d > 0) json.append(',');
                    json.append(tableau(pas.contributions()[d], 4));
                }
                json.append("]}");
            }
            json.append("]}");
        }
        json.append("]\n}");
        return json.toString();
    }

    // =====================================================================
    // Briques communes
    // =====================================================================

    /**
     * Sérialise une suite d'états (un tableau mot x dimension par instantané) :
     * positions projetées, vecteurs bruts, pas, chemin cumulé, normes et chemin
     * total. Le même bloc sert aux phases 1 et 2, qui suivent toutes deux un
     * nuage de vecteurs qui se déplace.
     */
    private String jsonSuite(List<double[][]> etats, Projection vue, int dimension, int mots) {
        StringBuilder positions = new StringBuilder("\"positions\":[");
        StringBuilder vecteurs = new StringBuilder("\"vecteurs\":[");
        StringBuilder pas = new StringBuilder("\"pas\":[");
        StringBuilder chemin = new StringBuilder("\"chemin\":[");
        StringBuilder normes = new StringBuilder("\"normes\":[");
        StringBuilder total = new StringBuilder("\"cheminTotal\":[");
        double[] cumul = new double[mots];

        for (int t = 0; t < etats.size(); t++) {
            double[][] courant = etats.get(t);
            double[][] precedent = t > 0 ? etats.get(t - 1) : null;
            if (t > 0) {
                positions.append(',');
                vecteurs.append(',');
                pas.append(',');
                chemin.append(',');
                normes.append(',');
                total.append(',');
            }
            positions.append('[');
            vecteurs.append('[');
            pas.append('[');
            chemin.append('[');
            normes.append('[');
            double somme = 0;
            for (int m = 0; m < mots; m++) {
                if (m > 0) {
                    positions.append(',');
                    vecteurs.append(',');
                    pas.append(',');
                    chemin.append(',');
                    normes.append(',');
                }
                double avance = precedent == null ? 0 : distance(precedent[m], courant[m]);
                cumul[m] += avance;
                somme += cumul[m];
                double[] p = vue.projeter(courant[m]);
                positions.append(String.format(Locale.ROOT, "[%.4f,%.4f]", p[0], p[1]));
                vecteurs.append(tableau(courant[m], 4));
                pas.append(String.format(Locale.ROOT, "%.5f", avance));
                chemin.append(String.format(Locale.ROOT, "%.4f", cumul[m]));
                normes.append(String.format(Locale.ROOT, "%.4f", norme(courant[m])));
            }
            positions.append(']');
            vecteurs.append(']');
            pas.append(']');
            chemin.append(']');
            normes.append(']');
            total.append(String.format(Locale.ROOT, "%.4f", somme));
        }

        return positions + "],\n" + vecteurs + "],\n" + pas + "],\n"
                + chemin + "],\n" + normes + "],\n" + total + "]";
    }

    /** Locale.ROOT partout : le séparateur décimal du JSON doit rester le point. */
    private static String nb(double valeur, int decimales) {
        return Double.isNaN(valeur) ? "null" : String.format(Locale.ROOT, "%." + decimales + "f", valeur);
    }

    private static String tableau(double[] valeurs, int decimales) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < valeurs.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(nb(valeurs[i], decimales));
        }
        return sb.append(']').toString();
    }

    private static double distance(double[] a, double[] b) {
        double somme = 0;
        for (int d = 0; d < a.length; d++) somme += (a[d] - b[d]) * (a[d] - b[d]);
        return Math.sqrt(somme);
    }

    private static double norme(double[] v) {
        double somme = 0;
        for (double x : v) somme += x * x;
        return Math.sqrt(somme);
    }

    private static String echapper(String texte) {
        return texte.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
