# minillm

Minimal LLM implementation in Java.

## Prerequis

- JDK 25 (Temurin 25.0.4 verifie)
- Maven 3.9+

## Commandes

```bash
mvn clean verify     # compile + tests
mvn exec:java        # lance fr.minillm.MiniLLM
                     # (exec-maven-plugin : necessite le reseau au 1er lancement)
mvn package          # produit target/minillm-0.1.0-SNAPSHOT.jar
                     # et target/minillm-0.1.0-SNAPSHOT-all.jar (shade, executable)
java -jar target/minillm-0.1.0-SNAPSHOT-all.jar
```

## Structure

```
pom.xml
src/main/java/io/github/ipso8110/minillm/
  MiniLLM.java                              # le modele : pre-entrainement, RL, inference
  Projection.java                           # ACP maison, partagee par les phases
  EspaceVectoriel.java                      # phase 1 : deplacement des embeddings
  JournalRenforcement.java                  # phase 2 : deplacement des vecteurs de sortie
  JournalInference.java                     # phase 3 : parcours d'un mot dans le reseau
  ExportHtml.java                           # assemble les trois phases en JSON
src/main/resources/
  espace-vectoriel.html                     # gabarit de la page interactive
src/main/resources/                         # ressources runtime
src/test/java/io/github/ipso8110/minillm/   # tests JUnit 5
src/test/resources/                         # ressources de test
```

## Page interactive

Chaque execution ecrit `target/espace-vectoriel.html`, une page autonome (aucune
dependance, aucun serveur) avec un onglet par phase. Une couleur et une forme par
mot partout, curseur temporel, table des valeurs et mode sombre.

- **Phase 1** : le deplacement des embeddings, epoque par epoque, plus un panneau
  par mot et les trajectoires completes.
- **Phase 2** : le renforcement ne touche pas a `E`. Ce qui bouge, ce sont les
  colonnes de `W`, les vecteurs de sortie. Le repere n'est pas une ACP mais une
  projection dirigee par `h`, l'embedding du contexte : l'abscisse y vaut
  exactement `h · W / ||h||`, donc l'alignement se lit sans approximation. On y
  voit que les gradients de REINFORCE sont tous des multiples de `h` : les
  vecteurs de sortie ne convergent pas vers `h`, ils glissent le long de son axe
  a ordonnee constante. Avec `P(cible | contexte)`, le cosinus de chaque vecteur
  avec `h`, et la distribution complete essai par essai.
- **Phase 3** : le parcours d'un mot dans le reseau, etage par etage
  (`mot -> h -> logits -> softmax -> tirage`), plus la decomposition de chaque
  logit dimension par dimension, qui repond a « pourquoi ce mot-la ». A `T = 0`
  le tirage est deterministe : une generation par amorce, ce qui varie est le mot
  de depart. Au-dessus, plusieurs tirages depuis la meme amorce montrent la
  dispersion.

A ouvrir directement dans un navigateur :

```bash
xdg-open target/espace-vectoriel.html
```

Pour desactiver le tracage et l'export, passer `TRACER_ESPACE` a `false` dans
`MiniLLM.java` ; `SORTIE_HTML` a `null` ne desactive que l'export.

## Notes

- Le miroir Maven configure dans `~/.m2/settings.xml`
  (`artefact-repo.pole-emploi.intra`) n'est joignable que sur le reseau
  interne. Hors de ce reseau, seuls les artefacts deja presents dans
  `~/.m2/repository` sont resolvables : les versions du `pom.xml` ont ete
  choisies parmi celles disponibles en cache.
