# Jeu de données CWRU Bearing au format .npz

---



Ce dépôt présente le jeu de données CWRU Bearing au format .npz, corrigé et allégé en ce qui concerne certaines métadonnées. Tous les crédits pour ce jeu de données reviennent à la CWRU.

## Présentation du jeu de données

Le jeu de données CWRU Bearing est un jeu de données open-source fourni [ici](https://engineering.case.edu/bearingdatacenter) par la Case School of Engineering de la Case Western Reserve University. Les données correspondent à des séries temporelles mesurées à proximité et à distance des roulements d’un moteur électrique Reliance de 2 chevaux. En ce qui concerne la procédure expérimentale, la page officielle de la CWRU indique :

> Les roulements du moteur ont été endommagés artificiellement à l’aide de l’usinage par décharge électrique (EDM). Des défauts allant de 0,007 pouces à 0,040 pouces de diamètre ont été introduits séparément sur la piste intérieure, l’élément roulant (bille) et la piste extérieure. Les roulements endommagés ont été réinstallés dans le moteur d’essai et les données de vibration ont été enregistrées pour des charges moteur de 0 à 3 chevaux (vitesses de 1797 à 1720 tr/min).

Une présentation plus détaillée de la méthodologie est disponible [ici](https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures).

## Motivation

Ce dépôt a été créé pour deux raisons principales :

1. Le jeu de données original est fourni au format .mat, car MATLAB était (et reste encore dans une certaine mesure) l’outil de référence pour l’analyse de données en ingénierie. Cependant, avec les avancées du Deep Learning au cours de la dernière décennie, ce jeu de données est largement utilisé pour entraîner, évaluer et déployer des modèles DL. Les frameworks principaux pour ces tâches étant écrits en Python, la conversion des fichiers en format .npz permet aux chercheurs et passionnés de DL de les charger facilement, puis de les convertir en tenseurs pour les utiliser avec des réseaux de neurones ou d'autres modèles.
2. Les fichiers originaux contiennent certaines incohérences et (peut-être) des redondances dans leurs métadonnées (voir la section [Modifications](#changes) pour plus de détails). La version présentée ici ne contient que les séries temporelles nécessaires à l’analyse et au DL.

## Fichiers de données

Les fichiers de données originaux sont au format .mat et sont répartis en quatre « familles » différentes, en fonction de la charge du moteur et donc de sa vitesse en tr/min : 1797 (charge : 0 HP), 1772 (charge : 1 HP), 1750 (charge : 2 HP) et 1730 (charge : 3 HP). Pour chaque valeur de tr/min, les données sont divisées en trois grandes catégories :

* **Données de référence normales** , contenant des séries temporelles de roulements sains,
* **Données de défauts sur le roulement côté entraînement (Drive End - DE)** , contenant des séries temporelles de roulements avec défaut ponctuel sur l’extrémité entraînement,
* **Données de défauts sur le roulement côté ventilateur (Fan End - FE)** , contenant des séries temporelles de roulements avec défaut côté ventilateur.

Pour le cas DE, les données ont été collectées à deux fréquences différentes : 12 kHz et 48 kHz, tandis que les données FE ont été collectées à 12 kHz.

Concernant les données de défauts, les fichiers sont encore subdivisés en fonction de :

1. la taille du défaut, qui peut être de 0,007", 0,014", 0,021" ou 0,028" ;
2. le type de défaut, selon qu’il a été introduit sur la  **piste intérieure (IR)** , sur l’**élément roulant (bille - B)** ou sur la  **piste extérieure (OR)** .

En particulier pour les défauts OR, une distinction supplémentaire est faite selon la position par rapport à la zone de charge, centrée à 6h. Ainsi, une série temporelle peut correspondre à un défaut OR Centré (6h), OR Orthogonal (3h) ou OR Opposé (12h).

Chaque fichier `.mat` peut contenir une ou plusieurs séries temporelles liées à des données d’accéléromètres. Ces séries peuvent provenir de capteurs DE, FE ou base (BA). De plus, chaque fichier .mat possède un identifiant unique, visible dans la dernière clé des métadonnées du fichier .mat. Ce format est `X___`, où `___` est un code à trois chiffres.

Pour nos besoins, nous utiliserons le format `RPM_Fault_Diameter_End` pour identifier les fichiers contenant des anomalies, où :

* `RPM` : famille selon la vitesse en tr/min,
* `Fault` : type d’anomalie (`IR`, `B`, `OR@6`, `OR@3` ou `OR@12`),
* `Diameter` : diamètre du défaut en milli-pouces (`7`, `14`, `21` ou `28`),
* `End` : emplacement du capteur (`FE`, `DE12` ou `DE48`).

Pour les données de référence, on les désignera simplement par `RPM_Normal`. Notez que le format `X___` est utilisé dans la section [Modifications](#changes) pour illustrer les problèmes des fichiers .mat originaux.

## Modifications

1. La convention ci-dessus a été suivie pour nommer les fichiers .npz.
2. Toutes les métadonnées des fichiers .mat ont été supprimées. En effet, des incohérences existaient (par exemple, certaines fichiers IR contenaient une clé `'i'`, d'autres non). Cela signifie que le format `X___` n’a pas été conservé.
3. Les fichiers .npz ne contiennent que des séries temporelles. Leurs clés sont soit DE, FE ou BA selon le type de données d’accéléromètre. Le nombre de séries temporelles varie selon les fichiers (certains n’en ont qu’une, d’autres jusqu’à trois).
4. Le fichier `1750_Normal.mat` contient deux ensembles de séries temporelles DE/FE. Après inspection, une paire (les deux séries `X098`) est identique à celle du fichier `1772_Normal.mat`. Cette redondance a donc été supprimée.
5. Le fichier `1730_IR_21_DE48` contient également deux ensembles DE/FE. L’un des ensembles (les deux séries `X215`) est identique à celui du fichier `1750_IR_21_DE48`. Cette redondance a donc été supprimée.
6. Le fichier `1772_IR_14_DE48` contient une paire DE/FE qui semble correcte, mais aussi une série DE isolée (`X217`) qui ne ressemble pas aux deux séries `X217` présentes dans `1730_IR_21_DE48`. De plus, aucune clé `X217RPM` n’existe dans le fichier. Cette série temporelle `X217` a donc été supprimée.

La table "Contents" disponible [ici](/Contents.md) contient la liste de tous les fichiers .npz, ainsi que les types de séries temporelles qu’ils contiennent (DE, FE et/ou BA) et leur nombre.

## Chargement des fichiers

Pour charger un fichier .npz, utilisez la fonction `numpy.load()`, dont la documentation est disponible [ici](https://numpy.org/doc/stable/reference/generated/numpy.load.html). Les clés des tableaux contenus dans les fichiers .npz sont DE, FE et BA, selon la présence ou non de ces séries dans le fichier (voir la table "Contents" pour les détails [ici](/Contents.md)).

## Attribution

Tous les crédits pour le jeu de données reviennent à la CWRU. Cette version corrigée et réduite a été développée dans le cadre de la publication suivante :

@article{s24165310,
  author = {Rigas, Spyros and Tzouveli, Paraskevi and Kollias, Stefanos},
  title = {An End-to-End Deep Learning Framework for Fault Detection in Marine Machinery},
  journal = {Sensors},
  volume = {24},
  year = {2024},
  number = {16},
  doi = {10.3390/s24165310}
}

# Roulement informations

---

Les spécifications des roulements du côté moteur (drive end) et du côté ventilateur (fan end), incluant la géométrie des roulements et les fréquences de défaut, y sont listées.

## **Drive end bearing** (DE) : 6205-2RS JEM SKF, deep groove ball bearing

Description taille :

| Diamètre intérieur (m) | Diamètre extérieur (m) | Epaisseur (m) | Diamètre bille (m) | Diamètre de pas (m) |
| ------------------------ | ------------------------ | ------------- | ------------------- | -------------------- |
| 0,0250                   | 0,0520                   | 0,0150        | 0,0079              | 0,0390               |

Fréquence de défaut (multiple de la vitesse en Hz) :

| Bague interne | Bague externe | Cage    | Bille  |
| ------------- | ------------- | ------- | ------ |
| 5,4152        | 3,5848        | 0,39828 | 4,7135 |

## **Fan end bearing (FE)** : 6203-2RS JEM SKF, deep groove ball bearing

Description taille :

| Diamètre intérieur (m) | Diamètre extérieur (m) | Epaisseur (m) | Diamètre bille (m) | Diamètre de pas (m) |
| ------------------------ | ------------------------ | ------------- | ------------------- | -------------------- |
| 0,0170                   | 0,040                    | 0,0120        | 0,0067              | 0,0285               |

Fréquence de défaut (multiple de la vitesse en Hz) :

| Bague interne | Bague externe | Cage   | Bille  |
| ------------- | ------------- | ------ | ------ |
| 4,9469        | 3,0530        | 0,3817 | 3,9874 |
