## EXPLICATION DES METHODES de comparaison

 
## Method 0 : CORREL (Correlation)
Description : Calcule la corrélation entre les deux histogrammes.
Résultat attendu : Une valeur de 1 indique une correspondance parfaite, tandis que des valeurs proches de 0 indiquent une faible correspondance.

### Method 1 : CHISQR (Chi-Square)
Description : Calcule la distance du chi carré entre les histogrammes.
Résultat attendu : Une valeur de 0 indique une correspondance parfaite. Des valeurs plus élevées indiquent une plus grande différence.

### Method 2 : INTERSECT (Intersection)
Description : Mesure l'intersection des histogrammes.
Résultat attendu : Des valeurs plus élevées indiquent une plus grande correspondance. La valeur maximale dépend de la taille des histogrammes.

### Method 3 : BHATTACHARYYA (Bhattacharyya distance)
Description : Calcule la distance de Bhattacharyya entre les histogrammes.
Résultat attendu : Une valeur de 0 indique une correspondance parfaite. Des valeurs plus élevées indiquent une plus grande différence.

## INTERPRETATION DES RESULTATS
+----------+-----------+-------------+----------------+----------------+
|   Method |   Perfect |   Base-Half |   Base-Test(1) |   Base-Test(2) |
+==========+===========+=============+================+================+
|        0 |   1       |    0.996665 |       0.526326 |       0.977149 |
+----------+-----------+-------------+----------------+----------------+
|        1 |   0       |    0.236579 |   12488.5      |       7.77917  |
+----------+-----------+-------------+----------------+----------------+
|        2 |   1.57766 |    1.41202  |       1.21467  |       1.2483   |
+----------+-----------+-------------+----------------+----------------+
|        3 |   0       |    0.211473 |       0.771408 |       0.406947 |
+----------+-----------+-------------+----------------+----------------+

- Method 0 (Correlation) :
Perfect : 1.0 (les histogrammes sont identiques)
Base-Half : 0.996665 (très similaire)
Base-Test(1) : 0.526326 (faible similarité)
Base-Test(2) : 0.977149 (très similaire)

- Method 1 (Chi-Square) :
Perfect : 0.0 (les histogrammes sont identiques)
Base-Half : 0.236579 (faible différence)
Base-Test(1) : 12488.5 (grande différence)
Base-Test(2) : 7.77917 (modérée différence)

- Method 2 (Intersection) :
Perfect : 1.57766 (les histogrammes sont identiques, valeur de référence pour l'intersection)
Base-Half : 1.41202 (grande intersection)
Base-Test(1) : 1.21467 (modérée intersection)
Base-Test(2) : 1.2483 (modérée intersection)

- Method 3 (Bhattacharyya) :
Perfect : 0.0 (les histogrammes sont identiques)
Base-Half : 0.211473 (faible différence)
Base-Test(1) : 0.771408 (modérée différence)
Base-Test(2) : 0.406947 (faible différence)

