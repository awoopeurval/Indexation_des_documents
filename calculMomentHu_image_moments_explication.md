Moments
Les moments sont des quantités qui fournissent des informations sur les propriétés géométriques d'une forme. Les moments d'ordre zéro, premier et second sont souvent utilisés pour calculer des propriétés telles que l'aire, le centre de masse, et l'orientation de la forme.

Area (M_00) : Le moment d'ordre zéro représente l'aire du contour.
Area OpenCV : C'est la valeur de l'aire calculée par la fonction cv.contourArea, qui doit correspondre à M_00.
Length : La longueur du périmètre du contour, calculée par la fonction cv.arcLength.

Moments de Hu
Les moments de Hu sont des moments invariants aux transformations (translation, rotation et mise à l'échelle) utilisés pour la reconnaissance de formes.


## IMPORTANTCE DES MOMENTS DE HU
- Invariance et Reconnaissance de Formes
Invariance aux Transformations Géométriques :

Translation : Les moments de Hu sont invariants à la translation, ce qui signifie que la position de la forme dans l'image n'affecte pas les valeurs des moments.
Rotation : Ils sont également invariants à la rotation. Une forme qui est tournée dans l'image aura les mêmes moments de Hu que la forme originale.
Mise à l'Échelle : Les moments de Hu sont invariants à la mise à l'échelle, ce qui permet de reconnaître des formes même si elles apparaissent à des tailles différentes.

- Identification et Classification :

Signature Unique : Chaque forme a une signature unique en termes de moments de Hu. Cela permet de distinguer différentes formes dans une image.
Comparaison de Formes : En utilisant les moments de Hu, on peut comparer des formes pour voir si elles sont similaires ou différentes, même en présence de transformations géométriques.
Reconnaissance de Modèles : Dans les applications de reconnaissance de modèles, les moments de Hu sont utilisés pour identifier des objets ou des formes spécifiques dans les images.