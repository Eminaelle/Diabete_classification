# Journal de bord - Projet de classification du diabète
## Étape 1 : Exploration des données et identification des valeurs problématiques

Exploration des données :

    J'ai exploré les premières lignes du dataset avec df.head() et utilisé df.describe() pour obtenir des statistiques descriptives des colonnes.
    J'ai observé que certaines colonnes (Glucose, BloodPressure, BMI, SkinThickness, Insulin) contiennent des valeurs égales à 0, qui sont médicalement improbables et probablement des données manquantes. Par exemple, une valeur de 0 pour le taux de glucose est impossible chez une personne vivante.

Décision initiale :

    Pas d'imputation des valeurs nulles pour cette première série de tests. J'ai décidé de laisser les valeurs 0 telles quelles pour tester les modèles sans modification.

## Étape 2 : Prétraitement des données

Séparation des features et de la variable cible :

    J'ai séparé les features (X) de la variable cible (y), où y représente l'issue diabétique (1 = diabétique, 0 = non diabétique).

python

    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

Standardisation :

    J'ai standardisé les données à l'aide de StandardScaler pour uniformiser l'échelle des différentes variables et ainsi faciliter l'entraînement des modèles comme la régression logistique.

python

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

## Étape 3 : Test des modèles sans imputation des valeurs nulles

### Modèle 1 : Régression logistique :
    J'ai commencé par entraîner un modèle de régression logistique sur les données standardisées et sans imputation.
    J'ai évalué la performance du modèle en utilisant la validation croisée à 5 folds pour obtenir des résultats robustes.

python

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    model = LogisticRegression()
    scores = cross_val_score(model, X_scaled, y, cv=5)
    print(f'Accuracy moyenne : {scores.mean()}')

Résultats : L'accuracy moyenne obtenue est d'environ 0.74, et le modèle s'est bien comporté malgré la présence de valeurs nulles (0).

### Modèle 2 : K-Nearest Neighbors (KNN) :
    J'ai ensuite testé un modèle KNN, en utilisant GridSearchCV pour optimiser le paramètre n_neighbors.

python

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV

    param_grid = {'n_neighbors': [3, 5, 7, 9]}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X_scaled, y)

    print(f'Best n_neighbors: {grid_search.best_params_}')

Résultats : Le modèle KNN a obtenu une accuracy inférieure à la régression logistique, avec des résultats légèrement en dessous, et une précision de 0.62 et un rappel de 0.68.

### Modèle 3 : Random Forest :
    J'ai aussi testé une forêt aléatoire en optimisant plusieurs hyperparamètres (nombre d'arbres et profondeur maximale) à l'aide de GridSearchCV.

python

    from sklearn.ensemble import RandomForestClassifier
    param_grid_rf = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None]}

    rf_grid_search = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5)
    rf_grid_search.fit(X_scaled, y)

    print(f'Best params for Random Forest: {rf_grid_search.best_params_}')

Résultats : Le modèle Random Forest a donné une performance inférieure à la régression logistique, mais supérieure à KNN.

## Étape 4 : Test de différentes stratégies d'imputation

Imputation par la médiane :
    Après avoir observé que les valeurs nulles (égales à 0) pouvaient potentiellement poser problème, j'ai essayé d'imputer les valeurs 0 dans les colonnes Glucose, BloodPressure, BMI, SkinThickness, et Insulin avec la médiane.

python

    cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_replace:
        df[col].replace(0, df[col].median(), inplace=True)

Résultats après imputation :
    J'ai observé une légère baisse de performance avec la régression logistique et les autres modèles (Random Forest et KNN). Les résultats étaient systématiquement moins bons que ceux obtenus sans imputation.

Imputation par régression :
    J'ai également tenté d'utiliser un modèle de régression pour imputer certaines valeurs manquantes, comme SkinThickness et Insulin, en fonction des autres variables comme BMI et Glucose.

python

    from sklearn.linear_model import LinearRegression

    # Exemple pour SkinThickness
    X_train_reg = df[df['SkinThickness'] != 0][['BMI', 'Insulin']]
    y_train_reg = df[df['SkinThickness'] != 0]['SkinThickness']

    model_reg = LinearRegression()
    model_reg.fit(X_train_reg, y_train_reg)

    X_test_reg = df[df['SkinThickness'] == 0][['BMI', 'Insulin']]
    df.loc[df['SkinThickness'] == 0, 'SkinThickness'] = model_reg.predict(X_test_reg)

Résultats après imputation par régression :
    Les performances ont continué de diminuer, surtout avec le modèle KNN, mais aussi avec RandomForest et la régression logistique. L’imputation par régression a probablement introduit du bruit.