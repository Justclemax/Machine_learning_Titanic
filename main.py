from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
# ON charge notre fichier
titanic_df = pd.read_csv("titanic-passangers.csv", sep=",")
titanic_df.head(5)
# DESCRIPTION DE NOTRE DATA FRAME
data = titanic_df.describe()
data

#
# LES INFORMATIONS SUR NOS DONNEES

df = titanic_df.info()
(df)
# CHERCHONS LES DONNEES MANQUNANTES
titanic_df.isnull().sum()
# On a peut constante qui'il manque beaucoup de donnnees pour la colone age nous calculons le pourcentage de donnnees maquantes
age_manquants = data.loc['count']["Survived"]-data.loc['count']["Age"]
pourcentage_de_donnesDM = (age_manquants*100)/data.loc['count']['Survived']

print("Il manque", pourcentage_de_donnesDM, "%", "de donnees")


# PHASE DE NETTOYADGE
# Nettoyage de donnnees
# on remplace les donnes manquantes par la moyenne des ages

titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())


# On supprime les donnees manquantes car on a seulement deux
# donnees qui manques

titanic_df.dropna(axis=0, subset=['Embarked'], inplace=True)


# ici 0n cherche seulement titre de chaque personne
titanic_df['Survived'] = titanic_df['Survived'].replace({1: "Yes", 0: "No"})

# oN cherche seulement les titres de chaque personne a bord du bateau
titanic_df['Titre'] = titanic_df['Name'].str.split(
    ',', expand=True)[1].str.split('.', expand=True)[0]


titanic_df['Titre'].value_counts()

# On regroupe  les titre et les sex
pd.crosstab(titanic_df['Titre'], titanic_df['Sex'])
titanic_df.isnull().sum().iloc[10::10]
# on reste avec beaucoup de donnees maquantes pour la colonne cabin
# Puis qu'il manque beaucoup de donnees concerant la cabin on decide de ramplace les valeurs maquantes par les valeurs booleen
titanic_df["CabinBool"] = (titanic_df["Cabin"].notnull().astype("int"))
titanic_df.head()

# ON supprime les colonnes Name et Cabin on peut trouver
titanic_df.dropna()

new_df = titanic_df.drop(columns=['Name', 'Cabin'])
new_df.head(5)

# VERIFICATION DE DONNEES MANQUNANTES
new_df.isnull().sum()

new_df.describe()
# NOS DONNEES SONT PRETES

# PHASE DE VISUALISATION
# la fonction de survivant en fonction  de != classe et age

sns.catplot(y="Age", x="Sex", data=new_df, hue="Survived")
print("Pour notre presention il y'a plus d'hommes qui ont dans le titanic")
print("Voici une image plus précise de la distubution  de l’âge des passagers sur le Titanc:")
new_df['Age'].hist(bins=70, color='indianred', alpha=0.9)


def maj(passager):
    age, sex = passager
    # On compare les ages avec les differentes tenages d'ages
    if (age < 12):

        return "Enfant"
    elif (age >= 12 and age < 18):
        return 'Ado'
    else:
        return sex


new_df['Person'] = new_df[['Age', 'Sex']].apply(maj, axis=1)

print("Comme on peut le voir, il y en avait sur le Titanic :",

      new_df.Person.value_counts().iloc[0], "hommes ,",
      new_df.Person.value_counts().iloc[1], " femmes ,",
      new_df.Person.value_counts().iloc[2], "Enfants et",
      new_df.Person.value_counts().iloc[3], "Ado")
new_df['Person'].hist(bins=70, color='blue', alpha=0.9)
df = new_df


def plot_correlation_map(df):

    corr = df.corr()

    s, ax = plt.subplots(figsize=(12, 10))

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    s = sns.heatmap(

        corr,

        cmap=cmap,

        square=True,

        cbar_kws={'shrink': .9},

        ax=ax,

        annot=True,

        annot_kws={'fontsize': 12}

    )


plot_correlation_map(df)


1  # . Appliquez la régression logistique.
new_df['Sex'] = new_df['Sex'].replace({"female": 0, "male": 1})
# features extraction
x = new_df[['Age', 'Pclass', 'Sex']]
y = new_df['Survived']  # le facture de sortir
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0)  # splitting data with test size of 25%


logreg = LogisticRegression()  # build our logistic model
logreg.fit(x_train, y_train)
# fitting training data
y_pred = logreg.predict(x_test)
logreg.score(x, y)*100


# 2. Utilisez une matrice de confusion pour valider votre modèle.
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=[
                               'Actuellement'], colnames=['Prediction'])
sns.heatmap(confusion_matrix, annot=True)

# Confusion Matrix
# 3. Une autre matrice de validation pour la classification est ROC/AUC. Faites vos recherches sur eux, expliquez-les et appliquez-les dans notre cas.

x = new_df['Sex']
y = new_df['Survived']
# train models
# logistic regression
model1 = LogisticRegression()
# knn
model2 = KNeighborsClassifier(n_neighbors=4)

# fit model
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)

# On predit les probabilites de modele
pred_prob1 = model1.predict_proba(x_test)
pred_prob2 = model2.predict_proba(x_test)

# on rece roc curve
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:, 1], pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:, 1], pos_label=1)

# roc curve for tpr = fpr
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
# Création de taux de faux positifs et de vrais positifs et impression des scores

# auc scores
auc_score1 = roc_auc_score(y_test, pred_prob1[:, 1])
auc_score2 = roc_auc_score(y_test, pred_prob2[:, 1])

print(auc_score1, auc_score2)


# Pour la  matrice de validation pour la classification est ROC/AUC
# Il s'agit ici de comparer la performance des models differents
