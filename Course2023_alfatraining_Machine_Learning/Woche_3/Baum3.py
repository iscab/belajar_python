#Aufgaben:



import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree,export_graphviz
from sklearn.model_selection import train_test_split
#train_test_split ist in diesem Beispiel nicht nötig,
# weil es nur auf die Darstellung ankommt.
# Um es einfach zu halten, werden die gesamten Iris-DAten zur Konstruktion des Baumes benutzt.
#Für die Voraussage werden dann die restlichen Punkte des gesamten Koordinatensystems benutzt
#######################################################################
########################################################################
#  der komplette  Baum wird gezeichnet als  Grafik

# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02

# Load data
iris = load_iris()

plt.figure()
clf = DecisionTreeClassifier().fit(iris.data, iris.target)
plot_tree(clf, filled=True)
#plot_tree wurde importiert, wie gut, dass wir den nicht selbst schreiben müssen
plt.show()
# exit()

##########################################################################
##########################################################################
######## Es werden immer nur 2 Spalten benutzt und damit jeweils ein Decisiontree gefüttert.
######## Damit ergeben sich 6 Teilgrafiken


# Hier werde Paare aus Spalten ausgewählt, die nacheinander vom Classifier verarbeitet werden sollen.
#Dabei entstehen so viele Plots , wie es Spaltenpaare gibt
#Beispiel: Spalte 0 steht für Sepal_length
#Die for-Schleife baut für jede Spaltenkombination eine eigene Instanz des Classifiers auf
 #       und plottet das Ergebnis

plt.figure()
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):


    # We only take the two corresponding features
    # Pair bezeichnet die Indizes der Spalten, die aus der gesamten Irismenge verwendet werden
    X = iris.data[:, pair]
    y = iris.target

    # Train
    # Das wichtigste Ergebnis von dem fit ist , dass der Baum jetzt aufgebaut ist.
    # Er wird unter dem Attribut clf.tree_ abgespeichert und hinterher für Voraussagen benutzt

    clf = DecisionTreeClassifier().fit(X, y)

    # Plot the decision boundary
    # insgesamt 6 Spaltenkombinationen führen zu 6 Grafiken,
    # die in 2 Zeilen a 3 Grafiken dargestellt werden.
    #pairidx+1 gibt die Nummer der Grafik an
    plt.subplot(2, 3, pairidx + 1)


    # Für die Zeichnumg sind wir etwas großzügiger mit den Grenzen
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1


    # Meshgrid wird eingerichtet, weil man nicht nur die Originaldatenpunkte mit ihren Voraussagen plotten will,
    # sondern auch jeden weitern Punkt einer Klasse zuordnen will
    # das Meshgrid ist aber diskret, für die Zwischenräume müssen noch Lösungen gefunden werden
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    # Für jeden Punkt(xx,yy), der im Meshgrid definiert ist, werden Features und Daten
    # hintereinandergeschrieben und vorausgesagt: Schleife in der for-Schleife
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Die Klassenvoraussage wird 2-dimensional, ordnet sich gemäß x und y an
    Z = Z.reshape(xx.shape)
    i=1
    # Färbe nicht nur die meshgrid-Punkte, sondern fülle auch die Zwischenräume
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)


    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])

    # Plot the training points
    #Füge nachträglich die Punkte der Ausgangsdaten hinzu.
    #Für jede Klasse wird eine andere Farbe gewählt.
    for i, color in zip(range(n_classes), plot_colors):
        #i ist eine Klasse, color eine Farbe aus plot_colors,
        # beide werden im Reißverschlussverfahren in einem Tupel zusammengeführt.
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
# plt.show()
# exit()

#######################################################################
#######################################################################
#noch einmal ein Plot für die Entscheidung mit sämtlichen Spalten
X = iris.data
y = iris.target

clf = DecisionTreeClassifier().fit(X, y)
importance_liste=clf.feature_importances_
i_features=np.argsort(importance_liste)
i_wichtige_features=i_features[-2:]
i_unwichtige_features=i_features[:2]

# Zeichnungsgrenzen
#Es werden nur Spalte 3 und 4 geplottet
x2_min, x2_max = X[:, i_wichtige_features[0]].min() - 1, X[:, 2].max() + 1
x3_min, x3_max = X[:, 3].min() - 1, X[:, i_wichtige_features[1]].max() + 1

# Meshgrid
xx2, xx3 = np.meshgrid(np.arange(x2_min, x2_max, plot_step),
                       np.arange(x3_min, x3_max, plot_step))
xx2_geglättet=xx2.ravel()
xx3_geglättet=xx3.ravel()
anzahl_xxx2=xx2_geglättet.shape[0]
xxx0=np.repeat(X[:,i_unwichtige_features[0]].mean(),anzahl_xxx2)
xxx1=np.repeat(X[:,i_unwichtige_features[1]].mean(),anzahl_xxx2)
zusammen=np.c_[xxx0,xxx1,xx2_geglättet, xx3_geglättet]

Z = clf.predict(zusammen)

# Die Klassenvoraussage wird 2-dimensional, ordnet sich gemäß x und y an
Z = Z.reshape(xx2.shape)

plt.figure()
# Färbe nicht nur die meshgrid-Punkte, sondern fülle auch die Zwischenräume
cs = plt.contourf(xx2, xx3, Z, cmap=plt.cm.RdYlBu)


Z = Z.reshape(xx2.shape)


# Färbe nicht nur die meshgrid-Punkte, sondern fülle auch die Zwischenräume
cs = plt.contourf(xx2, xx3, Z, cmap=plt.cm.RdYlBu)


plt.xlabel(iris.feature_names[i_wichtige_features[0]])
plt.ylabel(iris.feature_names[i_wichtige_features[1]])
# plt.show()
# exit()


#######################
#Für Mac-User und andere, denen die obige Grafik "verrutscht" ist
"""
# Export der Baumgraphik in eine besondere Form und Abspeichern in einer Datei.
#Die Datei kann man mit einem Editor lesen,wenn man möchte
"""
export_graphviz(clf,out_file="BaumBeschreibung.dot")
#besser
#export_graphviz(clf,out_file="baum.dot",feature_names=iris.feature_names,filled=True,rounded=True)
plt.figure()
plot_tree(clf, filled=True)
plt.show()
"""
Die Datei kann danach mit Hilfe des Programms dot in ein normales Bild umgewandelt werden.
Hinweise zur Installation
Falls anaconda installiert ist, kann man das Paket python-graphviz über den Paketmanager conda nachinstallieren. 
Das geht entweder graphisch. 
Oder auf Kommandozeile durch den Befehl
        conda install python-graphviz 
Für die Installation auf dem Mac kann man auch den Paketmanager Homebrew benutzen.
pathon-graphviz lässt sich auch mittels Pycharm installieren.
In allen Fällen muss man darauf achten, dass das richtige Environment aktiviert ist. 

Das Paket enthält das Programm dot, welches Beschreibungen in Bilder umwandelt.
Um das Programm aufzrufen braucht man ein Terminal.
Man kann das Programm dot eventuell nur in den richtigen Environment finden.
Die richtigen Optionen findet man mit dot --help

Wir brauchen so etwas wie 
        dot -T png BaumBeschreibung.dot -o BaumBild.png
oder
        dot -T jpg BaumBeschreibung.dot -o BaumBild.jpg


"""