# code_for_raspberry_pi
Code using a model converted in tensorflow_lite. It aims to recognize birds on images, giving bounding boxes and labels.

#How to use it

1) L'étape du code:

-cloner le github suivant: https://github.com/lucien92/code_for_raspberry_pi

-dans le fichier tflite_converter. py rentrer les paramètres de son modèle afin d'obtenir un fichier .tflite uqi est notre modèle passé en version tensorflow lite.

-Ensuite nous passons aux prédictions. Tous le dossier tf_lite_path_rpi ne contient que trois packages: numpy,tflite_runtime (pour l'interpreter, une classe spécialement faite pour les rpi), pandas et opencv. En effet, la rpi ne peut pas tolérer des packages aussi gros que tensorflow et éviter d'utiliser les gros packages permet aussi de faire des prédictions rapidemment et en grand nombre.
Dans le fichier cité ci-dessus se trouve des images à tester, notre modèle converti en tflite et deux fichiers python. Le premier fichier python se nomme utils_lite.py. Il contient seulement des outils qui vont nous permettre d'extraire les informations de prédiction (fonction netout) à partir des derniers tenseurs du réseau de neurones.
On trouve aussi le fichier predict_tflite.py. Dans ce fichier il suffit de rentrer le nom de l'image à prédire, le nom du modèle et de run afin d'obtenir une images labellisée (en 2,5 frames par sec).

---> après avoir compris et cloner tout ça essayez dans un premier temps de lancer cela sur votre ordinateur pour vérifier que tout marche. Puis changer les chemins en les adaptant à ceux de la raspberry pi et déposer cela sur un github (cela permet de transmettre facilement les scripts à la rpi grâce à internet, la connexion ssh pouvant être une galère (-->n7)).

2) La mise en place de la raspberry pi

Installation: connectez la rpi à un point de recharge afin de l'allumer puis branchez la à un écran grâce à un cable hdmi afin de pouvoir bénéficier de l'interface graphique proposée par la rpi.

-télécharger python sur la rpi 

-upgrader tous les pip etc

-installation via le terminale de 4 modules: opencv, numpy, pandas et tflite_runtime. 

-lancer un git clone de votre repository précédemment créé (comme conseillé plus haut)

-pour lancer des predict: lancer la commande python3 "nom du fichier qui fait la predict"

Vous pouvez maintenant lancer une prédiction de photo avec votre rpi!

PS: vous la detection real-time et l'écriture dans le csv, rajoutez un nouveau script...
