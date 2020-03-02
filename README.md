# biofilm_recognizer_cnn

Breve introduzione: 
Il sistema realizzato è in grado di riconoscere in maniera automatica la presenza di biofilm in scansioni di rinocitologia, attraverso un sistema basato su rete neurale convoluzionale.

Librerie: 
Per implementare il sistema sono state utilizzate le seguenti librerie:
●	os: libreria contenente metodi per sfruttare le funzionalità specifiche di un sistema operativo. È stata utilizzata per verificare l’esistenza di un percorso, effettuare il join tra path, per iterare sulle immagini e per conoscere il percorso della directory contenente le immagini da analizzare;
●	cv2: libreria contenente funzionalità utili per risolvere i problemi di computer vision. Fornisce sia algoritmi classici che algoritmi aggiornati allo stato dell’arte. È stata impiegata per leggere, ridimensionare e convertire le immagini in differenti modelli di colore;
●	skimage: libreria contenente una raccolta di algoritmi per l'elaborazione delle immagini e per la computer vision. In particolare è stato importato il subpackage io utilizzato per salvare le immagini;
●	numpy: in esteso Numerical Python, è una libreria che fornisce un rapido calcolo matematico su array e matrici. Impiegata per trasformare strutture dati in array, per salvare dati di train e test e per confrontare valori di array;
●	random: libreria utile per generare numeri in maniera casuale, in particolare è stato importato il metodo shuffle per mescolare le immagini all’interno del dataset;
●	matplotlib: libreria per la creazione di grafici;
●	tflearn: libreria di Deep Learning modulare e trasparente costruita su Tensorflow. È stata utilizzata per implementare e gestire la CNN;
●	scikit-learn: libreria che mette a disposizione l’implementazione di molte delle più note tecniche di Machine Learning per l’analisi dei dati. È stata utilizzata per effettuare operazioni sul dataset come train_test_split, k-fold Cross Validation, generare report per la classificazione e creare la matrice di confusione;

Implementazione: 
Il sistema è basato sui seguenti metodi:
●	create_data(): in cui ciascuna immagine viene convertita in grayscale e ne viene effettuato il resize. Tale metodo restituisce due array, uno contenente le immagini e uno contenente i label. Il label riportato coincide con il nome della cartella (Biofilm o Other) contenente l’immagine;
●	split_data(): metodo che consente di suddividere il dataset in x_train, x_test, y_train e y_test;
●	conf_matrix(): in cui viene calcolata una matrice di confusione con lo scopo di verificare quanti tiles sono stati classificati in maniera errata. Basato sul set di test, tale grafico presenta sulle asse delle ascisse i valori predetti e sull’asse delle ordinate i valori reali;
●	create_net(): in tale metodo viene creata la rete neurale convoluzionale descritta in precedenza;
●	evaluate_classifier(): metodo che permette di valutare il modello addestrato, in termini di accuratezza, sensibilità e miss rate in base al test set;
●	fit_model(): metodo in cui viene effettuato l’addestramento della rete, viene salvato il miglior modello, ottenuto dopo l’applicazione della k-fold Cross Validation, e viene calcolata l’accuratezza dell’intero sistema. Mentre, se il training è stato effettuato in precedenza, viene caricato esclusivamente il miglior modello;
●	predict(): in cui viene effettuata la predizione per tutte le immagini contenute nella cartella Unlabeled. Questo metodo effettua una stampa di tutte le predizioni effettuate;

