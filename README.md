Questa Repository include:
- train.py: script Python usato per addestrare la rete neurale
- data_importer.py: script usato per creare il dataset. Salva l'immagine che vede il robot e genera anche la relativa maschera (colora di giallo le linee discontinue, bianco per le linee continue, nero background)
- data_importer_binary_class.py: script usato in test precedenti. Salva l'immagine che vede il robot e genera una maschera binaria (bianco per le linee, nero background)

Link download dataset: https://www.kaggle.com/datasets/danielefiumicelli/turtebot-dataset
Il dataset contiene:
- test_images: cartella contenente immagini usate per il training della rete neurale (nome cartella fuorviante)
- mask_images_auto: cartella contenente le maschere usate per la versione ultimata della rete neurale multi class
- mask_images: cartella contenente le maschere usate per una versione primordiale della rete neurale binary class
