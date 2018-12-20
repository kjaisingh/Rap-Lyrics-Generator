# Rap Lyrics Generator
An LSTM implementation for a Rap Lyric Generator that spawns rap lyrics based on a Kaggle dataset with over 38,000 lines.


In order to run the Generator, the following must be done after getting into the Generator's working directory:

1. Unzip the dataset through the two following commands:
```
sudo apt-get install unzip
```
```
unzip lyrics.csv.zip
```

2. Train the LSTM network through the following command:
```
python train.py
```

3. Generate new lyrics through the following command:
```
python run.py -s <seed>
```

The 'seed' parser argument can be replaced for anything, such as a word, phrase or line. It is simply the starting point of generating the text.


The features of the Rap Lyrics Generator include:
* Ability to create lyrics based on a seed
* Generates 1000 character lyric outputs
* Trains efficiently
* Utilizes rap lyrics from various artists, such as Kanye West and Kendrick Lamar, to learn


The Rap Lyrics Generator requires the following to be installed on the system:
* Python 3
* Tensorflow
* Keras
* Numpy
* Pandas
