# Long Short-Term Memory (LSTM) Model Implementation for Machine Translation (MT)
### _Machine Translation (MT) system to carry out translations from Cape Verdean Creole to English and vice versa_

The model was proposed by [Hochreiter and Schmidhuber (1997)] and can be considered one of the first big steps in Natural Language Processing (NLP) tasks such as MT.

> This project was made with the intention of being an integral part of my final final project.
> All aspects addressed in the implementation were made according to the needs of the project as a whole.
> Cape Verdean Creole is the mother language of Cape Verde, which is not an official language and is not well represented and known around the world. Therefore, it is a great honor to carry out studies and projects that contribute to its recognition and dissemination.


### Features

This implementation provides some features that can be accessed by commands in the terminal. Some of the features are:

- Train and test the translation models;
- Calculate the evaluation metrics (BLUE, METEOR and TER);
- Calculate the model parameter numbers;
- Test of translation by the terminals;
- Confusion confusion matrix generator.



### Dependencies

To run this project there are some python dependencies/libraries that need to be installed.
To perform this installation run:
```sh
pip install -r requirements.txt
```


### Execution
To execute this project you should pick one the commands according te task you want to make. The commads are:
- **console** - to open a test simulator for MT in the terminal;
- **train** - to train a model;
- **test_model** - to open a test console of a few sentences of the test data;
- **blue_score** - to calculate the BLUE score according to the test data;
- **meteor_score** - to calculate the METEOR score according to the test data;
- **ter_score** - to calculate the BLUE score according to the test data;
- **confusion_matrix** - to generate a confusion matrix according to the given sentence;
- **count_parameters** - to count the number of model parameters.

Then it is necessary to specify the source and target language, for example, being English (**en**) as source and Cape Verdean Creole (**cv**) as target language, the complete command to execute a training would be:
```sh
python main.py -a train -s en -t cv
```
**Notes** that the parameters have the following meanings:
- **'-a'** or **'--action'** is the action to be performed;
- **'-s'** or **'--source'** is the source language;
- **'-t'** or **'--target'** is the target language;


### All parts of the project into a whole
The whole project is divided into parts and each part has an essential function in it.
They are distributed as shown in the subtopics below.


#### Models implementation
This are the model used in the whole project:

- [Transformer model implementation]
- [GRU model implementation]
- [LSTM model implementation]
- [Models Training Graphs Generator]


#### Frontend test platform
This is a React App made to test all the translations made by the models, similar to the App [Google Translator]. 
Projects related to using the frontend application can be found at:

- [MT Models API implementation]
- [Cape Verdean Creole Translator Frontend test App]


#### Dataset
The dataset used to train, validate and test the model was the [CrioleSet dataset].
If the dataset is not in the project while executing any of the action commands, it will be downloaded and added to the project.

- [CrioleSet dataset]


### Projects References
For the implementations of this project, these are some of the resources available on Github that were based on:

- https://github.com/fastai/course-nlp
- https://github.com/NavneetNandan/MachineTranslationEvaluation
- https://github.com/lucidrains/mlm-pytorch
- https://github.com/LiyuanLucasLiu/Transformer-Clinic
- https://github.com/jadore801120/attention-is-all-you-need-pytorch
- https://github.com/tunz/transformer-pytorch
- https://github.com/lmthang/nmt.hybrid


### License

MIT


**Feel free to use and get in touch with any questions.**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [Hochreiter e Schmidhuber (1997)]: <https://doi.org/10.1162/neco.1997.9.8.1735>
   [Transformer model implementation]: <https://github.com/robertocarlosmedina/attention-transformer-translator>
   [GRU model implementation]: <https://github.com/robertocarlosmedina/rnn-gru-attention-translator>
   [LSTM model implementation]: <https://github.com/robertocarlosmedina/rnn-lstm-translator>
   [MT Models API implementation]: <https://github.com/robertocarlosmedina/machine-translation-models-api>
   [CrioleSet dataset]: <https://github.com/robertocarlosmedina/crioleSet>
   [Cape Verdean Creole Translator Frontend test App]: <https://github.com/robertocarlosmedina/cv-creole-translator>
   [Models Training Graphs Generator]: <https://github.com/robertocarlosmedina/models-graphs-generator>
   [Google Translator]: <https://translate.google.com>
