# DeepLearningFinal


To Train this model, it is necessary to have a directory of .wav files called "Wav_Files" and a directory and .xml files called "Xml_files" in a directory'Jazz_Standards_Data/Wav_Files'. Or, it is necessary to change these paths in the code.

It will take quite some time for the model to parse all of the data, and then for the training of the model. 

After the model is trained, it can take any spectrogram as input. (it will need to be stored in array-form like the training data was). The spectrogram can be produced from any .wav file, it should be a jazz harmony .wav file for best results.  The output will be a matrix of likelihoods, so an argmax function will be necessary to determine the index of the result.

Because all chord qualities are encoded as integers, the output of the argmax function will need to be decoded using the Chord_to_IDX dictionary

Unfortunately, I was not able to upload the data to github, but please contact me for testing data if you want to test the model.
