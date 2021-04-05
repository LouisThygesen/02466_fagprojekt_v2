############################################ Baseline model main script ################################################

if __name__ == '__main__':
    """ Program description: Test ASR model on a test split of the LibriSpeech dataset. The model is pretrained on a 
                             training split of the LibriSpeech dataset (see more: https://github.com/borgholt/asr). The 
                             model functions as a baseline for performance comparisons with a WaveNet model and a 
                             Tacotron 2 model. 
                             
                             The result of this program is an output of best ASR WER through stdout stream. """

    """ Part 1: Define dataset path-files """
    test_paths = ""   # LibriSpeech test  (for validating ASR)

    """ Part 3: Train and test ASR model """
    best_wer = pretrained_ASR(test_paths)
    print("The best WER: {}".format(best_wer))
