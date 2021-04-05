############################################## Tacotron 2 main script ##################################################

# Import libraries
import numpy as np
from scipy.io.wavfile import write
import torch

def gen_tacotron2(text_filepath):
    """ Function: Generate speech from 1 input transcript (collection of sentences) using Tacotron 2.
        Input:    Transcript file path (string) with 8 digit ID
        Output:   Returns nothing but saves audio file (.wav) in folder 'gen_tacotron2' """

    # Load pretrained Tacotron 2, transfer computation to GPU and set model in test mode
    tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
    tacotron2 = tacotron2.to('cuda')
    tacotron2.eval()

    # Load pretrained WaveGlow, transfer computation to GPU and set model in test mode
    waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to('cuda')
    waveglow.eval()

    # Load text from file
    text = open(text_filepath, "r")   #TODO: Update to correct path
    text = text.readline()

    # Preprocessing text
    sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)

    # Generate speech using models
    with torch.no_grad():
        _, mel, _, _ = tacotron2.infer(sequence)
        audio = waveglow.infer(mel)
    audio_numpy = audio[0].data.cpu().numpy()
    rate = 22050

    # Save file with correct 8 digit ID
    file_id = text_filepath[-8:]
    output_filepath = "gen_tacotron/" + file_id + ".wav"
    write(output_filepath, rate, audio_numpy)

if __name__ == '__main__':
    """ Program description: Generate speech using Tacotron 2. Tacotron 2 is pretrained on the entire LJSpeech dataset
                             (see more at: https://github.com/NVIDIA/tacotron2). The speech is generated from a training
                             split of the LibriSpeech dataset. Next, we train an ASR model on the generated dataset (see
                             more at: https://github.com/borgholt/asr). The performance of the ASR model is evaluated on 
                             a test split of the LibriSpeech dataset. 
                             
                             In result of this program is (1) a synthesized speech dataset, (2) a parameter save of the 
                             ASR model and (3) output of best ASR WER through stdout stream. """

    """ Part 1: Define dataset path-files """   # Todo: Update paths below
    train_paths = ""           # LibriSpeech train (for generating)
    gen_paths = ""             # Generated data
    test_paths = ""            # LibriSpeech test  (for validating ASR)

    """ Part 2: Generate speech using Tacotron 2 """
    # Get file with all transcript paths and call the generator routine for each transcript one by one
    transcript_paths = open(train_paths, "r")  # TODO: Update to correct path
    done = False

    while not done:
        # Read 1 transcript and stop if transcript is empty (all transcripts have been read)
        transcript_path = transcript_paths.readline()
        if transcript_path == '': break

        # Generate speech and save
        gen_tacotron2(transcript_path)

    """ Part 3: Train and test ASR model """
    best_wer = train_ASR(gen_paths, test_paths)
    print("The best WER: {}".format(best_wer))