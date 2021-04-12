############################################## Tacotron 2 main script ##################################################

# Import libraries
import numpy as np
from scipy.io.wavfile import write
import torch
from model_ASR_train import train_ASR

def gen_tacotron2(path_ID):
    """ Function: Generate speech from 1 input transcript (collection of sentences) using Tacotron 2.
        Input:    Transcript file with ID
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
    text_path = "" + path_ID + ".txt"
    text = open(text_path, "r")
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

    # Save file with correct ID (same as training ID)
    file_id = text_path.split("/")[-1]
    output_filepath = "../../Data/gen_tacotron/" + file_id + ".wav"
    write(output_filepath, rate, audio_numpy)

if __name__ == '__main__':
    """ Program description: Generate speech using Tacotron 2. Tacotron 2 is pretrained on the entire LJSpeech dataset
                             (see more at: https://github.com/NVIDIA/tacotron2). The speech is generated from a training
                             split of the LibriSpeech dataset. Next, we train an ASR model on the generated dataset (see
                             more at: https://github.com/borgholt/asr). The performance of the ASR model is evaluated on 
                             a test split of the LibriSpeech dataset. 
                             
                             In result of this program is (1) a synthesized speech dataset, (2) a parameter save of the 
                             ASR model and (3) output of best ASR WER through stdout stream. """

    """ Part 1: Define dataset file-IDs """
    train_IDs = "../../Data/LibriSpeech/train.txt"           # LibriSpeech train (for generating)
    test_IDs = "../../Data/LibriSpeech/evaluation.txt"       # LibriSpeech test  (for validating ASR)

    """ Part 2: Generate speech using Tacotron 2 """
    # Get all transcript IDs and call the generator routine for each transcript one by one
    transcript_IDs = open(train_IDs, "r")
    done = False

    while not done:
        # Read 1 transcript ID and stop if transcript is empty (all transcripts have been read)
        transcript_ID = transcript_IDs.readline()
        if transcript_ID == '': break

        # Store transcript path (represents both .wav and .txt)
        transcript_path = "../../Data/LibriSpeech/data_files/" + transcript_ID

        # Generate speech and save
        gen_tacotron2(transcript_path)

    """ Part 3: Train and test ASR model """
    # Pass generated data (same IDs as training data) and test data
    train_ASR(train_IDs, test_IDs)
