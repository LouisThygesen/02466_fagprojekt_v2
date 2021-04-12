#################################### Script to synthezise from TTS model ###############################################
########################################################################################################################

import numpy as np
from tqdm import tqdm
import torchaudio
import torch
import torch.nn.functional as F

def synthesize(transcript, size, model, sentence_embedding, tokenizer, device, temperature=1.0):
    """ Function: Generate speech from transcripts using original WaveNet paper (2016) implementation
        Input:
        Output:
    """

    # Tokenize the transcript with the BERT tokenizer
    tokens = tokenizer(transcript, return_attention_mask=False, return_token_type_ids=False,return_tensors='pt')['input_ids'].to(device)

    # Feed into sentence embedding class
    gc_embed, lc_embed = sentence_embedding(tokens)

    #Interpolate the locally conditioned signal from BERT so it fits with the waveform size and then trim the same portion of the signal as for the waveform.
    lc_embed = F.interpolate(lc_embed, size=size)
    lc_embed = F.pad(lc_embed, (model.receptive_field,0))

    model.eval()

    rec_fld = model.receptive_field + 1

    T = size

    generated = np.ones((rec_fld+T))*MuLawEncoding(torch.tensor(0.0)).item()

    with tqdm(range(T)) as t_bar, torch.no_grad():
        for n in t_bar:
            input = torch.tensor(generated[n:rec_fld+n], device=device).long().unsqueeze(0)
            predictions = model(input, lc=lc_embed[:,:,n:rec_fld+n], gc=gc_embed)
            predictions = torch.softmax(predictions/temperature, dim=1)
            max_index = torch.multinomial(predictions[0, :, 0], 1).squeeze()
            generated[rec_fld+n] = max_index.item()
    generated = generated[rec_fld:]

    return generated