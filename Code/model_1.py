######################################### TTS model Implementation #####################################################
########################################################################################################################

import torch
import torch.nn.functional as F
import torch.nn as nn

class WaveNet(nn.Module):
    """ Description: Main model class for model based on original WaveNet paper (2016) """

    def __init__(self, quantization_bins=256, channels=32, kernel_size=2, dilation_depth=5, blocks=1, condition_size=512,
                 global_condition=False, local_condition=False, hugging_face_model='bert-base-uncased'):
        # Inherent from Pytorch parent class
        super(WaveNet, self).__init__()

        """ Part 1: Define model information and parameters """""
        self.C = channels
        self.kernel_size = kernel_size
        self.bins = quantization_bins
        self.dilations = [2 ** i for i in range(dilation_depth)] * blocks
        self.receptive_field = sum([(self.kernel_size-1)*2 ** i for i in range(dilation_depth)] * blocks)

        """ Part 2: Get transcript embedding """""
        self.sentence_embedding = SentenceEmbedding(768, 512, hugging_face_model=hugging_face_model)
        self.input_embedding = nn.Embedding(self.bins,self.C)

        """ Part 2: Define model layers """
        # First layer is just a simple 1D convolution with 20 input channels and 32 output channels. The kernel size is
        # 1. The function of this layer is to convert the sparse representation of a 20 element 1D vector with only one
        # '1' (rest is '0's) to something more meaningful (of higher dimension)
        self.pre_process_conv = nn.Conv1d(in_channels=self.C, out_channels=self.C, kernel_size=1)

        # Create object that represents all the casual layers that follows the pre-processing layer. All the casual
        # layers have 32 input and output channels. The dilation depends on the layer (defined in array). The kernel
        # size is 2 (as in the WaveNet paper)
        self.causal_layers = nn.ModuleList()

        for d in self.dilations:
            self.causal_layers.append(ResidualLayer(in_channels=self.C,
                                                    out_channels=self.C,
                                                    dilation=d,
                                                    kernel_size=self.kernel_size, 
                                                    condition_size=condition_size, 
                                                    global_condition=global_condition, 
                                                    local_condition=local_condition))

        # Define parameters for global and local conditioning
        self.local_condition = local_condition
        self.global_condition = global_condition

        if local_condition or global_condition:
            self.condition_size = condition_size

        # Define layers that depend on type of conditioning
        if global_condition:
            self.gc_initial = nn.Sequential(nn.Linear(768, condition_size), 
                                            nn.ReLU(), 
                                            nn.Linear(condition_size, condition_size), 
                                            nn.ReLU())
        if local_condition:
            self.lc_initial = nn.Sequential(nn.Conv1d(512, condition_size, kernel_size=1), 
                                            nn.ReLU(), 
                                            nn.Conv1d(condition_size, condition_size, kernel_size=1), 
                                            nn.ReLU())

        # 2 post processing layers. The last layer outputs a 1D tensor with 20 elements (corresponding to dimension of
        # one-hot encoded amplitude - should later be send through softmax)
        self.post_process_conv1 = nn.Conv1d(self.C, self.C, kernel_size=1)
        self.post_process_conv2 = nn.Conv1d(self.C, self.bins, kernel_size=1)

    def forward(self, quantisized_x, gc=None, lc=None):
        """ Function: Makes the forward pass/model prediction
            Input:    Mu- and one-hot-encoded waveform. The shape of the input is (batch_size, quantization_bins,
                      samples). It is important that 'x' has at least the length of the models receptive field.
            Output:   Distribution for prediction of next sample. Shape (batch_size, quantization_bins, what's left
                      after dilation, should be 1 at inference) """

        """ Part 1: Get transcript embedding """
        #embed = F.one_hot(quantisized_x, num_classes=self.bins).permute(0,2,1) #<--- one_hot encoding
        embed = self.input_embedding(quantisized_x).permute(0,2,1) #<--- Embedded encoding

        """ Part 2: Define layer block that depends on conditioning """
        if self.global_condition and gc is not None:
            gc = self.gc_initial(gc)
        if self.local_condition and lc is not None:
            lc = self.lc_initial(lc)

        """ Part 3: Through pre-processing layer """
        x = self.pre_process_conv(embed) # shape --> (batch_size, channels, samples)

        """ Part 4: Through stack of dilated causal convolutions """
        skips = []
        for layer in self.causal_layers:
            x, skip = layer(x, gc=gc, lc=lc)
            skips.append(skip)

        """ Part 5: Post processes (-softmax) """
        # Add skip connections together (linearly - meaning we actually only get a vector that has the sum of the last
        # (in this case 37) elements in each skip connection vector - the rest of the information is lost). Notice that
        # the output of the last causual layer (residual) is not a part of the output of the model - only the sum of
        # the skip connections so far is.
        x = torch.stack([s[:, :, -skip.size(2):] for s in skips],0).sum(0) # adding up skip-connections

        # Do the rest of the preprocessing (se figure in paper). We return a tensor of shape (10,20,37).
        x = F.relu(x)
        x = self.post_process_conv1(x) # shape --> (batch_size, channels, samples)
        x = F.relu(x)
        x = self.post_process_conv2(x)  # shape --> (batch_size, quantization_bins, samples)

        return x

class ResidualLayer(nn.Module):
    """ Description: This class is a sub-model of a residual layer (see research paper)"""

    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, dilation:int, condition_size=None,
                 global_condition:bool = False, local_condition:bool = False):
        # Inherent from Pytorch parent class
        super(ResidualLayer, self).__init__()

        """ Part 1: Define model parameters """
        self.dilation = dilation
        self.kernel_size = kernel_size

        """ Part 2: Define model layers """
        # The original Wa original WaveNet paper used a single shared 1x1 conv for both filter (f) and gate (g).
        # Instead we use one for each here i.e. conv_f and conv_g.
        self.conv_fg = nn.Conv1d(in_channels, 2*out_channels, kernel_size=kernel_size, dilation=dilation)

        # 1 shared 1x1 convolution
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, kernel_size=1)

        # Define layer that depend on conditioning
        if global_condition:
            self.gc_layer_fg = nn.Linear(condition_size, 2*out_channels)
        if local_condition:
            self.lc_layer_fg = nn.Conv1d(condition_size, 2*out_channels, 1)

    def forward(self, x, gc=None, lc=None):
        """ Function: Do forward pass/make model prediction """
        # Do conditioning
        fg = self.conv_fg(x)
        
        if lc is not None and gc is not None:
            lc = self.lc_layer_fg(lc)[:,:,-fg.size(-1):]
            gc = self.gc_layer_fg(gc).view(fg.size(0),-1,1)
            fg = fg+lc+gc
        elif lc is not None:
            lc = self.lc_layer_fg(lc)[:,:,-fg.size(-1):]
            fg = fg+lc
        elif gc is not None:
            gc = self.gc_layer_fg(gc).view(fg.size(0),-1,1)
            fg = fg+gc

        # Send through gate
        f,g = torch.chunk(fg,2,dim=1)
        f = torch.tanh(f)
        g = torch.sigmoid(g)
        fg = f * g

        # Save for skip connection
        skip = self.conv_1x1(fg) # <-- TODO try with ReLU instead

        # Save residual as input to next layer residual layer
        residual = x[:, :, -skip.size(2):] + skip

        return residual, skip
    
class SentenceEmbedding(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=8, stride=4, hugging_face_model='bert-base-uncased'):
        super(SentenceEmbedding, self).__init__()
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', hugging_face_model)
        self.transposed_convs = nn.Sequential(nn.ConvTranspose1d(in_channels,out_channels,kernel_size,stride), nn.ReLU(),
                                              nn.ConvTranspose1d(out_channels,out_channels,kernel_size,stride), nn.ReLU(),
                                              nn.ConvTranspose1d(out_channels,out_channels,kernel_size,stride), nn.ReLU(),
                                              nn.ConvTranspose1d(out_channels,out_channels,kernel_size,stride), nn.ReLU())

    def forward(self, tokens):
        """
        :param tokens: Embedding of sentence shape (B, C, T+2) where T is sequence length and C is the size of the embedding dimension.
                       The additional items in the third dimension are a start token [CLS] and the end of sentence token [SEP].
        :return: [CLS] token from BERT embedding as global condition signal and upsampled local condition signal.
        """
        #Feed tokenized transcript into BERT.
        bert_out = self.bert(input_ids=tokens, return_dict=True)['last_hidden_state'] #shape: (B, C, T+2)
        
        #Get first token ([CLS]) as the global condition signal.
        gc_embed = bert_out[:,0]
        
        #Take the innermost items (between the [CLS] and [SEP] tokens) as the local condition signal.
        bert_lc = bert_out[:,1:-1,:].permute(0,2,1)
        #Feed into transposed convolution to perform learned upsampling.
        lc_embed = self.transposed_convs(bert_lc)
        
        return gc_embed, lc_embed