##########################################################################
#
# Author: Kushal Virupakshappa
# Date: 2023-10-01
#
# ######################################################


import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms


# define models

class LinearProjection(nn.Module):
    """
    Linear projection layer for clinical and image features.
    """
    def __init__(self, in_features_dim, out_features_dim):
        """
        Initialize the LinearProjection layer.
        :param in_features_dim: Input feature dimension.
        :param out_features_dim: Output feature dimension.
        """
        super(LinearProjection, self).__init__()
        if config.in_features_dim <= 0 or config.in_features_dim is None:
            raise Exception("Invalid in_features_dim")
        if out_features_dim <= 0 or out_features_dim is None:
            raise Exception("Invalid embedding_dim")
        self.in_features_dim = in_features_dim
        self.out_features_dim = out_features_dim
        self.linear = nn.Linear(self.in_features_dim, self.out_features_dim)

    def forward(self, x):
        return self.linear(x)

class NonLinearProjection(nn.Module):
    def __init__(self, in_features_dim, nlp_dim, out_features_dim):
        """
        Initialize the NonLinearProjection layer.
        :param in_features_dim: Input feature dimension.
        :param out_features_dim: Output feature dimension.
        """
        super(NoneLinearProjection, self).__init__()
        if in_features_dim <= 0 or in_features_dim is None:
            raise Exception("Invalid in_features_dim")
        if out_features_dim <= 0 or out_features_dim is None:
            raise Exception("Invalid embedding_dim")
        if nlp_dim <= 0 or nlp_dim is None:
            raise Exception("Invalid nlp_dim")
        self.in_features_dim = in_features_dim
        self.nlp_dim = nlp_dim
        self.out_features_dim = out_features_dim
        self.linear1 = nn.Linear(self.in_features_dim, self.nlp_dim)
        self.linear2 = nn.Linear(self.nlp_dim, self.out_features_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        return x

class Flatten(nn.Module):
    """
    Flatten layer to flatten the input tensor.
    """
    def forward(self, x):
        return x.view(x.size(0), -1)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        """
        Initialize the TransformerEncoderLayer.
        :param config: Configuration object containing model parameters.
        """
        super().__init__()
        #d_model=768, nhead=12, dim_feedforward=3072, dropout=0.1
        self.config = config
        self.d_model = self.config.d_model
        self.nhead = self.config.nhead
        self.dim_feedforward = self.config.dim_feedforward
        self.dropout = self.config.dropout
        self.self_attn = nn.MultiheadAttention(self.d_model, self.nhead, dropout= self.dropout, batch_first=True)
        self.linear1 = nn.Linear(self.d_model, self.dim_feedforward)
        self.dropout = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(self.dim_feedforward, self.d_model)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)

    def forward(self, src, src_key_padding_mask=None):
        # Bidirectional attention (no causal mask)
        src2 = self.self_attn(
            src, src, src,
            key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return x


class BidirectionalTransformer(nn.Module):
    def __init__(self, config):
        """
        Initialize the BidirectionalTransformer.
        :param config: Configuration object containing model parameters.
        """
        super().__init__()
        #,vocab_size, d_model=768, nhead=12, num_layers=6, dim_feedforward=3072, max_seq_len=512
        self.config = config
        self.vocab_size = self.config.vocab_size
        self.d_model = self.config.d_model
        self.nhead = self.config.nhead
        self.num_layers = self.config.num_layers
        self.dim_feedforward = self.config.dim_feedforward
        self.max_seq_len = self.config.max_seq_len
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, self.max_seq_len)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(self.config)
            for _ in range(self.num_layers)
        ])
        self.d_model = self.config.d_model

    def forward(self, src, src_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        for layer in self.encoder_layers:
            src = layer(src, src_key_padding_mask=src_key_padding_mask)
            
        return src



class TumorMicroEnvironmentModel(nn.Module):
    """
    Tumor Micro Environment Model for TME project.
    """
    def __init__(self, config, num_classes=2):
        """
        Initialize the TumorMicroEnvironmentModel.
        :param num_classes: Number of classes for classification.
        """
        super(TumorMicroEnvironmentModel, self).__init__()
        self.config = config
        self.num_pathology_patches = config.num_pathology_patches
        self.cancer_type_latent = None
        self.cancer_type_detailed_latent = None
        self.time_latent = None
        self.age_latent = None
        self.event_latent = None
        self.sex_latent= None
        self.race_latent = None
        self.exposure_latent= None
        self.survival_latent= None
        self.TME_latent = None
        if config.projection_type == "linear":
            self.cancer_type_latent = LinearProjection(self.config.cancer_type_dict_dim,self.config.latent_dim)
            self.cancer_type_detailed_latent = LinearProjection(self.config.cancer_type_detailed_dict_dim,self.config.latent_dim)
            self.time_latent = LinearProjection(self.config.time_dict_dim,self.config.latent_dim)
            self.age_latent = LinearProjection(self.config.age_dict_dim,self.config.latent_dim)
            self.event_latent = LinearProjection(self.config.event_dict_dim,self.config.latent_dim)
            self.sex_latent =  LinearProjection(self.config.sex_dict_dim,self.config.latent_dim)
            self.race_latent = LinearProjection(self.config.race_dict_dim,self.config.latent_dim)
            self.exposure_latent = LinearProjection(self.config.exposure_dict_dim,self.config.latent_dim)
            self.survival_latent = LinearProjection(self.config.survival_dict_dim,self.config.latent_dim)
            for i in range(self.config.num_pathology_patches):
                self.TME_latent.append(LinearProjection(self.config.pathology_patch_emb_dim,self.config.latent_dim))
        elif config.projection_type == "nonlinear":
            self.cancer_type_latent = NonLinearProjection(self.config.cancer_type_dict_dim,self.config.nlp_dim,self.config.latent_dim)
            self.cancer_type_detailed_latent = NonLinearProjection(self.config.cancer_type_detailed_dict_dim,self.config.nlp_dim,self.config.latent_dim)
            self.time_latent = NonLinearProjection(self.config.time_dict_dim,self.config.nlp_dim,self.config.latent_dim)
            self.age_latent = NonLinearProjection(self.config.age_dict_dim,self.config.nlp_dim,self.config.latent_dim)
            self.event_latent = NonLinearProjection(self.config.event_dict_dim,self.config.nlp_dim,self.config.latent_dim)
            self.sex_latent = NonLinearProjection(self.config.sex_dict_dim,self.config.nlp_dim,self.config.latent_dim)
            self.race_latent = NonLinearProjection(self.config.race_dict_dim,self.config.nlp_dim,self.config.latent_dim)
            self.exposure_latent = NonLinearProjection(self.config.exposure_dict_dim,self.config.nlp_dim,self.config.latent_dim)
            self.survival_latent = NonLinearProjection(self.config.survival_dict_dim,self.config.nlp_dim,self.config.latent_dim)
            for i in range(self.config.num_pathology_patches):
                self.TME_latent.append(NonLinearProjection(self.config.pathology_patch_emb_dim,self.config.nlp_dim,self.config.latent_dim))
        else:
            raise Exception("Invalid projection type")
        self.BidirectionalTransformer = BidirectionalTransformer(self.config)
        self.flatten = Flatten()
    
    def create_latent(self, x):
        """
        Create clinical latent features from the input tensor.
        :param x: Input tensor.
        :return: Clinical latent features.
        """
        cancer_type_tokenized = x['cancer_type_tokenized']
        cancer_type_detailed_tokenized = x['cancer_type_detailed_tokenized']
        time_tokenized = x['time_tokenized']
        age_tokenized = x['age_tokenized']
        event_tokenized = x['event_tokenized']
        sex_tokenized = x['sex_tokenized']
        race_tokenized = x['race_tokenized']
        exposure_tokenized = x['exposure_tokenized']
        survival_tokenized = x['survival_tokenized']
        pathology_patches_embeddings = x['pathology_patches']

        # Extract clinical features
        cancer_type_latent = self.cancer_type_latent(cancer_type_tokenized)
        cancer_type_detailed_latent = self.cancer_type_detailed_latent(cancer_type_detailed_tokenized)
        time_latent = self.time_latent(time_tokenized)
        age_latent = self.age_latent(age_tokenized)
        event_latent = self.event_latent(event_tokenized)
        sex_latent = self.sex_latent(sex_tokenized)
        race_latent = self.race_tokenized(race_tokenized)
        exposure_latent = self.exposure_latent(exposure_tokenized)
        survival_latent = self.survival_latent(survival_tokenized)

        combined_latent = []
        combined_latent.append(cancer_type_latent)
        combined_latent.append(cancer_type_detailed_latent)
        combined_latent.append(time_latent)
        combined_latent.append(age_latent)
        combined_latent.append(event_latent)
        combined_latent.append(sex_latent)
        combined_latent.append(race_latent)
        combined_latent.append(exposure_latent)
        combined_latent.append(survival_latent)
        for i in range(self.config.num_pathology_patches):
            combined_latent_latent.append(self.TME_latent[i](pathology_patches_embeddings[i]))   

        return torch.cat(combined_latent, dim=1) 


    def forward(self, x):
        """
        Forward pass of the TumorMicroEnvironmentModel.
        :param x: Input tensor.
        :return: Output tensor.
        """
        concatenated_latent = self.create_latent(x)

        # Apply Bidirectional Transformer
        if x['src_key_padding_mask'] is not None:
            src_key_padding_mask = x['src_key_padding_mask']
            src = self.BidirectionalTransformer(concatenated_latent, src_key_padding_mask=src_key_padding_mask)
            cls_token = src[:, 0, :]
            src = self.flatten(src)
            return src
        else:
            src = self.BidirectionalTransformer(concatenated_latent)
            cls_token = src[:, 0, :]
            src = self.flatten(src)
            src = nn.Linear(src.numel(), src.numel())(src)
            src = nn.ReLU()(src)
            src = nn.Dropout(self.config.dropout)(src)
            src = nn.Linear(src.numel(), self.config.num_classes)(src)
            return src
