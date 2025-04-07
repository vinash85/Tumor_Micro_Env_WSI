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
import math


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
        if in_features_dim <= 0 or in_features_dim is None:
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
        self.dpout = self.config.dropout
        self.self_attn = nn.MultiheadAttention(self.d_model, self.nhead, dropout= self.dpout, batch_first=True)
        self.linear1 = nn.Linear(self.d_model, self.dim_feedforward)
        self.dropout = nn.Dropout(self.dpout)
        self.linear2 = nn.Linear(self.dim_feedforward, self.d_model)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout1 = nn.Dropout(self.dpout)
        self.dropout2 = nn.Dropout(self.dpout)

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
        #self.vocab_size = self.config.vocab_size
        self.d_model = self.config.d_model
        self.latent = self.config.latent_dim
        self.nhead = self.config.nhead
        self.num_layers = self.config.num_layers
        self.dim_feedforward = self.config.dim_feedforward
        self.max_seq_len = self.config.max_seq_len
        self.embedding = nn.Linear(self.latent, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, self.max_seq_len)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(self.config)
            for _ in range(self.num_layers)
        ])
        self.d_model = self.config.d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))

    def forward(self, src, src_key_padding_mask=None):
        """
        Forward pass of the BidirectionalTransformer.
        :param src: Input tensor.
        :param src_key_padding_mask: Source key padding mask.
        :return: Output tensor.
        """
        batch_size = src.size(0)
        src = self.embedding(src) * math.sqrt(self.d_model)
        # Add cls_token to the beginning of the sequence
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        src = torch.cat((cls_token, src), dim=1)
        # Update the sequence length
        #src_len = src.size(1)
        # Create a new padding mask for the updated sequence
        if src_key_padding_mask is not None:    
            src_key_padding_mask = torch.cat((torch.zeros(batch_size, 1, dtype=torch.bool).to(src.device), src_key_padding_mask), dim=1)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        for layer in self.encoder_layers:
            src = layer(src, src_key_padding_mask=src_key_padding_mask)
            
        return src



class TumorMicroEnvironmentModel(nn.Module):
    """
    Tumor Micro Environment Model for TME project.
    """
    def __init__(self, config):
        """
        Initialize the TumorMicroEnvironmentModel.
        :param num_classes: Number of classes for classification.
        """
        super(TumorMicroEnvironmentModel, self).__init__()
        self.config = config
        self.num_pathology_patches = config.num_pathology_patches
        self.disease_type_latent = None
        self.primary_site_latent = None
        self.age_at_index_latent = None
        self.race_latent = None
        self.gender_latent = None
        # self.age_at_diagnosis_latent = None
        # self.days_to_birth_latent = None
        # self.pathologic_stage_latent = None
        # self.staging_latent = None
        # self.tumor_class_latent = None
        # self.primary_disease_latent = None
        # self.primary_diagnosis_latent = None
        # self.site_biopsy_latent = None
        # self.specific_site_latent = None
        # self.tissue_organ_origin_latent = None
        # self.theraputic_agents_latent = None
        # self.treatment_or_therapy_latent = None
        # self.treatment_type_latent = None
        self.TME_latent = [] # list of linear projection layers for pathology patches


        self.disease_type_latent = nn.Embedding(self.config.disease_type_dict_dim, self.config.latent_dim)
        self.primary_site_latent = nn.Embedding(self.config.primary_site_dict_dim, self.config.latent_dim)
        #self.age_at_index_latent = nn.Embedding(self.config.age_at_index_dict_dim, self.config.latent_dim)
        self.race_latent = nn.Embedding(self.config.race_dict_dim, self.config.latent_dim)
        self.gender_latent = nn.Embedding(self.config.gender_dict_dim, self.config.latent_dim)
        #self.age_at_diagnosis_latent = nn.Embedding(self.config.age_at_diagnosis_dict_dim, self.config.latent_dim)
        #self.days_to_birth_latent = nn.Embedding(self.config.days_to_birth_dict_dim, self.config.latent_dim)
        # self.pathologic_stage_latent = nn.Embedding(self.config.pathologic_stage_dict_dim, self.config.latent_dim)
        # self.staging_latent = nn.Embedding(self.config.staging_dict_dim, self.config.latent_dim)
        # self.tumor_class_latent = nn.Embedding(self.config.tumor_class_dict_dim, self.config.latent_dim)
        # self.primary_disease_latent = nn.Embedding(self.config.primary_disease_dict_dim, self.config.latent_dim)
        # self.primary_diagnosis_latent = nn.Embedding(self.config.primary_diagnosis_dict_dim, self.config.latent_dim)
        # self.site_biopsy_latent = nn.Embedding(self.config.site_biopsy_dict_dim, self.config.latent_dim)
        # self.specific_site_latent = nn.Embedding(self.config.specific_site_dict_dim, self.config.latent_dim)
        # self.tissue_organ_origin_latent = nn.Embedding(self.config.tissue_organ_origin_dict_dim, self.config.latent_dim)
        # self.theraputic_agents_latent = nn.Embedding(self.config.theraputic_agents_dict_dim, self.config.latent_dim)
        # self.treatment_or_therapy_latent = nn.Embedding(self.config.treatment_or_therapy_dict_dim, self.config.latent_dim)
        # self.treatment_type_latent = nn.Embedding(self.config.treatment_type_dict_dim, self.config.latent_dim)

        if config.projection_type == "linear":
            self.age_at_index_latent = nn.Linear(self.config.age_at_index_dict_dim, self.config.latent_dim)
            #self.age_at_index_latent = LinearProjection(self.config.age_at_index_dict_dim, self.config.latent_dim)
            #self.age_at_diagnosis_latent = LinearProjection(self.config.age_at_diagnosis_dict_dim, self.config.latent_dim)
            #self.days_to_birth_latent = LinearProjection(self.config.days_to_birth_dict_dim, self.config.latent_dim)
            #for i in range(self.config.num_pathology_patches):
            #    self.TME_latent.append(LinearProjection(self.config.pathology_patch_emb_dim,self.config.latent_dim))
        elif config.projection_type == "nonlinear":
            self.age_at_index_latent = NonLinearProjection(self.config.age_at_index_dict_dim, self.config.latent_dim)
            #self.age_at_diagnosis_latent = NonLinearProjection(self.config.age_at_diagnosis_dict_dim, self.config.latent_dim)
            #self.days_to_birth_latent = NonLinearProjection(self.config.days_to_birth_dict_dim, self.config.latent_dim)
            #for i in range(self.config.num_pathology_patches):
            #    self.TME_latent.append(NonLinearProjection(self.config.pathology_patch_emb_dim,self.config.nlp_dim,self.config.latent_dim))
        else:
            raise Exception("Invalid projection type")
        self.BidirectionalTransformer = BidirectionalTransformer(self.config)
        self.ClLinear1 = nn.Linear(self.config.latent_dim, self.config.latent_dim)
        self.ClLinear2 = nn.Linear(self.config.latent_dim, self.config.num_classes)
        self.ClRelu = nn.ReLU()
        self.ClDropout = nn.Dropout(self.config.dropout)
        self.flatten = Flatten()


    def create_latent(self, x):
        """
        Create clinical latent features from the input tensor.
        :param x: Input tensor.
        :return: Clinical latent features.
        """
        disease_type, primary_site, age_at_index, race, gender, pathology_patches = x
        #age_at_diagnosis, days_to_birth, pathologic_stage, staging, tumor_class, primary_disease, primary_diagnosis, site_biopsy, specific_site, tissue_organ_origin, theraputic_agents, treatment_or_therapy, treatment_type, 

        disease_type_tokenized = disease_type.int()	
        primary_site_tokenized	= primary_site.int()	
        age_at_index_tokenized = age_at_index.float()	
        race_tokenized = race.int()	
        gender_tokenized =gender.int()
        # age_at_diagnosis_tokenized = age_at_diagnosis.float()
        # days_to_birth_tokenized = days_to_birth.float()	
        # pathologic_stage_tokenized = pathologic_stage.int()
        # staging_tokenized = staging.int()
        # tumor_class_tokenized = tumor_class.int()
        # primary_disease_tokenized = primary_disease.int()
        # primary_diagnosis_tokenized = primary_diagnosis.int()	
        # site_biopsy_tokenized = site_biopsy.int()
        # specific_site_tokenized = specific_site.int()
        # tissue_organ_origin_tokenized = tissue_organ_origin.int()	
        # theraputic_agents_tokenized = theraputic_agents.int()	
        # treatment_or_therapy_tokenized = treatment_or_therapy.int()	
        # treatment_type_tokenized = treatment_type.int()	
        pathology_patches_embeddings = pathology_patches.float()



        disease_type_latval = self.disease_type_latent(disease_type_tokenized)
        primary_site_latval = self.primary_site_latent(primary_site_tokenized)
        age_at_index_latval = self.age_at_index_latent(age_at_index_tokenized).unsqueeze(1)
        race_latval = self.race_latent(race_tokenized)
        gender_latval = self.gender_latent(gender_tokenized)
        # age_at_diagnosis_latval = self.age_at_diagnosis_latent(age_at_diagnosis_tokenized).unsqueeze(1)
        # days_to_birth_latval = self.days_to_birth_latent(days_to_birth_tokenized).unsqueeze(1)
        # pathologic_stage_latval = self.pathologic_stage_latent(pathologic_stage_tokenized)
        # staging_latval = self.staging_latent(staging_tokenized)
        # tumor_class_latval = self.tumor_class_latent(tumor_class_tokenized)
        # primary_disease_latval = self.primary_disease_latent(primary_disease_tokenized)
        # primary_diagnosis_latval = self.primary_diagnosis_latent(primary_diagnosis_tokenized)
        # site_biopsy_latval = self.site_biopsy_latent(site_biopsy_tokenized)
        # specific_site_latval = self.specific_site_latent(specific_site_tokenized)
        # tissue_organ_origin_latval = self.tissue_organ_origin_latent(tissue_organ_origin_tokenized)
        # theraputic_agents_latval = self.theraputic_agents_latent(theraputic_agents_tokenized)
        # treatment_or_therapy_latval = self.treatment_or_therapy_latent(treatment_or_therapy_tokenized)
        # treatment_type_latval = self.treatment_type_latent(treatment_type_tokenized)

        combined_latent = []
        combined_latent.append(disease_type_latval)
        combined_latent.append(primary_site_latval)
        combined_latent.append(age_at_index_latval)
        combined_latent.append(race_latval)
        combined_latent.append(gender_latval)
        # combined_latent.append(age_at_diagnosis_latval)
        # combined_latent.append(days_to_birth_latval)
        # combined_latent.append(pathologic_stage_latval)
        # combined_latent.append(staging_latval)
        # combined_latent.append(tumor_class_latval)
        # combined_latent.append(primary_disease_latval)
        # combined_latent.append(primary_diagnosis_latval)
        # combined_latent.append(site_biopsy_latval)
        # combined_latent.append(specific_site_latval)
        # combined_latent.append(tissue_organ_origin_latval)
        # combined_latent.append(theraputic_agents_latval)
        # combined_latent.append(treatment_or_therapy_latval)
        # combined_latent.append(treatment_type_latval)
        # combined_latent.append(pathology_patches_latval)
        combined_latent.append(pathology_patches_embeddings)  

        return torch.cat(combined_latent,dim=1) 


    def forward(self, x, src_key_padding_mask=None):
        """
        Forward pass of the TumorMicroEnvironmentModel.
        :param x: Input tensor.
        :return: Output tensor.
        """
        concatenated_latent = self.create_latent(x)

        # Apply Bidirectional Transformer
        if src_key_padding_mask is not None:
            src = self.BidirectionalTransformer(concatenated_latent, src_key_padding_mask=src_key_padding_mask)
            cls_token = src[:, 0, :]
            src = self.flatten(src)
            return src
        else:
            src = self.BidirectionalTransformer(concatenated_latent)
            # try mean pooling
            cls_token = src[:, 0, :]

            src = self.flatten(cls_token)
            src = self.ClLinear1(src)
            src = self.ClRelu(src)
            src = self.ClDropout(src)
            src = self.ClLinear2(src)
            return src



