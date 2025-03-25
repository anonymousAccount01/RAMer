from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math

from .module_encoder import TfModel, TextConfig, VisualConfig, AudioConfig
from .until_module import PreTrainedModel, LayerNorm
from .until_module import getBinaryTensor, CTCModule, MLLinear, MLAttention, GradReverse
import warnings
from .losses import *
import numpy as np
import pdb

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(0, 1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
    

class RAMerPreTrainedModel(PreTrainedModel, nn.Module):
    def __init__(self, text_config, visual_config, audio_config,*inputs, **kwargs):
        # utilize bert config as base config
        super(RAMerPreTrainedModel, self).__init__(visual_config)
        self.text_config = text_config
        self.visual_config = visual_config
        self.audio_config = audio_config
        self.visual = None
        self.audio = None
        self.text = None

    
    @classmethod
    def from_pretrained(cls, text_model_name, visual_model_name, audio_model_name, personality_model_name,
                        state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):
        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0
        text_config, _= TextConfig.get_config(text_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        visual_config, _ = VisualConfig.get_config(visual_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        audio_config, _ = AudioConfig.get_config(audio_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        personality_config, _ = TextConfig.get_config(personality_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        model = cls(text_config, visual_config, audio_config, personality_config, *inputs, **kwargs)
        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)
        return model


class Normalize(nn.Module):
    def __init__(self, dim):
        super(Normalize, self).__init__()
        self.norm2d = LayerNorm(dim)

    def forward(self, inputs):
        inputs = torch.as_tensor(inputs).float()
        inputs = inputs.view(-1, inputs.shape[-2], inputs.shape[-1])
        output = self.norm2d(inputs)
        return output


def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config
    
class RAMer(RAMerPreTrainedModel):
    def __init__(self, text_config, visual_config, audio_config, personality_config, task_config):
        super(RAMer, self).__init__(text_config, visual_config, audio_config)
        self.task_config = task_config
        self.num_classes = task_config.num_classes
        self.aligned = task_config.aligned
        self.proto_m = task_config.proto_m
        # self.num_classes = 9 if config['emo_type'] == 'primary' else 14

        text_config = update_attr("text_config", text_config, "num_hidden_layers",
                                  self.task_config, "text_num_hidden_layers")
        self.text = TfModel(text_config) # transformer model
        visual_config = update_attr("visual_config", visual_config, "num_hidden_layers",
                                    self.task_config, "visual_num_hidden_layers")
        self.visual = TfModel(visual_config)
        audio_config = update_attr("audio_config", audio_config, "num_hidden_layers",
                                   self.task_config, "audio_num_hidden_layers")
        self.audio = TfModel(audio_config)
        personality_config = update_attr("personality_config", personality_config, "num_hidden_layers",
                                         self.task_config, "personality_num_hidden_layers")
        self.personality = TfModel(personality_config)

        self.text_norm = Normalize(task_config.text_dim)
        self.visual_norm = Normalize(task_config.video_dim)
        self.audio_norm = Normalize(task_config.audio_dim)
        # self.personality_norm = Normalize(D_p)
        self.bce_loss = nn.BCEWithLogitsLoss() # binary cross entropy loss with sigmoid transform
        self.mse_loss = nn.MSELoss()
        self.criterion_cl = SupConLoss()
        self.ml_loss = nn.BCELoss() # binary cross entropy loss without sigmoid transform
        self.adv_loss = nn.CrossEntropyLoss()

        self.apply(self.init_weights)

        self.common_feature_extractor= nn.Sequential(
            nn.Linear(task_config.hidden_size, task_config.hidden_size),
            nn.Dropout(p=0.3),
            nn.Tanh()
        ) 

        self.common_classfier = nn.Sequential(
            nn.Linear(task_config.hidden_size, self.num_classes),
            nn.Dropout(0.1),
            nn.Sigmoid()
        )

        self.private_feature_extractor = nn.ModuleList([nn.Sequential(
            nn.Linear(task_config.hidden_size, task_config.hidden_size),
            nn.Dropout(p=0.1),
            nn.Tanh()
        ) for _ in range(3)])

        self.modal_discriminator = nn.Sequential(
            nn.Linear(task_config.hidden_size, task_config.hidden_size // 2),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(task_config.hidden_size // 2, 3),
        )

        self.text_attention = MLAttention(self.num_classes, task_config.hidden_size)# Multi-label attention
        self.visual_attention = MLAttention(self.num_classes, task_config.hidden_size)
        self.audio_attention = MLAttention(self.num_classes, task_config.hidden_size)
        self.personality_attention = MLAttention(self.num_classes, task_config.hidden_size)


        self.proj_text = MLLinear([task_config.hidden_size, task_config.hidden_size//2], task_config.proj_size)
        self.proj_visual = MLLinear([task_config.hidden_size, task_config.hidden_size//2], task_config.proj_size)
        self.proj_audio = MLLinear([task_config.hidden_size, task_config.hidden_size//2], task_config.proj_size)

        self.de_proj_text = MLLinear([task_config.proj_size, task_config.hidden_size//2], task_config.hidden_size)
        self.de_proj_visual = MLLinear([task_config.proj_size, task_config.hidden_size//2], task_config.hidden_size)
        self.de_proj_audio = MLLinear([task_config.proj_size, task_config.hidden_size//2], task_config.hidden_size)


        self.tv2a = MLLinear([task_config.hidden_size * 3], task_config.hidden_size)
        self.ta2v = MLLinear([task_config.hidden_size * 3], task_config.hidden_size)
        self.va2t = MLLinear([task_config.hidden_size * 3], task_config.hidden_size)
        self.max_pool = nn.MaxPool1d(3)

        self.agg = MLLinear([task_config.hidden_size * self.num_classes, task_config.hidden_size], self.num_classes)
        

        self.text_clf_weight = nn.Parameter(torch.Tensor(self.num_classes, task_config.hidden_size))
        nn.init.kaiming_uniform_(self.text_clf_weight, a=math.sqrt(5))
        self.visual_clf_weight = nn.Parameter(torch.Tensor(self.num_classes, task_config.hidden_size))
        nn.init.kaiming_uniform_(self.visual_clf_weight, a=math.sqrt(5))
        self.audio_clf_weight = nn.Parameter(torch.Tensor(self.num_classes, task_config.hidden_size))
        nn.init.kaiming_uniform_(self.audio_clf_weight, a=math.sqrt(5))

        self.sigmoid = nn.Sigmoid()

        self.register_buffer('text_pos_protos', torch.zeros(self.num_classes, task_config.proj_size))
        self.register_buffer('text_neg_protos', torch.zeros(self.num_classes, task_config.proj_size))
        self.register_buffer('visual_pos_protos', torch.zeros(self.num_classes, task_config.proj_size))
        self.register_buffer('visual_neg_protos', torch.zeros(self.num_classes, task_config.proj_size))
        self.register_buffer('audio_pos_protos', torch.zeros(self.num_classes, task_config.proj_size))
        self.register_buffer('audio_neg_protos', torch.zeros(self.num_classes, task_config.proj_size))

        self.register_buffer('queue', torch.randn(task_config.moco_queue, task_config.proj_size))
        self.register_buffer("queue_label", torch.randn(task_config.moco_queue, 1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue = F.normalize(self.queue, dim=0)
        

        if not self.aligned:
            self.a2t_ctc = CTCModule(task_config.audio_dim, 50 if task_config.unaligned_mask_same_length else 500)
            self.v2t_ctc = CTCModule(task_config.video_dim, 50 if task_config.unaligned_mask_same_length else 500)

    def dequeue_and_enqueue(self, feats, labels):
        batch_size = feats.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size >= self.task_config.moco_queue:
            self.queue[ptr:,:] = feats[:self.task_config.moco_queue-ptr,:]
            self.queue[:batch_size - self.task_config.moco_queue + ptr,:] = feats[self.task_config.moco_queue-ptr:,:]
            self.queue_label[ptr:, :] = labels[:self.task_config.moco_queue - ptr, :]
            self.queue_label[:batch_size - self.task_config.moco_queue + ptr, :] = labels[self.task_config.moco_queue - ptr:,
                                                                             :]
        else:
            self.queue[ptr:ptr+batch_size, :] = feats
            self.queue_label[ptr:ptr + batch_size, :] = labels
        ptr = (ptr + batch_size) % self.task_config.moco_queue  # move pointer
        self.queue_ptr[0] = ptr

    def get_text_visual_audio_pers_output(self, text, text_mask, visual, visual_mask, audio, audio_mask, personality=None, personality_mask=None):
        text_layers, text_pooled_output = self.text(text, text_mask, output_all_encoded_layers=True)
        text_output = text_layers[-1]
        visual_layers, visual_pooled_output = self.visual(visual, visual_mask, output_all_encoded_layers=True)
        visual_output = visual_layers[-1]
        audio_layers, audio_pooled_output = self.audio(audio, audio_mask, output_all_encoded_layers=True)
        audio_output = audio_layers[-1]
        if personality is not None:
            personality_layers, personality_pooled_output = self.personality(personality, personality_mask, output_all_encoded_layers=True)
            personality_output = personality_layers[-1]
        else:
            personality_output = None
        # print(f"personality output: {personality_output}")
        return text_output, visual_output, audio_output, personality_output

    def get_cl_labels(self, labels, times = 1):
        text_labels = torch.zeros_like(labels) + labels
        visual_labels = torch.zeros_like(labels) + labels
        audio_labels = torch.zeros_like(labels) + labels

        text_cl_labels = torch.zeros_like(text_labels, dtype=torch.long)
        visual_cl_labels = torch.zeros_like(visual_labels, dtype=torch.long)
        audio_cl_labels = torch.zeros_like(audio_labels, dtype=torch.long)

        example_idx, label_idx = torch.where(text_labels >= 0.5)
        text_cl_labels[example_idx, label_idx] = label_idx
        example_idx, label_idx = torch.where(text_labels < 0.5)
        text_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 1

        example_idx, label_idx = torch.where(visual_labels >= 0.5)
        visual_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 2
        example_idx, label_idx = torch.where(visual_labels < 0.5)
        visual_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 3

        example_idx, label_idx = torch.where(audio_labels >= 0.5)
        audio_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 4
        example_idx, label_idx = torch.where(audio_labels < 0.5)
        audio_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 5

        cl_labels = torch.stack([text_cl_labels, visual_cl_labels, audio_cl_labels], dim=1)

        cl_labels = cl_labels.to(torch.int)
        if times > 1:
            final_cl_labels = torch.cat([cl_labels, cl_labels], dim=1)
            for i in range(2, times):
                final_cl_labels = torch.cat([final_cl_labels, cl_labels], dim=1)
        else:
            final_cl_labels = cl_labels
        return final_cl_labels

    def get_cl_mask(self, cl_labels, batch_size):
        mask = torch.eq(cl_labels[:batch_size], cl_labels.T).float()
        neg_mask = torch.ones_like(mask)
        return mask, neg_mask

    def update_protos(self, pos_protos, neg_protos, feats, gt_labels):
        b, c = gt_labels.shape[0], gt_labels.shape[1]
        for i in range(b):
            for j in range(c):
                if gt_labels[i][j] == 1:
                    pos_protos[j] = pos_protos[j] * self.proto_m + (1 - self.proto_m) * feats[i][j]
                else:
                    neg_protos[j] = neg_protos[j] * self.proto_m + (1 - self.proto_m) * feats[i][j]

    def forward(self, visual,audio,text,visual_mask, audio_mask,text_mask,seq_lengths=None, target_loc=None, seg_len=None, n_c=None,
                personality=None, personality_mask=None, groundTruth_labels=None, training=True):
        
        visual = self.visual_norm(visual)
        audio = self.audio_norm(audio)
        text = self.text_norm(text)
        if personality is not None: # personality is optional
            personality = self.personality_norm(personality)
        text_output, visual_output, audio_output, personality_output = self.get_text_visual_audio_pers_output(text, text_mask, visual,visual_mask, 
                                                                                                              audio, audio_mask, personality, personality_mask)  
        
        text, text_attention = self.text_attention(text_output, (1 - text_mask).type(torch.bool)) 
        visual, visual_attention = self.visual_attention(visual_output, (1 - visual_mask).type(torch.bool)) 
        audio, audio_attention = self.audio_attention(audio_output, (1 - audio_mask).type(torch.bool)) 
        
        if personality is not None:
            personality, personality_attention = self.personality_attention(personality_output, (1 - personality_mask).type(torch.bool))
            U_all_visual = []; U_all_text = []; U_all_audio = [] 
            for i in range(visual_mask.shape[0]): 
                target_moment, target_character = -1, -1
                for j in range(target_loc.shape[1]): 
                    if target_loc[i][j] == 1:
                        target_moment = j % int(seg_len[i].cpu().numpy())
                        target_character = int(j / seg_len[i].cpu().numpy())
                        break
                
                inp_V = visual[i, : seq_lengths[i], :].reshape((n_c[i], seg_len[i], -1)).transpose(0, 1)
                inp_T = text[i, : seq_lengths[i], :].reshape((n_c[i], seg_len[i], -1)).transpose(0, 1)
                inp_A = audio[i, : seq_lengths[i], :].reshape((n_c[i], seg_len[i], -1)).transpose(0, 1)
                inp_P = personality[i, : seq_lengths[i], :].reshape((n_c[i], seg_len[i], -1)).transpose(0, 1)

                mask_V = visual_mask[i, : seq_lengths[i]].reshape((n_c[i], seg_len[i])).transpose(0, 1)
                mask_T = text_mask[i, : seq_lengths[i]].reshape((n_c[i], seg_len[i])).transpose(0, 1)
                mask_A = audio_mask[i, : seq_lengths[i]].reshape((n_c[i], seg_len[i])).transpose(0, 1)

                
                inp_V = torch.cat([inp_V, inp_P], dim=2)
                inp_A = torch.cat([inp_A, inp_P], dim=2)
                inp_T = torch.cat([inp_T, inp_P], dim=2)

                U_visual = []; U_text = []; U_audio = [] 

                for k in range(n_c[i]): 
                    new_inp_A, new_inp_T, new_inp_V = inp_A.clone(), inp_T.clone(), inp_V.clone(),
                    
                    
                    for j in range(seg_len[i]):
                        att_V, _ = self.attn(inp_V[j, :], inp_V[j, :], inp_V[j, :], mask_V[j, :])
                        att_T, _ = self.attn(inp_T[j, :], inp_T[j, :], inp_T[j, :], mask_T[j, :])
                        att_A, _ = self.attn(inp_A[j, :], inp_A[j, :], inp_A[j, :], mask_A[j, :])
                        new_inp_V[j, :] = att_V + inp_V[j, :]
                        new_inp_A[j, :] = att_A + inp_A[j, :]
                        new_inp_T[j, :] = att_T + inp_T[j, :]

                    
                    att_V, _ = self.attn(new_inp_V[:, k], new_inp_V[:, k], new_inp_V[:, k], mask_V[:, k])
                    att_A, _ = self.attn(new_inp_A[:, k], new_inp_A[:, k], new_inp_A[:, k], mask_A[:, k])
                    att_T, _ = self.attn(new_inp_T[:, k], new_inp_T[:, k], new_inp_T[:, k], mask_T[:, k])

                    
                    inner_V = (att_V[target_moment] + new_inp_V[target_moment][k]).squeeze()
                    inner_A = (att_A[target_moment] + new_inp_A[target_moment][k]).squeeze()
                    inner_T = (att_T[target_moment] + new_inp_T[target_moment][k]).squeeze()

                    
                    U_visual.append(inner_V)
                    U_text.append(inner_T)
                    U_audio.append(inner_A)

                if len(U_visual) == 1: 
                    U_all_visual.append(U_visual[0])
                    U_all_text.append(U_text[0])
                    U_all_audio.append(U_audio[0])
                else:
                    U_all_visual.append(U_all_visual[target_character])
                    U_all_text.append(U_all_text[target_character])
                    U_all_audio.append(U_all_audio[target_character])

            
            U_all_visual = torch.stack(U_all_visual, dim=0) 
            U_all_text = torch.stack(U_all_text, dim=0)
            U_all_audio = torch.stack(U_all_audio, dim=0)
        else:
            U_all_visual = visual; U_all_text = text; U_all_audio = audio
        
        latent_text = self.proj_text(U_all_text) 
        latent_visual = self.proj_visual(U_all_visual)
        latent_audio = self.proj_audio(U_all_audio)
        recon_text = self.de_proj_text(latent_text) 
        recon_visual = self.de_proj_visual(latent_visual)
        recon_audio = self.de_proj_audio(latent_audio)


        text_n = F.normalize(latent_text, p=2, dim=-1)
        visual_n = F.normalize(latent_visual, p=2, dim=-1)
        audio_n = F.normalize(latent_audio, p=2, dim=-1)
        
        text_protos = torch.stack([self.text_pos_protos, self.text_neg_protos])
        visual_protos = torch.stack([self.visual_pos_protos, self.visual_neg_protos])
        audio_protos = torch.stack([self.audio_pos_protos, self.audio_neg_protos])
        text_sim = torch.einsum('bld,nld->bln', text_n, text_protos)
        visual_sim = torch.einsum('bld,nld->bln', visual_n, visual_protos)
        audio_sim = torch.einsum('bld,nld->bln', audio_n, audio_protos)
        text_sim = torch.softmax(text_sim, dim=-1)
        visual_sim = torch.softmax(visual_sim, dim=-1)
        audio_sim = torch.softmax(audio_sim, dim=-1)
        if not training:
            text_pos_sim, text_neg_sim = text_sim[:, :, 0], text_sim[:, :, 1]
            visual_pos_sim, visual_neg_sim = visual_sim[:, :, 0], visual_sim[:, :, 1]
            audio_pos_sim, audio_neg_sim = audio_sim[:, :, 0], audio_sim[:, :, 1]
            text_pos_mask = (text_pos_sim > text_neg_sim).to(torch.float)
            text_neg_mask = 1 - text_pos_mask
            visual_pos_mask = (visual_pos_sim > visual_neg_sim).to(torch.float)
            visual_neg_mask = 1 - visual_pos_mask
            audio_pos_mask = (audio_pos_sim > audio_neg_sim).to(torch.float)
            audio_neg_mask = 1 - audio_pos_mask
            text_latent_padding = text_pos_mask.unsqueeze(-1) * self.text_pos_protos.unsqueeze(0) + \
                                  text_neg_mask.unsqueeze(-1) * self.text_neg_protos.unsqueeze(0)
            visual_latent_padding = visual_pos_mask.unsqueeze(-1) * self.visual_pos_protos.unsqueeze(0) + \
                                  visual_neg_mask.unsqueeze(-1) * self.visual_neg_protos.unsqueeze(0)
            audio_latent_padding = audio_pos_mask.unsqueeze(-1) * self.audio_pos_protos.unsqueeze(0) + \
                                  audio_neg_mask.unsqueeze(-1) * self.audio_neg_protos.unsqueeze(0)
        else:
            text_latent_padding = torch.einsum('bln,nld->bld', text_sim, text_protos) 
            visual_latent_padding = torch.einsum('bln,nld->bld', visual_sim, visual_protos) 
            audio_latent_padding = torch.einsum('bln,nld->bld', audio_sim, audio_protos)
        text_padding = self.de_proj_text(text_latent_padding)
        visual_padding = self.de_proj_visual(visual_latent_padding)
        audio_padding = self.de_proj_audio(audio_latent_padding) 



        audio_aug = self.tv2a(torch.cat([recon_text, recon_visual, audio_padding], dim=-1)) 
        visual_aug = self.ta2v(torch.cat([recon_text, visual_padding, recon_audio], dim=-1)) 
        text_aug = self.va2t(torch.cat([text_padding, recon_visual, recon_audio], dim=-1))
        text_clf_out_3 = torch.einsum('bld,ld->bl', text_aug, self.text_clf_weight)
        visual_clf_out_3 = torch.einsum('bld,ld->bl', visual_aug, self.visual_clf_weight)
        audio_clf_out_3 = torch.einsum('bld,ld->bl', audio_aug, self.audio_clf_weight)

        audio_beta = self.tv2a(torch.cat([text_aug, visual_aug, audio_aug], dim=-1)) 
        visual_beta = self.ta2v(torch.cat([text_aug, visual_aug, audio_aug], dim=-1)) 
        text_beta = self.va2t(torch.cat([text_aug, visual_aug, audio_aug], dim=-1)) 
        if self.task_config.comp_rec_loss or self.task_config.save_sub_tensor_rec:
            total_tensor_rec = torch.stack([text_beta, visual_beta, audio_beta], dim=2)
            
        text_clf_out_4 = torch.einsum('bld,ld->bl', text_beta, self.text_clf_weight)  
        visual_clf_out_4 = torch.einsum('bld,ld->bl', visual_beta, self.visual_clf_weight)
        audio_clf_out_4 = torch.einsum('bld,ld->bl', audio_beta, self.audio_clf_weight)

        if self.task_config.comp_adv_loss or self.task_config.save_sub_tensor:
            
            private_text = self.private_feature_extractor[0](U_all_text)
            
            private_visual = self.private_feature_extractor[1](U_all_visual)
            private_audio = self.private_feature_extractor[2](U_all_audio)
            common_text = self.common_feature_extractor(text_beta)
            common_visual = self.common_feature_extractor(visual_beta)
            common_audio = self.common_feature_extractor(audio_beta)

            if self.task_config.save_sub_tensor:
                total_tensor_private = torch.stack([private_text, private_visual, private_audio], dim=2)
                total_tensor_common = torch.stack([common_text, common_visual, common_audio], dim=2)
                total_tensor = torch.stack([total_tensor_private, total_tensor_common], dim=3)
                

            common_feature = common_text + common_visual + common_audio 
            common_mask = torch.ones(text.shape[:2], dtype=torch.long, device=text.device) 
            

        text_clf_out_1 = torch.einsum('bld,ld->bl', U_all_text, self.text_clf_weight)  
        visual_clf_out_1 = torch.einsum('bld,ld->bl', U_all_visual, self.visual_clf_weight)
        audio_clf_out_1 = torch.einsum('bld,ld->bl', U_all_audio, self.audio_clf_weight)


        if training:
            latent_aug_text = self.proj_text(text_aug) 
            latent_aug_visual = self.proj_visual(visual_aug)
            latent_aug_audio = self.proj_audio(audio_aug) 
            latent_beta_text = self.proj_text(text_beta) 
            latent_beta_visual = self.proj_visual(visual_beta) 
            latent_beta_audio = self.proj_audio(audio_beta) 
            total_proj = torch.stack([latent_text, latent_visual, latent_audio,
                                      latent_aug_text, latent_aug_visual, latent_aug_audio,
                                      latent_beta_text, latent_beta_visual, latent_beta_audio], dim=1)
            
            text_modal = torch.zeros_like(common_mask).view(-1) 
            visual_modal = torch.ones_like(common_mask).view(-1) 
            audio_modal = visual_modal.data.new(visual_modal.size()).fill_(2) 
            
            private_text_modal_pred = self.modal_discriminator(private_text).view(-1, 3)
            private_visual_modal_pred = self.modal_discriminator(private_visual).view(-1, 3)
            private_audio_modal_pred = self.modal_discriminator(private_audio).view(-1, 3)
            
            common_text_modal_pred = self.modal_discriminator(GradReverse.grad_reverse(common_text, 1)).view(-1, 3)
            common_visual_modal_pred = self.modal_discriminator(GradReverse.grad_reverse(common_visual, 1)).view(-1, 3)
            common_audio_modal_pred = self.modal_discriminator(GradReverse.grad_reverse(common_audio, 1)).view(-1, 3)

            label_time = 3

            total_proj = total_proj.view(-1, total_proj.shape[-1])
            total_proj = F.normalize(total_proj, dim=-1)
            cl_labels = self.get_cl_labels(groundTruth_labels, times=label_time).view(-1).unsqueeze(-1)
            text_norm = F.normalize(latent_text.data, dim=-1)
            visual_norm = F.normalize(latent_visual.data, dim=-1)
            audio_norm = F.normalize(latent_audio.data, dim=-1)
            cl_feats = torch.cat((total_proj, self.queue.clone().detach()), dim=0)
            total_cl_labels = torch.cat((cl_labels, self.queue_label.clone().detach()), dim=0)
            batch_size = cl_feats.shape[0]
            cl_mask, cl_neg_mask = self.get_cl_mask(total_cl_labels, batch_size)
            cl_loss = self.criterion_cl(cl_feats, cl_mask, cl_neg_mask, batch_size)
            self.dequeue_and_enqueue(total_proj, cl_labels)
            self.update_protos(self.text_pos_protos, self.text_neg_protos, text_norm, groundTruth_labels)
            self.update_protos(self.visual_pos_protos, self.visual_neg_protos, visual_norm, groundTruth_labels)
            self.update_protos(self.audio_pos_protos, self.audio_neg_protos, audio_norm, groundTruth_labels)


        clf_out_1 = torch.stack([text_clf_out_1, visual_clf_out_1, audio_clf_out_1], dim=-1)
        _, clf_max_idx_1 = torch.max(clf_out_1, -1)
        clf_out_1 = self.max_pool(clf_out_1).squeeze(-1) 
        clf_out_3 = torch.stack([text_clf_out_3, visual_clf_out_3, audio_clf_out_3], dim=-1)
        _, clf_max_idx_3 = torch.max(clf_out_3, -1)
        clf_out_3 = self.max_pool(clf_out_3).squeeze(-1) 
        clf_out_4 = torch.stack([text_clf_out_4, visual_clf_out_4, audio_clf_out_4], dim=-1)
        _, clf_max_idx_4 = torch.max(clf_out_4, -1)
        clf_out_4 = self.max_pool(clf_out_4).squeeze(-1) 
        predict_scores_clf4 = self.sigmoid(clf_out_4)


        total_aug = torch.stack([text_beta, visual_beta, audio_beta], dim=1) 
        agg_out = self.agg(total_aug.view(total_aug.shape[0], total_aug.shape[1], -1))
        agg_scores = self.sigmoid(agg_out)
        predict_agg_scores = torch.mean(agg_scores, dim=1)

        predict_final_scores_mean = (predict_agg_scores + predict_scores_clf4) / 2
        predict_final_labels_mean = getBinaryTensor(predict_final_scores_mean,
                                                  boundary=self.task_config.binary_threshold)
        predict_scores = predict_final_scores_mean
        predict_labels = predict_final_labels_mean
        

        if training:
            total_aug_clf_loss = self.bce_loss(agg_out, groundTruth_labels.unsqueeze(-2).repeat(1, 3, 1)) 
            shuffle_sample_idx = torch.zeros(self.num_classes, total_aug.shape[1], total_aug.shape[0], dtype=torch.long)
            
            for l in range(self.num_classes): 
                for m in range(total_aug.shape[1]):
                    one_idx_ = np.arange(total_aug.shape[0]).tolist()
                    one_idx = self.stack_shuffle(one_idx_, stacks=4, iterations=2)
                    one_idx = torch.tensor(one_idx, dtype=torch.long)
                    shuffle_sample_idx[l][m] += one_idx
                     
            shuffle_sample_idx = shuffle_sample_idx.permute(2, 1, 0)

            shuffle_modality_idx = torch.zeros(self.num_classes, total_aug.shape[0], total_aug.shape[1], dtype=torch.long)
            
            for l in range(self.num_classes): 
                for s in range(total_aug.shape[0]):
                    one_idx_ = np.arange(total_aug.shape[1]).tolist()
                    one_idx = self.stack_shuffle(one_idx_, stacks=3, iterations=2)
                    one_idx = torch.tensor(one_idx, dtype=torch.long)
                     
                    
                    shuffle_modality_idx[l][s] += one_idx
                    # pdb.set_trace()
            shuffle_modality_idx = shuffle_modality_idx.permute(1, 2, 0)

            label_idx = torch.zeros(total_aug.shape[0], total_aug.shape[1], self.num_classes) + torch.tensor(
                list(range(self.num_classes)))
            label_idx = label_idx.to(torch.long)
            shuffle_total_aug = total_aug[shuffle_sample_idx, shuffle_modality_idx, label_idx]
            shuffle_aug_out = self.agg(shuffle_total_aug.view(total_aug.shape[0], total_aug.shape[1], -1))
            shuffle_gt_labels = groundTruth_labels.unsqueeze(-2).repeat(1, 3, 1)[shuffle_sample_idx, shuffle_modality_idx, label_idx]
            shuffle_aug_clf_loss = self.bce_loss(shuffle_aug_out, shuffle_gt_labels)

        if training:
            all_loss = 0
            
            pooled_common = common_feature[:, 0] 
            common_pred = self.common_classfier(pooled_common)
            ml_loss = self.ml_loss(predict_scores, groundTruth_labels)
            cml_loss = self.ml_loss(common_pred, groundTruth_labels)
            preivate_diff_loss = self.calculate_orthogonality_loss(private_text, private_visual) + self.calculate_orthogonality_loss(private_text, private_audio) + self.calculate_orthogonality_loss(private_visual, private_audio)
            common_diff_loss = self.calculate_orthogonality_loss(common_text, private_text) + self.calculate_orthogonality_loss(common_visual, private_visual) + self.calculate_orthogonality_loss(common_audio, private_audio)
            adv_preivate_loss = self.adv_loss(private_text_modal_pred, text_modal) + self.adv_loss(private_visual_modal_pred, visual_modal) + self.adv_loss(private_audio_modal_pred, audio_modal)
            adv_common_loss = self.adv_loss(common_text_modal_pred, text_modal) + self.adv_loss(common_visual_modal_pred, visual_modal) + self.adv_loss(common_audio_modal_pred, audio_modal)
            if self.task_config.comp_rec_loss:
                clf_loss = self.bce_loss(clf_out_1, groundTruth_labels) * self.task_config.lsr_clf_weight
                clf_loss += self.bce_loss(clf_out_3, groundTruth_labels) * self.task_config.aug_clf_weight
                clf_loss += self.bce_loss(clf_out_4, groundTruth_labels) * self.task_config.beta_clf_weight
                all_loss += clf_loss

                all_loss += cl_loss * self.task_config.cl_weight

                aug_mse_loss = self.mse_loss(text_aug, U_all_text) + self.mse_loss(visual_aug, U_all_visual)\
                            + self.mse_loss(audio_aug, U_all_audio)
                beta_mse_loss = self.mse_loss(text_beta, U_all_text) + self.mse_loss(visual_beta, U_all_visual)\
                                + self.mse_loss(audio_beta, U_all_audio)
                recon_mse_loss = self.mse_loss(recon_text, U_all_text) + self.mse_loss(recon_visual, U_all_visual) \
                                + self.mse_loss(recon_audio, U_all_audio)
                all_loss += recon_mse_loss * self.task_config.recon_mse_weight\
                            + aug_mse_loss * self.task_config.aug_mse_weight + beta_mse_loss * self.task_config.beta_mse_weight

            
            all_loss += shuffle_aug_clf_loss * self.task_config.shuffle_aug_clf_weight
            if self.task_config.comp_adv_loss:
                all_loss += self.task_config.com_pri_weight * (adv_common_loss + adv_preivate_loss) + self.task_config.diff_weight * (preivate_diff_loss + common_diff_loss) + self.task_config.cml_weight * cml_loss
            else:
                all_loss += self.task_config.cml_weight * cml_loss
            return all_loss, predict_labels, groundTruth_labels, predict_scores
        else:
            if self.task_config.save_sub_tensor:
                max_idx_list = [clf_max_idx_1, clf_max_idx_3, clf_max_idx_4]
                return predict_labels, groundTruth_labels, predict_scores, total_tensor, max_idx_list
            else:
                return predict_labels, groundTruth_labels, predict_scores
    
    def calculate_orthogonality_loss(self, first_feature, second_feature):
        diff_loss = torch.norm(torch.bmm(first_feature, second_feature.transpose(1, 2)), dim=(1, 2)).pow(2).mean()
        return diff_loss
    
    def stack_shuffle(self, lst, stacks=3, iterations=1):
        for _ in range(iterations):
            num_stacks = stacks if isinstance(stacks, int) else len(stacks)
            stack_size = len(lst) // num_stacks
            stacks = [lst[i:i + stack_size] for i in range(0, len(lst), stack_size)]

            shuffled = []
            while any(stacks):  
                for stack in stacks:
                    if stack:
                        shuffled.append(stack.pop(0))  

            lst = shuffled
            
            sorted_data = sorted(lst, reverse=True)  

        return sorted_data


