import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)   # non-linear activation
        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class ArcSimilarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp, margin=0.05):
        super().__init__()
        self.temp = temp
        self.margin = margin
        self.cos = nn.CosineSimilarity(dim=-1)

    def calculate_arccos1(self, cos_sim, labels=None):
        theta = torch.acos(torch.clamp(cos_sim, -1, 1))

        if labels is None:
            labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)

        num_classes = labels.max().item() + 1
        one_hot_labels = F.one_hot(labels, num_classes)

        selected_labels = torch.where(
            torch.gt(theta, math.pi - self.margin),
            torch.zeros_like(one_hot_labels),one_hot_labels)


        final_theta = torch.where(selected_labels.bool(),
                                    theta + self.margin,
                                    theta)

        return torch.cos(final_theta)

    def calculate_arccos2(self, cos_sim, labels=None, slabels=None):
        theta = torch.acos(torch.clamp(cos_sim, -1, 1))

        if labels is None:
            labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
        num_classes = labels.max().item() + 1
        one_hot_labels = F.one_hot(labels, num_classes)

        selected_labels = torch.where(
            torch.gt(theta, self.margin),
            torch.ones_like(one_hot_labels),one_hot_labels) * torch.abs(one_hot_labels - 1)

        if slabels is None:
            final_theta = torch.where(selected_labels.bool(),
                                    theta - self.margin,
                                    theta)

        else:
            final_theta = torch.where(selected_labels.bool(),
                                    theta - (1-slabels)*self.margin,
                                    theta)

        return torch.cos(final_theta)

    def forward(self, x, y, slabels=None):
        return self.calculate_arccos2(self.cos(x, y), slabels=slabels) / self.temp


class ConsistencySimilarityModule(nn.Module):
    def __init__(self, shared_dim=768, sim_dim=256):
        super().__init__()
        self.text_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.image_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.sim_classifier_dim = sim_dim * 2
        self.sim_classifier = nn.Sequential(
            nn.BatchNorm1d(self.sim_classifier_dim),
            nn.Linear(self.sim_classifier_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, text, image):
        text_aligned = self.text_aligner(text)
        image_aligned = self.image_aligner(image)
        sim_feature = torch.cat([text_aligned, image_aligned], 1)
        pred_similarity = self.sim_classifier(sim_feature)
        return text_aligned, image_aligned, pred_similarity


class KLContrastiveSimLoss(nn.Module):
    """KL divergence loss: align logits distribution to softlabel distribution."""

    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def forward(self, logits, softlabel):
        # softmax for softlabel
        sim_targets = F.softmax(softlabel / self.tau, dim=1)

        # log softmax
        logit_inputs = F.log_softmax(logits / self.tau, dim=1)

        loss = F.kl_div(logit_inputs, sim_targets, reduction='batchmean')

        return loss


class ListNet(nn.Module):
    """
    ListNet objective for ranking distillation; minimizes the cross entropy between permutation [top-1] probability distribution and ground truth obtained from teacher
    """
    def __init__(self, tau, gamma_):
        super().__init__()
        self.teacher_temp_scaled_sim = Similarity(tau / 2)
        self.student_temp_scaled_sim = Similarity(tau)
        self.gamma_ = gamma_

    def forward(self, teacher_top1_sim_pred, student_top1_sim_pred):
        p = F.log_softmax(student_top1_sim_pred.fill_diagonal_(float('-inf')), dim=-1)
        q = F.softmax(teacher_top1_sim_pred.fill_diagonal_(float('-inf')), dim=-1)
        loss = -(q*p).nansum()  / q.nansum()
        return self.gamma_ * loss



class ListMLE(nn.Module):
    """
    ListMLE objective for ranking distillation; maximizes the likelihood of the ground truth permutation (sorted indices of the ranking lists obtained from teacher).
    """
    def __init__(self, tau, gamma_):
        super().__init__()
        self.temp_scaled_sim = Similarity(tau)
        self.gamma_ = gamma_
        self.eps = 1e-7

    def forward(self, teacher_top1_sim_pred, student_top1_sim_pred, k=None):
        if k is not None:
            sublist_indices = (student_top1_sim_pred.shape[1] * torch.rand(size=k)).long()
            y_pred = student_top1_sim_pred[:, sublist_indices]
            y_true = teacher_top1_sim_pred[:, sublist_indices]

        y_pred = student_top1_sim_pred
        y_true = teacher_top1_sim_pred

        # shuffle for randomised tie resolution
        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)
        mask = y_true_sorted == -1
        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
        preds_sorted_by_true[mask] = float('-inf')
        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
        observation_loss = torch.log(cumsums + self.eps) - preds_sorted_by_true_minus_max
        observation_loss[mask] = 0.0

        return self.gamma_ * torch.mean(torch.sum(observation_loss, dim=1))


class Divergence(nn.Module):
    """
    Jensen-Shannon divergence, used to measure ranking consistency between similarity lists obtained from examples with two different dropout masks
    """
    def __init__(self, beta_):
        super().__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.eps = 1e-7
        self.beta_ = beta_

    def forward(self, p: torch.Tensor, q: torch.Tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log().clamp(min=self.eps)
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.args = model_kargs['model_args']
        self.bert = BertModel(config)
        self.pooler = MLPLayer(config.hidden_size, config.hidden_size)
        self.init_weights()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        pooler_output = self.pooler(outputs.last_hidden_state[:, 0])

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )

class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.args = model_kargs['model_args']
        self.roberta = RobertaModel(config)
        self.pooler = MLPLayer(config.hidden_size, config.hidden_size)
        self.init_weights()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        pooler_output = self.pooler(outputs.last_hidden_state[:, 0])

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )


class ResNetVisnModel(nn.Module):
    def __init__(self, feature_dim,  proj_dim):
        super().__init__()
        self.mlp = MLPLayer(feature_dim, proj_dim) # visual features -> grounding space

    def forward(self, x):
        x = self.mlp(x)
        x = x / x.norm(2, dim=-1, keepdim=True)
        return x


class ClipVisnModel(nn.Module):
    def __init__(self, feature_dim, proj_dim):
        super().__init__()
        self.vmlp = MLPLayer(feature_dim, proj_dim)
        self.tmlp = MLPLayer(feature_dim, proj_dim)
        self.logit_scale = torch.tensor(np.log(1 / 0.05))
        self.loss_fct = nn.CrossEntropyLoss()

    def logit(self, image_features, text_features):
        device = image_features.device
        logit_scale = self.logit_scale.exp()
        logits_image_text = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_image_text.t()
        ground_truth = torch.arange(logits_image_text.size(0)).to(device)
        total_loss = (self.loss_fct(logits_image_text, ground_truth) + self.loss_fct(logits_per_text, ground_truth)) / 2
        return total_loss

    def forward(self, visn_feat, text_feat):
        visn_feat = self.vmlp(visn_feat)
        visn_feat = visn_feat / visn_feat.norm(2, dim=-1, keepdim=True)
        text_feat = self.tmlp(text_feat)
        text_feat = text_feat / text_feat.norm(2, dim=-1, keepdim=True)
        return visn_feat, text_feat, None


class ClipVisnModelAlignment(nn.Module):
    def __init__(self, feature_dim, proj_dim):
        super().__init__()
        self.logit_scale = torch.tensor(np.log(1 / 0.05))
        self.loss_fct = nn.CrossEntropyLoss()

    def logit(self, image_features, text_features):
        device = image_features.device
        logit_scale = self.logit_scale.exp()
        logits_image_text = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_image_text.t()
        ground_truth = torch.arange(logits_image_text.size(0)).to(device)
        total_loss = (self.loss_fct(logits_image_text, ground_truth) + self.loss_fct(logits_per_text, ground_truth)) / 2
        return total_loss

    def forward(self, visn_feat, text_feat, text_grounding, image_grounding):
        self.vmlp = text_grounding
        self.tmlp = image_grounding

        visn_feat = self.vmlp(visn_feat)
        visn_feat = visn_feat / visn_feat.norm(2, dim=-1, keepdim=True)
        text_feat = self.tmlp(text_feat)
        text_feat = text_feat / text_feat.norm(2, dim=-1, keepdim=True)
        return visn_feat, text_feat, None

class ImageGrounding(nn.Module):
    def __init__(self, feature_dim, proj_dim):
        super().__init__()
        self.vmlp = MLPLayer(feature_dim, proj_dim)

    def forward(self, visn_feat):
        visn_feat = self.vmlp(visn_feat)
        visn_feat = visn_feat / visn_feat.norm(2, dim=-1, keepdim=True)

        return visn_feat

class TextGrounding(nn.Module):
    def __init__(self, feature_dim, proj_dim):
        super().__init__()
        self.tmlp = MLPLayer(feature_dim, proj_dim)


    def forward(self, text_feat):
        text_feat = self.tmlp(text_feat)
        text_feat = text_feat / text_feat.norm(2, dim=-1, keepdim=True)

        return text_feat


class ImageGroundingAlignment(nn.Module):
    def __init__(self, shared_dim=768, sim_dim=256):
        super().__init__()
        self.image_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
    def forward(self, image):
        visn_feat = self.image_aligner(image)
        visn_feat = visn_feat / visn_feat.norm(2, dim=-1, keepdim=True)
        return visn_feat

class TextGroundingAlignment(nn.Module):
    def __init__(self, shared_dim=768, sim_dim=256):
        super().__init__()
        self.text_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
    def forward(self, text):
        text_feat = self.text_aligner(text)
        text_feat = text_feat / text_feat.norm(2, dim=-1, keepdim=True)
        return text_feat



class ConsistencySimilarityModuleAlignment(nn.Module):
    def __init__(self, text_aligner, image_aligner, sim_dim=256):
        super().__init__()
        self.text_aligner = text_aligner
        self.image_aligner = image_aligner

        self.sim_classifier_dim = sim_dim * 2
        self.sim_classifier = nn.Sequential(
            nn.BatchNorm1d(self.sim_classifier_dim),
            nn.Linear(self.sim_classifier_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, text, image):
        text_aligned = self.text_aligner(text)
        image_aligned = self.image_aligner(image)
        sim_feature = torch.cat([text_aligned, image_aligned], 1)
        pred_similarity = self.sim_classifier(sim_feature)
        return text_aligned, image_aligned, pred_similarity


def prepare_data(image, label):
    nr_index = list(range(len(label)))
    if len(nr_index) < 2:
        nr_index.append(np.random.randint(len(label)))
        nr_index.append(np.random.randint(len(label)))

    image_nr = image[nr_index]

    matched_image = image_nr.clone()
    unmatched_image = image_nr.clone().roll(shifts=5, dims=0)

    return matched_image, unmatched_image


class DALR(nn.Module):
    def __init__(self, lang_model, visn_model, teacher_model_first, teacher_model_second=None, args=None):
        super().__init__()
        self.args = args
        self.lang_model = lang_model
        self.visn_model = visn_model
        self.teacher_model_first = teacher_model_first
        self.teacher_model_second = teacher_model_second

        self.grounding_image = ImageGroundingAlignment(args.hidden_size, args.proj_dim)
        self.grounding_text = MLPLayer(args.hidden_size, args.proj_dim)

        self.sim = ArcSimilarity(temp=self.args.temp, margin=args.margin1)
        self.sim_vl = ArcSimilarity(temp=self.args.temp_vl, margin=args.margin2)
        self.cos_sim = Similarity(temp=self.args.temp)
        self.consistency = ConsistencySimilarityModule()
        self.loss_func_similarity = torch.nn.CosineEmbeddingLoss(margin=0.2)
        self.kl_loss = KLContrastiveSimLoss(tau=0.5)

        if self.args.distillation_loss == "listnet":
            self.distillation_loss_fct = ListNet(self.args.tau2, self.args.gamma_)
        elif self.args.distillation_loss == "listmle":
            self.distillation_loss_fct = ListMLE(self.args.tau2, self.args.gamma_)
        else:
            raise NotImplementedError

        self.loss_fct = nn.CrossEntropyLoss()
        self.div = Divergence(beta_=self.args.beta_)

        self.using_threshhold = args.using_threshhold
        if self.using_threshhold:
            print("USING THRESHOLD")

    def KLContrastiveSimLoss(self, logits, softlabel, tau, softlabel_tau, use_loss="kl"):
        """
        KL divergence loss
        make logits and softlabel have the same distribution
        logits to softlabel
        """
        # softmax for softlabel
        sim_targets = F.softmax(softlabel / softlabel_tau, dim=1)

        # log softmax
        logit_inputs = F.log_softmax(logits / tau, dim=1)

        if use_loss == "kl":
            # KL divergence
            loss = F.kl_div(logit_inputs, sim_targets, reduction='batchmean')
        elif use_loss == "contrastive":
            loss = -torch.sum(logit_inputs * sim_targets, dim=1).mean()

        else:
            raise ValueError("loss mode error")

        return loss


    def in_batch_g2l_loss(self, l, m, temp, attention_mask=None):
            m = m.unsqueeze(1)
            N, n_locals, dim = l.size()
            l_n = l.reshape(-1, dim) # (N * n_locals) * d
            m_n = m.reshape(-1, dim) # N * d

            # Inner product for positive samples. Outer product for negative. We need to do it this way
            # for the multiclass loss. For the outer product, we want a N x N x n_locals x 1 tensor.
            u_p = torch.matmul(l, m.permute(0,2,1)).unsqueeze(2) / temp # N * n_locals * 1 * 1

            # if l comes from text, then attention_mask is not None
            if attention_mask is not None:
                temp_mask = attention_mask.unsqueeze(2).unsqueeze(3)
                u_p = (temp_mask * u_p) + (10000. * (1-temp_mask))

            u_n = torch.mm(m_n, l_n.t()) / temp
            u_n = u_n.reshape(N, 1, N, n_locals).permute(0, 2, 3, 1) # N x N x n_locals x 1

            # We need to mask the diagonal part of the negative tensor.
            mask = torch.eye(N)[:, :, None, None].to(l.device) # N*N*1*1
            n_mask = 1 - mask

            # Masking is done by shifting the diagonal before exp.
            u_n = (n_mask * u_n) - (10000. * (1 - n_mask))  # mask out "self" examples
            # if l comes from test, we mask out the padding tokens
            if attention_mask is not None:
                temp_mask = attention_mask.unsqueeze(0).unsqueeze(3).expand(N, -1, -1, -1)
                u_n = (temp_mask * u_n) - (10000. * (1-temp_mask))

            u_n = u_n.reshape(N, N * n_locals, 1).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)

            # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
            pred_lgt = torch.cat([u_p, u_n], dim=2)
            pred_log = F.log_softmax(pred_lgt, dim=2)

            # The positive score is the first element of the log softmax.
            if attention_mask is not None:
                loss = (torch.sum(-pred_log[:, :, 0].squeeze(), dim=1) / torch.sum(attention_mask, dim=1)).mean()
            else:
                loss = -pred_log[:, :, 0].mean()

            return loss


    def forward(self, batch):
        lang_output = self.lang_model(input_ids=batch['input_ids'],
                                      attention_mask=batch['attention_mask'],
                                      token_type_ids=batch['token_type_ids'] if 'position_ids' in batch.keys() else None,
                                      position_ids=batch['position_ids'] if 'position_ids' in batch.keys() else None)

        batch_size = batch['input_ids'].size(0)
        num_sent = batch['input_ids'].size(1)

        # [bs*2, hidden] -> [bs, 2, hidden]
        lang_pooled_output = lang_output.last_hidden_state[:, 0].view((batch_size, num_sent, -1))
        lang_projection = lang_output.pooler_output.view((batch_size, num_sent, -1))  # [bs, 2,  hidden],  output of additional MLP layer

        return lang_pooled_output, lang_projection

    def compute_loss(self, batch, cal_inter=False):
        l_pool, l_proj = self.forward(batch)
        self.hidden_size = l_proj.size(-1)

        z1, z2 = l_proj[:, 0], l_proj[:, 1]  # (bs, hidden)
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))  # (bs, bs)

        labels = torch.arange(cos_sim.size(0)).long().to(self.args.device)
        loss = self.loss_fct(cos_sim, labels)

        if not cal_inter:
            return loss

        else:
            # Consistency Learning
            image = batch['img']

            matched_image, unmatched_image = prepare_data(image, labels)
            text_aligned_match, image_aligned_match, pred_similarity_match = self.consistency(z1, matched_image)
            text_aligned_unmatch, image_aligned_unmatch, pred_similarity_unmatch = self.consistency(z1, unmatched_image)

            # 1:positive/match  0:negative/unmatch
            similarity_label_1 = torch.cat([torch.ones(pred_similarity_match.shape[0]), -1 * torch.ones(pred_similarity_unmatch.shape[0])], dim=0).to(self.args.device)
            text_aligned_4_task1 = torch.cat([text_aligned_match, text_aligned_unmatch], dim=0) # N*64 cat N*64 -> 2N * 64
            image_aligned_4_task1 = torch.cat([image_aligned_match, image_aligned_unmatch], dim=0)
            loss_consistency = self.loss_func_similarity(text_aligned_4_task1, image_aligned_4_task1, similarity_label_1)

            cross_modal_loss, intra_modal_loss, contrastive_loss = torch.tensor(0.0, device=self.args.device), torch.tensor(
                0.0, device=self.args.device), torch.tensor(0.0, device=self.args.device)

            # Knowledge Distillation
            with torch.no_grad():
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]

                batch_size = batch['input_ids'].size(0)
                num_sent = batch['input_ids'].size(1)

                input_ids = input_ids.view((-1, input_ids.size(-1)))
                attention_mask = attention_mask.view((-1, attention_mask.size(-1)))

                teacher_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

                if self.teacher_model_second is None:
                    # Single teacher
                    embeddings, embeddings_local = self.teacher_model_first.encode(teacher_inputs, device=self.args.device)
                    embeddings = embeddings.view((batch_size, num_sent, -1))

                    z1T, z2T = embeddings[:, 0], embeddings[:, 1]

                    if self.args.fp16:
                        z1T = z1T.to(torch.float16)
                        z2T = z2T.to(torch.float16)

                    cos = nn.CosineSimilarity(dim=-1)

                    teacher_top1_sim_pred = cos(z1T.unsqueeze(1), z2T.unsqueeze(0)) / self.args.tau2
                    teacher_text_features = z1T

                else:
                    # Weighted average of two teachers
                    embeddings1, embeddings1_local = self.teacher_model_first.encode(teacher_inputs, device=self.args.device)
                    embeddings2, embeddings2_local = self.teacher_model_second.encode(teacher_inputs, device=self.args.device)
                    embeddings1 = embeddings1.view((batch_size, num_sent, -1))
                    embeddings2 = embeddings2.view((batch_size, num_sent, -1))
                    first_teacher_z1, first_teacher_z2 = embeddings1[:, 0], embeddings1[:, 1]
                    second_teacher_z1, second_teacher_z2 = embeddings2[:, 0], embeddings2[:, 1]

                    if self.args.fp16:
                        first_teacher_z1 = first_teacher_z1.to(torch.float16)
                        first_teacher_z2 = first_teacher_z2.to(torch.float16)
                        second_teacher_z1 = second_teacher_z1.to(torch.float16)
                        second_teacher_z2 = second_teacher_z2.to(torch.float16)

                    cos = nn.CosineSimilarity(dim=-1)
                    first_teacher_top1_sim = cos(first_teacher_z1.unsqueeze(1), first_teacher_z2.unsqueeze(0)) / self.args.tau2
                    second_teacher_top1_sim = cos(second_teacher_z1.unsqueeze(1), second_teacher_z2.unsqueeze(0)) / self.args.tau2

                    teacher_top1_sim_pred = (self.args.alpha_ * first_teacher_top1_sim) + ((1.0 - self.args.alpha_) * second_teacher_top1_sim)
                    teacher_text_features = (self.args.alpha_ * first_teacher_z1) + ((1.0 - self.args.alpha_) * second_teacher_z1)
                    embeddings_local = (self.args.alpha_ * embeddings1_local) + ((1.0 - self.args.alpha_) * embeddings2_local)


            student_top1_sim_pred = cos_sim.clone()
            embeddings_local = embeddings_local.view((batch_size, num_sent, -1, self.hidden_size))
            text_feats_local1, text_feats_local2 = embeddings_local[:, 0], embeddings_local[:, 1]
            text_feats_local1 = text_feats_local1 / text_feats_local1.norm(dim=1, keepdim=True)
            text_feats_local2 = text_feats_local2 / text_feats_local2.norm(dim=1, keepdim=True)

            attention_mask = batch['attention_mask'].view((batch_size, num_sent, -1))
            attention_mask_1 = attention_mask[:, 0]
            attention_mask_2 = attention_mask[:, 1]

            loss_t2t_l1 = self.in_batch_g2l_loss(text_feats_local1, z1, self.args.temp, attention_mask_1[:, 1:])
            loss_t2t_l2 = self.in_batch_g2l_loss(text_feats_local2, z1, self.args.temp, attention_mask_2[:, 1:])
            loss_g2l = (loss_t2t_l1 + loss_t2t_l2) / 2

            rank_loss = self.distillation_loss_fct(teacher_top1_sim_pred.to(self.args.device), student_top1_sim_pred)
            rank_loss += loss_g2l

            vis_feats_t = batch['img'] / batch['img'].norm(2, dim=-1, keepdim=True)
            text_feats_t = teacher_text_features / teacher_text_features.norm(2, dim=-1, keepdim=True)
            logits_per_image = vis_feats_t @ vis_feats_t.t()
            logits_per_text = text_feats_t @ text_feats_t.t()
            cos_sim_text_image = self.sim(z1.unsqueeze(1), vis_feats_t.unsqueeze(0))

            cross_modal_alignment_loss = self.KLContrastiveSimLoss(cos_sim_text_image.t(), logits_per_image, 0.45, 0.5, use_loss="kl")
            cross_modal_alignment_loss += self.KLContrastiveSimLoss(cos_sim_text_image, logits_per_text, 0.45, 0.5, use_loss="kl")
            cross_modal_alignment_loss /= 2.0

            z1_z2_cos = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))

            v, _, _ = self.visn_model(batch['img'], batch['clip_text_feat'])
            l2v_proj = self.grounding_text(l_pool)
            l2v_proj = l2v_proj / l2v_proj.norm(2, dim=-1, keepdim=True)

            p1, p2 = l2v_proj[:, 0], l2v_proj[:, 1]  # (bs, proj)
            cos_sim_p0_v = self.sim_vl(p1.unsqueeze(1), v.unsqueeze(0), slabels=logits_per_image)  # (bs, bs)
            cos_sim_p1_v = self.sim_vl(p2.unsqueeze(1), v.unsqueeze(0), slabels=logits_per_image)

            cross_loss = (self.loss_fct(cos_sim_p0_v, labels) + self.loss_fct(cos_sim_p1_v, labels)) / 2
            cross_modal_loss = 0.01 * loss_consistency + cross_modal_alignment_loss + cross_loss

            intra_modal_alignment_loss = self.KLContrastiveSimLoss(z1_z2_cos, logits_per_text, 0.45, 0.5, use_loss="kl")

            intra_modal_loss = intra_modal_alignment_loss + rank_loss

            return loss, cross_modal_loss, intra_modal_loss
