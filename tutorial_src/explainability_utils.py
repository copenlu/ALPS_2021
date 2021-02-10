from typing import Any, List

import numpy as np
import torch
from captum.attr import LimeBase, visualization
from tqdm.notebook import tqdm


def get_embeddings_input_captum(model: torch.nn.Module,
                                model_type: str,
                                batch: List[Any],
                                pad_token_id: int = None):
    token_ids = batch[0]
    sequence_lengths = batch[-1]

    if model_type == 'transformer':
        input_embeddings = model.transformer.bert.embeddings(token_ids)
    else:
        input_embeddings = model.embedding(token_ids)
    if model_type == 'lstm':
        additional_forward_args = (sequence_lengths,)
    elif model_type == 'cnn':
        additional_forward_args = None
    elif model_type == 'transformer':
        additional_forward_args = (token_ids != pad_token_id,)
    return input_embeddings, additional_forward_args


def summarize_attributions(attributions, type='mean'):
    if type == 'none':
        return attributions
    elif type == 'mean':
        attributions = attributions.mean(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
    elif type == 'l2':
        attributions = attributions.norm(p=1, dim=-1).squeeze(0)
    return attributions


class GradientBasedVisualizer:
    """
    Source: https://captum.ai/tutorials/IMDB_TorchText_Interpret
    """

    def __init__(self, collate_fn, tokenizer, ablator):
        self.vis_data_records_ig = []
        self.label_idx = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.softmax = torch.nn.Softmax(dim=1)
        self.collate_fn = collate_fn
        self.tokenizer = tokenizer
        self.ablator = ablator

    def add_attributions_to_visualizer(self, attributions, text, pred, pred_ind,
                                       label, delta, target):
        attributions = attributions.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()

        # storing couple samples in an array for visualization purposes
        self.vis_data_records_ig.append(visualization.VisualizationDataRecord(
            attributions,
            pred,
            self.label_idx[pred_ind],
            self.label_idx[label],
            self.label_idx[target],
            attributions.sum(),
            text,
            delta))

    def interpret_sentence(self, model, model_type, sentence, target=0):
        model.zero_grad()
        label = sentence[1]
        batch = self.collate_fn([sentence])
        input_, additional_args = get_embeddings_input_captum(
            model,
            model_type,
            batch,
            self.tokenizer.pad_token_id)

        text = self.tokenizer.convert_ids_to_tokens(
            batch[0].squeeze().detach().cpu().numpy().tolist())

        # predict
        pred = self.softmax(
            model(input_, additional_args[0] if additional_args else None)).detach().cpu().numpy().tolist()
        pred_ind = np.argmax(pred, axis=1).squeeze(-1).tolist()
        pred = [p[idx] for p, idx in zip(pred, [pred_ind])]

        # compute attributions
        attributions = self.ablator.attribute(input_,
                                                     additional_forward_args=
                                                     additional_args,
                                                     target=target)

        self.add_attributions_to_visualizer(attributions,
                                            text,
                                            pred[0],
                                            pred_ind,
                                            label,
                                            None,
                                            target)

    def visualize(self):
        visualization.visualize_text(self.vis_data_records_ig)
        self.vis_data_records_ig = []


def attribute_predict(collate_fn, model_args, dataset, attribution_method,
                      model, target):
    predicted_logits, attributions, token_ids_all, true_target = [], [], [], []

    dl = torch.utils.data.DataLoader(batch_size=model_args.batch_size,
                                     dataset=dataset,
                                     collate_fn=collate_fn)

    for batch in tqdm(dl):
        input_embeds, additional_args = get_embeddings_input_captum(
            model,
            model_args.model,
            batch,
            0)

        token_ids = batch[0]

        if isinstance(attribution_method, LimeBase):
            inputs = token_ids
            instance_attribution = attribution_method.attribute(inputs,
                                                                n_perturb_samples=100,
                                                                additional_forward_args=additional_args,
                                                                target=target)
        else:
            inputs = input_embeds
            instance_attribution = attribution_method.attribute(inputs,
                                                                additional_forward_args=additional_args,
                                                                target=target)
            instance_attribution = summarize_attributions(instance_attribution,
                                                          type='mean').detach().cpu()

        predicted_logits += model(inputs, additional_args[
            0] if additional_args else None).detach().cpu().numpy().tolist()
        attributions += instance_attribution
        token_ids_all += token_ids.detach().cpu().numpy().tolist()
        true_target += batch[1].detach().cpu().numpy().tolist()

    return predicted_logits, attributions, token_ids_all, true_target
