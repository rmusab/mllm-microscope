import os
import json
from tqdm import tqdm

import numpy as np
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from huggingface_hub import hf_hub_download
from models import CLIPVisionTower
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE


class LlmMicroscope(object):

    
    def __init__(self, device="cuda:0"):
        self.device = device


    def get_est_svd(self, X, Y):
        """
        X -- torch tensor with shape [n_samples, dim]
        Y -- torch tensor with shape [n_samples, dim]
    
        Approximates Y matrix with linear transformation Y = XA
        """
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        A_estimation = Vh.T * (1 / S)[None, ...] @ U.T @ Y # Y=XA
        Y_est =  X @ A_estimation
        return Y_est
    
    
    def procrustes_similarity(self, x, y, dtype=torch.float64):
        """
        x -- torch tensor with shape [n_samples, dim]
        y -- torch tensor with shape [n_samples, dim]
        device -- the device to perform calculations on
        """
        with torch.no_grad():
            # Use CPU for numerical stability
            x = x.to("cpu", dtype=dtype)
            y = y.to("cpu", dtype=dtype)
            
            X = x - x.mean(dim=0, keepdim=True)
            Y = y - y.mean(dim=0, keepdim=True)
        
            X = X / X.norm()
            Y = Y / Y.norm()
        
            Y_estimation = self.get_est_svd(X, Y)
        
            y_error = (Y_estimation - Y).square().sum()
            sim = float(1 - y_error)
        return sim
    
    
    def procrustes_similarity_centered(self, x, y0, dtype=torch.float64):
        """
        x -- torch tensor with shape [n_samples, dim]
        y0 -- torch tensor with shape [n_samples, dim]
        device -- the device to perform calculations on
        """
        with torch.no_grad():
            # Use CPU for numerical stability
            x = x.to("cpu", dtype=dtype)
            y0 = y0.to("cpu", dtype=dtype)
            y = y0 - x
            
            X = x - x.mean(dim=0, keepdim=True)
            Y = y - y.mean(dim=0, keepdim=True)
        
            X = X / X.norm()
            Y = Y / Y.norm()
        
            Y_estimation = self.get_est_svd(X, Y)
        
            y_error = (Y_estimation - Y).square().sum()
            sim = float(1 - y_error)
        return sim
    
    
    def intrinsic_dimension(self, emb, debug=False, reduction_factor=5):
        """
        emb: n x dim torch tensor
        """
        with torch.no_grad():
            eps = 1e-8
            embeddings = emb.to(self.device, torch.float64)
            embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
            avg_len = (embeddings * embeddings).sum(dim=1).sqrt().mean()
            embeddings = embeddings / avg_len
    
            r1 = []
            r2 = []
            n = len(embeddings)
            for i in range(n):
                dsts = torch.nn.functional.pairwise_distance(
                    embeddings[i, None, :],
                    embeddings[None, :, :],
                    eps=0
                )[0]
                dsts = torch.cat([dsts[:i], dsts[i + 1:]])
                r1.append(torch.kthvalue(dsts, k=1)[0])
                r2.append(torch.kthvalue(dsts, k=2)[0])
            
            r1 = torch.tensor(r1, device=self.device)
            r2 = torch.tensor(r2, device=self.device)
            bad_cases = (r1 < eps)
            r1[bad_cases] = eps
            r2[bad_cases] = eps
            mu = r2 / r1
            mu[bad_cases] = -1
    
            mu, ind = torch.sort(mu)
            all_mu = mu.clone().cpu().detach()
            useless_items = int((mu <= 1 + eps).sum())
            mu = mu[useless_items:]
            n = mu.shape[0]
            if debug:
                print('Removed points: ', useless_items)
                plt.plot(mu.cpu().detach().numpy())
                plt.show()
    
            f_emp = torch.arange(1 + useless_items, n + 1 + useless_items, device=self.device) / (n + useless_items)
            num_dots_to_use = min(n // reduction_factor, n - 1)
    
            mu_log = torch.log(mu)[:num_dots_to_use]
            dist_log = -torch.log(1 - f_emp)[:num_dots_to_use]
    
            if debug:
                print('Regression points:', len(mu_log))
                plt.scatter(mu_log.cpu().detach().numpy(), dist_log.cpu().detach().numpy(), marker='.')
                plt.show()
    
            dim = float((mu_log * dist_log).sum() / (mu_log * mu_log).sum())
    
            if debug:
                print('Dim: ', dim)
        return float(dim) # , all_mu
    
    
    def calculate_anisotropy_torch(self, emb):
        """
        Calculate the anisotropy of a set of embeddings.
    
        Parameters:
        emb: torch tensor of shape (n_samples, n_features) representing the embeddings.
    
        Returns:
        float: The anisotropy value.
        """
        emb = emb.to(self.device)
        embeddings = emb - emb.mean(dim=0, keepdim=True)
        U, S, Vh = torch.linalg.svd(embeddings, full_matrices=False)
        cov_eigenvalues = (S * S) / (embeddings.shape[0] - 1)
        
        anisotropy = float(cov_eigenvalues.max() / cov_eigenvalues.sum())
        return anisotropy


class MllmMicroscope(object):

    
    MODEL_NAMES = [
        "LLaVA-NeXT",
        "OmniFusion",
    ]
    OMNIFUSION_PROMPT = "This is a dialog with AI assistant.\n"

    
    def __init__(self, device="cuda:0"):
        self.device = device
        self.llm_microscope = LlmMicroscope(device=self.device)
        
        # Dictionary to map model names to initialization functions
        self.model_initializers = {
            "LLaVA-NeXT": self._initialize_llava_next,
            "OmniFusion": self._initialize_omnifusion
        }
        self.model_initialized = {
            "LLaVA-NeXT": False,
            "OmniFusion": False
        }

        # Dictionary for storing intermediate embeddings
        self.model_layerwise_embs = {
            "LLaVA-NeXT": {},
            "OmniFusion": {}
        }

        # Dictionary storing the analysis results for the models
        self.all_results = {
            'procrustes_similarity_text': {},
            'procrustes_similarity_image': {},
            'procrustes_similarity_centered_text': {},
            'procrustes_similarity_centered_image': {},
            'intrinsic_dimension_text': {},
            'intrinsic_dimension_image': {},
            'intrinsic_dimension_joint_text_image': {},
            'anisotropy_text': {},
            'anisotropy_image': {}
        }


    def _initialize_llava_next(self):
        if not self.model_initialized["LLaVA-NeXT"]:
            print("Initializing the LLaVA-NeXT model...")
            # Initialize LLaVA-NeXT model and processor
            self.llava_next_processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
            self.llava_next_model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
            self.llava_next_model.to(self.device)
            
            # Extract the image token index
            self.image_token_index = self.llava_next_model.config.image_token_index

            self.model_initialized["LLaVA-NeXT"] = True

    
    def _initialize_omnifusion(self):
        if not self.model_initialized["OmniFusion"]:
            print("Initializing the OmniFusion model...")
            # Loading some sources of the projection adapter and image encoder
            hf_hub_download(repo_id="AIRI-Institute/OmniFusion", filename="models.py", local_dir='./')
            
            self.omnifusion_tokenizer = AutoTokenizer.from_pretrained("AIRI-Institute/OmniFusion", subfolder="OmniMistral-v1_1/tokenizer", use_fast=False)
            self.omnifusion_model = AutoModelForCausalLM.from_pretrained("AIRI-Institute/OmniFusion", subfolder="OmniMistral-v1_1/tuned-model", torch_dtype=torch.bfloat16, device_map=self.device)
            
            hf_hub_download(repo_id="AIRI-Institute/OmniFusion", filename="OmniMistral-v1_1/projection.pt", local_dir='./')
            hf_hub_download(repo_id="AIRI-Institute/OmniFusion", filename="OmniMistral-v1_1/special_embeddings.pt", local_dir='./')
            self.omnifusion_projection = torch.load("OmniMistral-v1_1/projection.pt", map_location=self.device)
            self.omnifusion_special_embs = torch.load("OmniMistral-v1_1/special_embeddings.pt", map_location=self.device)
            
            self.clip = CLIPVisionTower("openai/clip-vit-large-patch14-336")
            self.clip.load_model()
            self.clip = self.clip.to(device=self.device, dtype=torch.bfloat16)

            self.model_initialized["OmniFusion"] = True


    def transform_to_layerwise_dict(self, model_embeddings):
        layer_2_text_embs_bag, layer_2_image_embs_bag = {}, {}
        layer_num = 0
        for example_output in model_embeddings:
            text_token_embs = example_output['text_token_embeddings']
            image_token_embs = example_output['image_token_embeddings']
            if image_token_embs is not None:  # The current example contains image tokens
                assert text_token_embs.shape[0] == image_token_embs.shape[0]
            layer_num = text_token_embs.shape[0]
            for i in range(layer_num):
                if i in layer_2_text_embs_bag:
                    layer_2_text_embs_bag[i] += list(text_token_embs[i, :, :])
                else:
                    layer_2_text_embs_bag[i] = list(text_token_embs[i, :, :])
                if image_token_embs is not None:
                    if i in layer_2_image_embs_bag:
                        layer_2_image_embs_bag[i] += list(image_token_embs[i, :, :])
                    else:
                        layer_2_image_embs_bag[i] = list(image_token_embs[i, :, :])
        for i in range(layer_num):
            layer_2_text_embs_bag[i] = torch.stack(layer_2_text_embs_bag[i]).to(torch.float32)
            if i in layer_2_image_embs_bag:
                layer_2_image_embs_bag[i] = torch.stack(layer_2_image_embs_bag[i]).to(torch.float32)
        return layer_2_text_embs_bag, layer_2_image_embs_bag


    def extract_llava_next_embeddings(self, text, image=None):
        # Prepare the prompt
        if image:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text},
                    ],
                },
            ]
        else:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                },
            ]
        prompt = self.llava_next_processor.apply_chat_template(conversation, add_generation_prompt=True)
        print(prompt)
        inputs = self.llava_next_processor(prompt, image, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids'][0, :].tolist()
        
        outputs = self.llava_next_model(**inputs, output_hidden_states=True)
        
        text_token_embeddings = []
        image_token_embeddings = []
    
        # The current example contains an image
        if "pixel_values" in inputs:
            # Extract the positions of the text and image tokens for all layers
            image_token_pos = input_ids.index(self.image_token_index)
            text_tokens_range = list(range(image_token_pos)) + list(range(-1, -(len(input_ids) - image_token_pos), -1))
            layer_total_token_num = outputs.hidden_states[-1].shape[1]
            image_tokens_range = list(range(image_token_pos, layer_total_token_num - (len(input_ids) - image_token_pos) + 1))
        
            # Extract the embeddings from the 32 decoder layers of the language model
            for layer in outputs.hidden_states:
                layer_embeddings = layer.detach().cpu()
                layer_embeddings = layer_embeddings[0, :]  # There is only one sequence in a batch
                text_token_embeddings.append(layer_embeddings[text_tokens_range, :])
                image_token_embeddings.append(layer_embeddings[image_tokens_range, :])
        else:  # The current example is text-only
            # Extract the embeddings from the 32 decoder layers of the language model
            for layer in outputs.hidden_states:
                layer_embeddings = layer.detach().cpu()
                layer_embeddings = layer_embeddings[0, :]  # There is only one sequence in a batch
                text_token_embeddings.append(layer_embeddings)
    
        text_token_embeddings = torch.stack(text_token_embeddings)
        if "pixel_values" in inputs:
            image_token_embeddings = torch.stack(image_token_embeddings)
        else:
            image_token_embeddings = None
        
        return text_token_embeddings, image_token_embeddings


    def extract_omnifusion_embeddings(self, text, image=None):
        # Define a function to register hooks to extract intermediate embeddings
        def get_hook(name, storage_dict):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    storage_dict[name] = output[0].detach()  # Assuming the tensor is the first element of the tuple
                else:
                    storage_dict[name] = output.detach()
            return hook
        
        storage_dict = {}  # Dictionary for storing hook values
        hooks = []
    
        # Register hooks for each transformer layer
        for i, layer in enumerate(self.omnifusion_model.model.layers):
            hooks.append(layer.register_forward_hook(get_hook(f"layer_{i}", storage_dict)))
        
        with torch.no_grad():
            if image:
                image_features = self.clip.image_processor(image, return_tensors='pt')
                image_embedding = self.clip(image_features['pixel_values']).to(device=self.device, dtype=torch.bfloat16)
                projected_vision_embeddings = self.omnifusion_projection(image_embedding).to(device=self.device, dtype=torch.bfloat16)
            
            prompt_ids = self.omnifusion_tokenizer.encode(f"{self.OMNIFUSION_PROMPT}", add_special_tokens=False, return_tensors="pt").to(device=self.device)
            print(f"<image>\n{text}" if image else text)
            question_ids = self.omnifusion_tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(device=self.device)
    
            prompt_embeddings = self.omnifusion_model.model.embed_tokens(prompt_ids).to(torch.bfloat16)
            question_embeddings = self.omnifusion_model.model.embed_tokens(question_ids).to(torch.bfloat16)
    
            if image:
                embeddings = torch.cat(
                    [
                        prompt_embeddings,
                        self.omnifusion_special_embs['SOI'][None, None, ...],
                        projected_vision_embeddings,
                        self.omnifusion_special_embs['EOI'][None, None, ...],
                        self.omnifusion_special_embs['USER'][None, None, ...],
                        question_embeddings,
                        self.omnifusion_special_embs['BOT'][None, None, ...]
                    ],
                    dim=1,
                ).to(dtype=torch.bfloat16, device=self.device)
    
                # Create masks for text and image tokens
                text_mask = torch.ones(embeddings.shape[1], dtype=torch.bool)
                image_mask = torch.zeros(embeddings.shape[1], dtype=torch.bool)
        
                # Update masks for image token positions
                image_start = prompt_embeddings.size(1) + 1  # After prompt and SOI
                image_end = image_start + projected_vision_embeddings.size(1)  # Length of image embeddings
                text_mask[image_start:image_end] = 0  # Zero out image positions in text mask
                image_mask[image_start:image_end] = 1  # One out image positions in image mask
            else:
                embeddings = torch.cat(
                    [
                        prompt_embeddings,
                        self.omnifusion_special_embs['USER'][None, None, ...],
                        question_embeddings,
                        self.omnifusion_special_embs['BOT'][None, None, ...]
                    ],
                    dim=1,
                ).to(dtype=torch.bfloat16, device=self.device)
    
                # Create a mask for text tokens
                text_mask = torch.ones(embeddings.shape[1], dtype=torch.bool)
            
            # Perform the forward pass
            outputs = self.omnifusion_model(
                inputs_embeds=embeddings,
                attention_mask=None,
                return_dict=True,
                output_hidden_states=True
            )
    
        # Remove hooks after forward pass
        for hook in hooks:
            hook.remove()
        
        # Extract text and image token embeddings from intermediate layers
        text_token_embeddings = []
        image_token_embeddings = []
    
        for i in range(len(self.omnifusion_model.model.layers)):
            layer_output = storage_dict[f"layer_{i}"]
            layer_output = layer_output.to(torch.float16)  # Convert to Float16 before conversion to numpy
    
            layer_output_text = layer_output[0, text_mask, :].cpu()
            text_token_embeddings.append(layer_output_text)
    
            if image:
                layer_output_image = layer_output[0, image_mask, :].cpu()
                image_token_embeddings.append(layer_output_image)
    
        text_token_embeddings = torch.stack(text_token_embeddings)
        if image:
            image_token_embeddings = torch.stack(image_token_embeddings)
        else:
            image_token_embeddings = None
    
        return text_token_embeddings, image_token_embeddings


    def get_intermediate_embeddings(self, model_name, texts, images=None):
        if model_name not in self.MODEL_NAMES:
            raise ValueError(f"Invalid model name: {model_name}. Available models: {', '.join(self.MODEL_NAMES)}")
        # Initialize the model if it has not already been initialized
        if not self.model_initialized[model_name]:
            self.model_initializers[model_name]()

        print(f"Computing the intermediate embeddings for the {model_name} model...")
        embeddings = []

        for i in tqdm(range(len(texts))):
            text = texts[i]
            image = images[i] if images is not None else None

            if model_name == "LLaVA-NeXT":
                text_token_embeddings, image_token_embeddings = self.extract_llava_next_embeddings(text, image=image)
            else:
                text_token_embeddings, image_token_embeddings = self.extract_omnifusion_embeddings(text, image=image)
    
            entry_output_data = {
                "index": i,
                "text_token_embeddings": text_token_embeddings,
                "image_token_embeddings": image_token_embeddings
            }
            embeddings.append(entry_output_data)

        layer2text_embs, layer2image_embs = self.transform_to_layerwise_dict(embeddings)

        self.model_layerwise_embs[model_name]['layer2text_embs'] = layer2text_embs
        self.model_layerwise_embs[model_name]['layer2image_embs'] = layer2image_embs

        return self.model_layerwise_embs[model_name]

    
    def analyze_embeddings(self, model_name, ps=True, psc=True, in_d=True, an=True):
        if model_name not in self.MODEL_NAMES:
            raise ValueError(f"Invalid model name: {model_name}. Available models: {', '.join(self.MODEL_NAMES)}")
        elif not self.model_initialized[model_name] or not self.model_layerwise_embs[model_name]:
            raise ValueError(f"Model {model_name} either was not initialized or did not produce any embeddings")

        print(f"Starting tests for the model {model_name}...")
        text_embs_dict = self.model_layerwise_embs[model_name]['layer2text_embs']
        image_embs_dict = self.model_layerwise_embs[model_name]['layer2image_embs']
        
        print('Text embs shape: ', text_embs_dict[0].shape)
        if image_embs_dict:
            print('Image embs shape: ', image_embs_dict[0].shape)

        if ps:
            similarities = []
            for layer in tqdm(range(max(text_embs_dict.keys()) - 1), desc=f'{model_name} text procrustes_similarity'):
                similarities.append(self.llm_microscope.procrustes_similarity(text_embs_dict[layer], text_embs_dict[layer+1]))
            self.all_results['procrustes_similarity_text'][model_name] = similarities
        
            if image_embs_dict:
                similarities = []
                for layer in tqdm(range(max(image_embs_dict.keys()) - 1), desc=f'{model_name} image procrustes_similarity'):
                    similarities.append(self.llm_microscope.procrustes_similarity(image_embs_dict[layer], image_embs_dict[layer+1]))
                self.all_results['procrustes_similarity_image'][model_name] = similarities
        
        if psc:
            similarities = []
            for layer in tqdm(range(max(text_embs_dict.keys()) - 1), desc=f'{model_name} text procrustes_similarity_centered'):
                similarities.append(self.llm_microscope.procrustes_similarity_centered(text_embs_dict[layer], text_embs_dict[layer+1]))
            self.all_results['procrustes_similarity_centered_text'][model_name] = similarities
    
            if image_embs_dict:
                similarities = []
                for layer in tqdm(range(max(image_embs_dict.keys()) - 1), desc=f'{model_name} image procrustes_similarity_centered'):
                    similarities.append(self.llm_microscope.procrustes_similarity_centered(image_embs_dict[layer], image_embs_dict[layer+1]))
                self.all_results['procrustes_similarity_centered_image'][model_name] = similarities

        if in_d:
            dims = []
            for layer in tqdm(range(max(text_embs_dict.keys())), desc=f'{model_name} text int. dim'):
                dims.append(self.llm_microscope.intrinsic_dimension(text_embs_dict[layer]))
            self.all_results['intrinsic_dimension_text'][model_name] = dims
        
            if image_embs_dict:
                dims = []
                for layer in tqdm(range(max(image_embs_dict.keys())), desc=f'{model_name} image int. dim'):
                    dims.append(self.llm_microscope.intrinsic_dimension(image_embs_dict[layer]))
                self.all_results['intrinsic_dimension_image'][model_name] = dims
        
            if image_embs_dict:
                joint_embs_dict = {
                    i: torch.cat(
                        (text_embs_dict[i], image_embs_dict[i]), dim=0
                    )[torch.randperm(
                        text_embs_dict[i].size(0) + image_embs_dict[i].size(0)
                    )] 
                    for i in range(max(image_embs_dict.keys()))
                }
                dims = []
                for layer in tqdm(range(max(joint_embs_dict.keys())), desc=f'{model_name} joint text/image int. dim'):
                    dims.append(self.llm_microscope.intrinsic_dimension(joint_embs_dict[layer]))
                self.all_results['intrinsic_dimension_joint_text_image'][model_name] = dims

        if an:
            anisotropies = []
            for layer in tqdm(range(max(text_embs_dict.keys())), desc=f'{model_name} text anisotropy'):
                anisotropies.append(self.llm_microscope.calculate_anisotropy_torch(text_embs_dict[layer]))
            self.all_results['anisotropy_text'][model_name] = anisotropies
        
            if image_embs_dict:
                anisotropies = []
                for layer in tqdm(range(max(image_embs_dict.keys())), desc=f'{model_name} image anisotropy'):
                    anisotropies.append(self.llm_microscope.calculate_anisotropy_torch(image_embs_dict[layer]))
                self.all_results['anisotropy_image'][model_name] = anisotropies

        return self.all_results


    def plot_results(self, save=True, saving_path="./"):
        for metric_name in self.all_results.keys():
            plt.title(metric_name)
            plt.grid()
            plt.xlabel("Layer index")
            for model_name in self.all_results[metric_name].keys():
                plt.plot(self.all_results[metric_name][model_name][1:], label=model_name)
            plt.legend()
            if save:
                plt.savefig(os.path.join(saving_path, metric_name + '.png'))
            plt.show()


    def visualize_intermediate_embeddings(self, save=True, saving_path="./", method='svd_tsne'):
        if not any(self.model_layerwise_embs[model_name] for model_name in self.model_layerwise_embs):
            raise ValueError("No model has layerwise embeddings available.")
        
        max_layers = 0
        for model_name in self.model_layerwise_embs:
            if self.model_layerwise_embs[model_name]:
                layer2text_embs = self.model_layerwise_embs[model_name].get('layer2text_embs', {})
                layer2image_embs = self.model_layerwise_embs[model_name].get('layer2image_embs', {})
                max_layers = max(max_layers, max(layer2text_embs.keys(), default=0), max(layer2image_embs.keys(), default=0))
    
        for layer_id in range(max_layers + 1):
            fig, axs = plt.subplots(1, 2, figsize=(14, 7))
            axs[0].set_title(f'Layer {layer_id} Text Embeddings')
            axs[1].set_title(f'Layer {layer_id} Image Embeddings')
            plots_done = False
    
            for model_name in self.model_layerwise_embs:
                if not self.model_layerwise_embs[model_name]:
                    continue
    
                layer2text_embs = self.model_layerwise_embs[model_name].get('layer2text_embs', {})
                layer2image_embs = self.model_layerwise_embs[model_name].get('layer2image_embs', {})
    
                text_embs = layer2text_embs.get(layer_id)
                image_embs = layer2image_embs.get(layer_id)
    
                if text_embs is not None:
                    if method == 'svd':
                        svd = TruncatedSVD(n_components=2)
                        text_embs_reduced = svd.fit_transform(text_embs)
                    elif method == 'tsne':
                        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
                        text_embs_reduced = tsne.fit_transform(text_embs)
                    elif method == 'svd_tsne':
                        svd = TruncatedSVD(n_components=50)
                        text_embs_svd = svd.fit_transform(text_embs)
                        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
                        text_embs_reduced = tsne.fit_transform(text_embs_svd)
    
                    axs[0].scatter(text_embs_reduced[:, 0], text_embs_reduced[:, 1], label=f'{model_name} Text', alpha=0.6)
                    plots_done = True
    
                if image_embs is not None:
                    if method == 'svd':
                        svd = TruncatedSVD(n_components=2)
                        image_embs_reduced = svd.fit_transform(image_embs)
                    elif method == 'tsne':
                        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
                        image_embs_reduced = tsne.fit_transform(image_embs)
                    elif method == 'svd_tsne':
                        svd = TruncatedSVD(n_components=50)
                        image_embs_svd = svd.fit_transform(image_embs)
                        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
                        image_embs_reduced = tsne.fit_transform(image_embs_svd)
    
                    axs[1].scatter(image_embs_reduced[:, 0], image_embs_reduced[:, 1], label=f'{model_name} Image', alpha=0.6)
                    plots_done = True
    
            if plots_done:
                axs[0].legend()
                axs[1].legend()
    
                if save:
                    fig_path = os.path.join(saving_path, f'layer_{layer_id}.png')
                    plt.savefig(fig_path)
                plt.show()
    
                plt.close(fig)
            else:
                plt.close(fig)
                print(f"No embeddings available for layer {layer_id}")