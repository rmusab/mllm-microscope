import os
import io
import json
from tqdm import tqdm
from typing import Tuple

import numpy as np
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from huggingface_hub import hf_hub_download
from models import CLIPVisionTower
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm


class LlmMicroscope(object):

    
    def __init__(self, device="cuda:0"):
        self.device = device


    # def get_est_svd(self, X, Y):
    #     """
    #     X -- torch tensor with shape [n_samples, dim]
    #     Y -- torch tensor with shape [n_samples, dim]
    
    #     Approximates Y matrix with linear transformation Y = XA
    #     """
    #     U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    #     A_estimation = Vh.T * (1 / S)[None, ...] @ U.T @ Y # Y=XA
    #     Y_est =  X @ A_estimation
    #     return Y_est


    def get_est_svd(self, X, Y):
        """
        X -- torch tensor with shape [n_samples, dim]
        Y -- torch tensor with shape [n_samples, dim]
    
        Approximates Y matrix with linear transformation Y = XA
        """
        # Compute X^T X
        XTX = X.T @ X
        
        # Compute X^T Y
        XTY = X.T @ Y
        
        # Solve for A using torch.linalg.solve
        A_estimation = torch.linalg.solve(XTX, XTY)
        
        # Estimate Y using the linear transformation
        Y_est = X @ A_estimation
        
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


    def get_linear_errors(self, embs_analysis):
        num_layers = embs_analysis.size(0)
        
        layer2errors = {}
        for layer in range(num_layers - 1):
            X = embs_analysis[layer].to(torch.float32)
            Y = embs_analysis[layer + 1].to(torch.float32)
            
            # Y = Y - X

            X_mean = X.mean(dim=0, keepdim=True)
            Y_mean = Y.mean(dim=0, keepdim=True)
            X = X - X_mean
            Y = Y - Y_mean
            
            X_norm = X.norm()
            Y_norm = Y.norm()
            X = X / X_norm
            Y = Y / Y_norm

            Y_estimation = self.get_est_svd(X, Y)
            # print(Y.shape, Y.norm(dim=-1).shape)
            y_errors = (Y_estimation - Y).norm(dim=-1) #/ Y.norm(dim=-1)
            layer2errors[layer] = y_errors
        
        return layer2errors


    def normalize_weights(
                self, 
                weights: np.ndarray, 
                normalization_type: str, 
                log_info: str, 
                show_first: bool = True, 
                show_last: bool = True, 
                log_weights: bool = False
            ) -> Tuple[np.ndarray, str]:
        log_info += f"Normalization: {normalization_type}"
        if log_weights:
            log_info += "\nWeights before norm.:\n"
            log_info += "\n".join(str(row) for row in weights)

        if not show_first:
            weights = weights[1:]  # Remove the first layer
        if not show_last:
            weights = weights[:-1]  # Remove the last layer

        if normalization_type == "token-wise":
            if len(weights.shape) != 2 or weights.shape[1] < 2:
                log_info += "\n Cannot apply token-wise normalization to one sentence, setting global normalization\n"
                normalization_type = "global"
            else:
                for tok_idx in range(weights.shape[1]):
                    max_, min_ = weights[:, tok_idx].max(), weights[:, tok_idx].min()
                    weights[:, tok_idx] = (weights[:, tok_idx] - min_) / (max_ - min_)
                normalized_weights = weights
        if normalization_type == "sentence-wise":
            if len(weights[0]) == 1: 
                log_info += "\n Cannot apply sentence-wise normalization to one word, setting global normalization\n"
                normalization_type = "global"
            else:
                normalized_weights = []
                for layer_idx in range(len(weights)):
                    max_, min_ = weights[layer_idx].max(), weights[layer_idx].min()
                    normalized_weights.append((weights[layer_idx] - min_) / (max_ - min_))
                normalized_weights = np.array(normalized_weights) 
        if normalization_type == "global":
            max_, min_ = weights.max(), weights.min()
            normalized_weights = (weights - min_) / (max_ - min_)

        # TODO: Check source of nan's
        normalized_weights = np.nan_to_num(normalized_weights, nan=1)
        
        if log_weights:
            log_info += "\nWeights after norm.:\n"
            log_info += "\n".join(str(row) for row in normalized_weights)
            log_info += "\n\n"
        
        return normalized_weights, log_info


    def visualize_layers_text(self, all_weights, words, show_first):
        cmap = mcolors.LinearSegmentedColormap.from_list("", ["green", "yellow", "red"])
    
        def add_colored_background(ax, words, normalized_weights, y_pos, text_widths):
            number_shift = 20 / fig.dpi / fig.get_size_inches()[0] * 1.3
            x_pos = number_shift
            for word, norm_weight in zip(words, normalized_weights):
                word_text = f' {word} '
                word_width = text_widths.get(word, 0)
                if not word_width:  # If the word's width has not been measured yet
                    text = ax.text(0, 0, word_text)
                    fig.canvas.draw()
                    bbox = text.get_window_extent()
                    word_width = bbox.width / fig.dpi / fig.get_size_inches()[0] * 1.3  # Convert to figure space
                    text_widths[word] = word_width  # Store the measured width
                    text.remove()  # We remove this text artist since it was only for measurement
    
                text_obj = ax.text(x_pos, y_pos, word_text, ha='left', va='center', fontsize=12,
                                   backgroundcolor=cmap(norm_weight), fontweight='bold')
                text_obj.set_bbox(dict(facecolor=cmap(norm_weight), edgecolor='none', pad=0.5))
    
                x_pos += word_width  # Use the measured width to increment x_pos
    
            return y_pos  # return new y position after this layer
    
        text_widths = {}
        single_layer_height = 0.14  # Adjust this based on the font size and padding
        fig_height = 1.3  # len(all_weights) * single_layer_height + 2*padding
        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.axis('off')
    
        padding = 0  # Adjust the padding value as needed in percent
        # Starting y position set to just below the top of the figure
        y_pos = 1 - padding - single_layer_height
    
        # Add an additional axis for line numbers
        ax_line_numbers = fig.add_axes([0, 0, 0.05, 1], frameon=False)
        ax_line_numbers.set_xlim(0, 1)
        ax_line_numbers.set_ylim(0, 1)
        ax_line_numbers.set_xticks([])
        ax_line_numbers.set_yticks([])
    
        # Plot each layer and add line numbers
        if show_first:
            line_number = 1
        else:
            line_number = 2
        for norm_weights in all_weights:
            y_pos = add_colored_background(ax, words, norm_weights, y_pos, text_widths)
            # Add the line number
            ax_line_numbers.text(0, y_pos, str(line_number), ha='center', va='center', fontsize=12)
            line_number += 1
            # Adjust this value if lines are too close or too far apart
            y_pos -= single_layer_height
    
        plt.subplots_adjust(left=padding / fig.get_figwidth(), right=1 - padding / fig.get_figwidth(),
                            top=1 - padding / fig.get_figheight(), bottom=padding / fig.get_figheight())
    
        return fig


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

        # List for storing all extracted embeddings
        self.embeddings = []

        # Dictionary for storing stacked intermediate layerwise embeddings
        self.model_layerwise_embs = {
            "LLaVA-NeXT": {},
            "OmniFusion": {}
        }

        # Dictionary storing the analysis results for the models
        self.all_results = {
            'procrustes_similarity_text': {},
            'procrustes_similarity_image': {},
            'procrustes_similarity_text_main_flow': {},
            'procrustes_similarity_image_main_flow': {},
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

            # Print out the model architecture
            print("LLaVA-NeXT Model Architecture:")
            print(self.llava_next_model)
            
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

            # Print out the model architecture
            print("OmniFusion Model Architecture:")
            print(self.omnifusion_model)
            
            hf_hub_download(repo_id="AIRI-Institute/OmniFusion", filename="OmniMistral-v1_1/projection.pt", local_dir='./')
            hf_hub_download(repo_id="AIRI-Institute/OmniFusion", filename="OmniMistral-v1_1/special_embeddings.pt", local_dir='./')
            self.omnifusion_projection = torch.load("OmniMistral-v1_1/projection.pt", map_location=self.device)
            self.omnifusion_special_embs = torch.load("OmniMistral-v1_1/special_embeddings.pt", map_location=self.device)
            
            self.clip = CLIPVisionTower("openai/clip-vit-large-patch14-336")
            self.clip.load_model()
            self.clip = self.clip.to(device=self.device, dtype=torch.bfloat16)

            self.model_initialized["OmniFusion"] = True


    def transform_to_layerwise_dict(self, model_embeddings, max_tokens=None):
        layer_2_text_embs_bag, layer_2_image_embs_bag = {}, {}
        layer_2_text_embs_mf_bag, layer_2_image_embs_mf_bag = {}, {}
        for example_output in model_embeddings:
            text_token_embs = example_output['text_token_embeddings']
            image_token_embs = example_output['image_token_embeddings']
            text_token_embs_mf = example_output['text_token_embeddings_mf']
            image_token_embs_mf = example_output['image_token_embeddings_mf']
            if image_token_embs is not None:  # The current example contains image tokens
                assert text_token_embs.shape[0] == image_token_embs.shape[0]
            layer_num = text_token_embs.shape[0]
            for i in range(layer_num):
                if i in layer_2_text_embs_bag:
                    if max_tokens is None or (max_tokens is not None and len(layer_2_text_embs_bag[i]) < max_tokens):
                        layer_2_text_embs_bag[i] += list(text_token_embs[i, :, :])
                        layer_2_text_embs_mf_bag[i] += list(text_token_embs_mf[i, :, :])
                else:
                    layer_2_text_embs_bag[i] = list(text_token_embs[i, :, :])
                    layer_2_text_embs_mf_bag[i] = list(text_token_embs_mf[i, :, :])
                if image_token_embs is not None:
                    if i in layer_2_image_embs_bag:
                        if max_tokens is None or (max_tokens is not None and len(layer_2_image_embs_bag[i]) < max_tokens):
                            layer_2_image_embs_bag[i] += list(image_token_embs[i, :, :])
                            layer_2_image_embs_mf_bag[i] += list(image_token_embs_mf[i, :, :])
                    else:
                        layer_2_image_embs_bag[i] = list(image_token_embs[i, :, :])
                        layer_2_image_embs_mf_bag[i] = list(image_token_embs_mf[i, :, :])
        for i in range(layer_num):
            layer_2_text_embs_bag[i] = torch.stack(layer_2_text_embs_bag[i]).to(torch.float32)
            layer_2_text_embs_mf_bag[i] = torch.stack(layer_2_text_embs_mf_bag[i]).to(torch.float32)
            if i in layer_2_image_embs_bag:
                layer_2_image_embs_bag[i] = torch.stack(layer_2_image_embs_bag[i]).to(torch.float32)
                layer_2_image_embs_mf_bag[i] = torch.stack(layer_2_image_embs_mf_bag[i]).to(torch.float32)
        return layer_2_text_embs_bag, layer_2_image_embs_bag, layer_2_text_embs_mf_bag, layer_2_image_embs_mf_bag


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

        # Token embeddings across layers in the main flow
        text_token_embeddings_mf = []
        image_token_embeddings_mf = []

        # Manually compute the hidden states without residuals for each layer,
        # modelling the main flow Y = Norm(FF(Norm(ATTN(X))))
        hidden_states = outputs.hidden_states[0]  # Start with initial hidden states
        # Use new cache system
        past_key_values_cache = None  # Or appropriate Cache class instance
        if "pixel_values" in inputs:
            text_token_embeddings_mf.append(hidden_states.detach().cpu()[0, text_tokens_range, :])
            image_token_embeddings_mf.append(hidden_states.detach().cpu()[0, image_tokens_range, :])
        else:
            text_token_embeddings_mf.append(hidden_states.detach().cpu()[0, :])
        # Ensure position_ids are properly initialized
        # position_ids = inputs.get("position_ids", None)
        if "pixel_values" in inputs:
            position_ids = torch.arange(0, inputs['input_ids'].size(-1) + len(image_tokens_range) - 1, dtype=torch.long, device=self.device).unsqueeze(0)
        else:
            position_ids = torch.arange(0, inputs['input_ids'].size(-1), dtype=torch.long, device=self.device).unsqueeze(0)
        for i, layer in enumerate(self.llava_next_model.language_model.model.layers):
            # Attention part without residuals
            normed_hidden_states = layer.input_layernorm(hidden_states)
            
            # # Print out the dimensions before applying attention
            # print(f"Layer {i} - normed_hidden_states shape: {normed_hidden_states.shape}")
    
            # Attempt to perform self-attention, capturing the required tensors
            attn_output, past_key_values_cache, _ = layer.self_attn(
                normed_hidden_states, 
                position_ids=position_ids, 
                past_key_value=past_key_values_cache,
                use_cache=True
            )  # Get only the attention output and cache
    
            normed_attn_output = layer.post_attention_layernorm(attn_output)
    
            # Feedforward part without residuals
            ff_output = layer.mlp(normed_attn_output)
    
            # Update the hidden states to feed into the next layer
            hidden_states = ff_output
    
            # Collect embeddings per token
            layer_embeddings = hidden_states.detach().cpu()
            if "pixel_values" in inputs:
                text_token_embeddings_mf.append(layer_embeddings[0, text_tokens_range, :])
                image_token_embeddings_mf.append(layer_embeddings[0, image_tokens_range, :])
            else:
                text_token_embeddings_mf.append(layer_embeddings[0, :])
        
        text_token_embeddings = torch.stack(text_token_embeddings)
        text_token_embeddings_mf = torch.stack(text_token_embeddings_mf)
        if "pixel_values" in inputs:
            image_token_embeddings = torch.stack(image_token_embeddings)
            image_token_embeddings_mf = torch.stack(image_token_embeddings_mf)
        else:
            image_token_embeddings = None
            image_token_embeddings_mf = None

        words = [self.llava_next_processor.tokenizer.decode([input_id]) for input_id in input_ids]
        
        return text_token_embeddings, image_token_embeddings, text_token_embeddings_mf, image_token_embeddings_mf, words


    # def extract_omnifusion_embeddings(self, text, image=None):
    #     # Define a function to register hooks to extract intermediate embeddings
    #     def get_hook(name, storage_dict):
    #         def hook(module, input, output):
    #             if isinstance(output, tuple):
    #                 storage_dict[name] = output[0].detach()  # Assuming the tensor is the first element of the tuple
    #             else:
    #                 storage_dict[name] = output.detach()
    #         return hook
        
    #     storage_dict = {}  # Dictionary for storing hook values
    #     hooks = []
    
    #     # Register hooks for each transformer layer
    #     for i, layer in enumerate(self.omnifusion_model.model.layers):
    #         hooks.append(layer.register_forward_hook(get_hook(f"layer_{i}", storage_dict)))
        
    #     with torch.no_grad():
    #         if image:
    #             image_features = self.clip.image_processor(image, return_tensors='pt')
    #             image_embedding = self.clip(image_features['pixel_values']).to(device=self.device, dtype=torch.bfloat16)
    #             projected_vision_embeddings = self.omnifusion_projection(image_embedding).to(device=self.device, dtype=torch.bfloat16)
            
    #         prompt_ids = self.omnifusion_tokenizer.encode(f"{self.OMNIFUSION_PROMPT}", add_special_tokens=False, return_tensors="pt").to(device=self.device)
    #         print(f"<image>\n{text}" if image else text)
    #         question_ids = self.omnifusion_tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(device=self.device)
    
    #         prompt_embeddings = self.omnifusion_model.model.embed_tokens(prompt_ids).to(torch.bfloat16)
    #         question_embeddings = self.omnifusion_model.model.embed_tokens(question_ids).to(torch.bfloat16)
    
    #         if image:
    #             embeddings = torch.cat(
    #                 [
    #                     prompt_embeddings,
    #                     self.omnifusion_special_embs['SOI'][None, None, ...],
    #                     projected_vision_embeddings,
    #                     self.omnifusion_special_embs['EOI'][None, None, ...],
    #                     self.omnifusion_special_embs['USER'][None, None, ...],
    #                     question_embeddings,
    #                     self.omnifusion_special_embs['BOT'][None, None, ...]
    #                 ],
    #                 dim=1,
    #             ).to(dtype=torch.bfloat16, device=self.device)
    
    #             # Create masks for text and image tokens
    #             text_mask = torch.ones(embeddings.shape[1], dtype=torch.bool)
    #             image_mask = torch.zeros(embeddings.shape[1], dtype=torch.bool)
        
    #             # Update masks for image token positions
    #             image_start = prompt_embeddings.size(1) + 1  # After prompt and SOI
    #             image_end = image_start + projected_vision_embeddings.size(1)  # Length of image embeddings
    #             text_mask[image_start:image_end] = 0  # Zero out image positions in text mask
    #             image_mask[image_start:image_end] = 1  # One out image positions in image mask
    #         else:
    #             embeddings = torch.cat(
    #                 [
    #                     prompt_embeddings,
    #                     self.omnifusion_special_embs['USER'][None, None, ...],
    #                     question_embeddings,
    #                     self.omnifusion_special_embs['BOT'][None, None, ...]
    #                 ],
    #                 dim=1,
    #             ).to(dtype=torch.bfloat16, device=self.device)
    
    #             # Create a mask for text tokens
    #             text_mask = torch.ones(embeddings.shape[1], dtype=torch.bool)
            
    #         # Perform the forward pass
    #         outputs = self.omnifusion_model(
    #             inputs_embeds=embeddings,
    #             attention_mask=None,
    #             return_dict=True,
    #             output_hidden_states=True
    #         )
    
    #     # Remove hooks after forward pass
    #     for hook in hooks:
    #         hook.remove()
        
    #     # Extract text and image token embeddings from intermediate layers
    #     text_token_embeddings = []
    #     image_token_embeddings = []
    
    #     for i in range(len(self.omnifusion_model.model.layers)):
    #         layer_output = storage_dict[f"layer_{i}"]
    #         layer_output = layer_output.to(torch.float16)  # Convert to Float16 before conversion to numpy
    
    #         layer_output_text = layer_output[0, text_mask, :].cpu()
    #         text_token_embeddings.append(layer_output_text)
    
    #         if image:
    #             layer_output_image = layer_output[0, image_mask, :].cpu()
    #             image_token_embeddings.append(layer_output_image)
    
    #     text_token_embeddings = torch.stack(text_token_embeddings)
    #     if image:
    #         image_token_embeddings = torch.stack(image_token_embeddings)
    #     else:
    #         image_token_embeddings = None
    
    #     return text_token_embeddings, image_token_embeddings


    def extract_omnifusion_embeddings(self, text, image=None):
        # Prepare the embeddings and masks
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
    
                # Generate words list and replace image tokens with '<image>'
                words = (
                    self.omnifusion_tokenizer.convert_ids_to_tokens(prompt_ids.squeeze().tolist()) +
                    ['<SOI>', '<image>', '<EOI>', '<USER>'] +
                    self.omnifusion_tokenizer.convert_ids_to_tokens(question_ids.squeeze().tolist()) +
                    ['<BOT>']
                )
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
                
                # Generate words list directly from tokens
                words = (
                    self.omnifusion_tokenizer.convert_ids_to_tokens(prompt_ids.squeeze().tolist()) +
                    ['<USER>'] +
                    self.omnifusion_tokenizer.convert_ids_to_tokens(question_ids.squeeze().tolist()) +
                    ['<BOT>']
                )
    
            # Perform the forward pass to get initial hidden states
            outputs = self.omnifusion_model(
                inputs_embeds=embeddings,
                attention_mask=None,
                return_dict=True,
                output_hidden_states=True
            )
    
        # Initialize lists to store embeddings
        text_token_embeddings = []
        image_token_embeddings = []
        text_token_embeddings_mf = []
        image_token_embeddings_mf = []
    
        # The initial hidden states
        hidden_states = outputs.hidden_states[0]  # Start with initial hidden states
        if image:
            text_token_embeddings_mf.append(hidden_states.detach().cpu()[0, text_mask, :])
            image_token_embeddings_mf.append(hidden_states.detach().cpu()[0, image_mask, :])
        else:
            text_token_embeddings_mf.append(hidden_states.detach().cpu()[0, :])
    
        # Position IDs for handling the image and text tokens
        position_ids = torch.arange(0, embeddings.size(1), dtype=torch.long, device=self.device).unsqueeze(0)
    
        # Iterate through each layer to capture the embeddings for both residual and main flows
        past_key_values_cache = None
        for i, layer in enumerate(self.omnifusion_model.model.layers):
            # Attention part without residuals
            normed_hidden_states = layer.input_layernorm(hidden_states)
            
            attn_output, past_key_values_cache, _ = layer.self_attn(
                normed_hidden_states,
                position_ids=position_ids,
                past_key_value=past_key_values_cache,
                use_cache=True
            )
            
            normed_attn_output = layer.post_attention_layernorm(attn_output)
    
            # Feedforward part without residuals
            ff_output = layer.mlp(normed_attn_output)
    
            # Update the hidden states to feed into the next layer
            hidden_states = ff_output
    
            # Collect main flow embeddings per token
            layer_embeddings = hidden_states.detach().cpu()
            if image:
                text_token_embeddings_mf.append(layer_embeddings[0, text_mask, :])
                image_token_embeddings_mf.append(layer_embeddings[0, image_mask, :])
            else:
                text_token_embeddings_mf.append(layer_embeddings[0, :])
    
        # Extract the residual flow embeddings from the original forward pass
        for layer_hidden_state in outputs.hidden_states:
            layer_embeddings = layer_hidden_state.detach().cpu()
            if image:
                text_token_embeddings.append(layer_embeddings[0, text_mask, :])
                image_token_embeddings.append(layer_embeddings[0, image_mask, :])
            else:
                text_token_embeddings.append(layer_embeddings[0, :])
    
        # Stack the embeddings for each flow
        text_token_embeddings = torch.stack(text_token_embeddings)
        text_token_embeddings_mf = torch.stack(text_token_embeddings_mf)
        if image:
            image_token_embeddings = torch.stack(image_token_embeddings)
            image_token_embeddings_mf = torch.stack(image_token_embeddings_mf)
        else:
            image_token_embeddings = None
            image_token_embeddings_mf = None
    
        return text_token_embeddings, image_token_embeddings, text_token_embeddings_mf, image_token_embeddings_mf, words


    def get_intermediate_embeddings(self, model_name, texts, images=None, max_tokens=None):
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
                text_token_embeddings, image_token_embeddings, text_token_embeddings_mf, image_token_embeddings_mf, words = self.extract_llava_next_embeddings(text, image=image)
            else:
                text_token_embeddings, image_token_embeddings, text_token_embeddings_mf, image_token_embeddings_mf, words = self.extract_omnifusion_embeddings(text, image=image)
    
            entry_output_data = {
                "text_token_embeddings": text_token_embeddings,
                "image_token_embeddings": image_token_embeddings,
                "text_token_embeddings_mf": text_token_embeddings_mf,
                "image_token_embeddings_mf": image_token_embeddings_mf,
                "words": words
            }
            embeddings.append(entry_output_data)

        # Save the extracted embeddings
        self.embeddings = embeddings
        layer2text_embs, layer2image_embs, layer2text_embs_mf, layer2image_embs_mf = self.transform_to_layerwise_dict(embeddings, max_tokens)

        self.model_layerwise_embs[model_name]['layer2text_embs'] = layer2text_embs
        self.model_layerwise_embs[model_name]['layer2image_embs'] = layer2image_embs
        self.model_layerwise_embs[model_name]['layer2text_embs_mf'] = layer2text_embs_mf
        self.model_layerwise_embs[model_name]['layer2image_embs_mf'] = layer2image_embs_mf

        return self.model_layerwise_embs[model_name]

    
    def analyze_embeddings(self, model_name, ps=True, psmf=True, in_d=True, an=True):
        if model_name not in self.MODEL_NAMES:
            raise ValueError(f"Invalid model name: {model_name}. Available models: {', '.join(self.MODEL_NAMES)}")
        elif not self.model_initialized[model_name] or not self.model_layerwise_embs[model_name]:
            raise ValueError(f"Model {model_name} either was not initialized or did not produce any embeddings")

        print(f"Starting tests for the model {model_name}...")
        
        text_embs_dict = self.model_layerwise_embs[model_name]['layer2text_embs']
        image_embs_dict = self.model_layerwise_embs[model_name]['layer2image_embs']
        text_embs_mf_dict = self.model_layerwise_embs[model_name]['layer2text_embs_mf']
        image_embs_mf_dict = self.model_layerwise_embs[model_name]['layer2image_embs_mf']
        
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
        
        if psmf:
            similarities = []
            for layer in tqdm(range(max(text_embs_mf_dict.keys()) - 1), desc=f'{model_name} text procrustes_similarity_main_flow'):
                similarities.append(self.llm_microscope.procrustes_similarity(text_embs_mf_dict[layer], text_embs_mf_dict[layer+1]))
            self.all_results['procrustes_similarity_text_main_flow'][model_name] = similarities
    
            if image_embs_dict:
                similarities = []
                for layer in tqdm(range(max(image_embs_mf_dict.keys()) - 1), desc=f'{model_name} image procrustes_similarity_main_flow'):
                    similarities.append(self.llm_microscope.procrustes_similarity(image_embs_mf_dict[layer], image_embs_mf_dict[layer+1]))
                self.all_results['procrustes_similarity_image_main_flow'][model_name] = similarities

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


    def visualize_tokenwise_linearity(
        self,
        example_idx: int = 0,
        model_name: str = 'LLaVA-NeXT',
        save: bool = True,
        saving_path: str = "./",
        normalization_type: str = "token-wise",
        show_first: bool = True,
        show_last: bool = True
    ):
        log_info = f"Model: {model_name}\n"
        log_info += f"Example index: {example_idx}\n"
        words = self.embeddings[example_idx]['words']
        image_token_pos = words.index("<image>")
        layer2errors_text = self.llm_microscope.get_linear_errors(self.embeddings[example_idx]['text_token_embeddings'])
        layer2errors_image = self.llm_microscope.get_linear_errors(self.embeddings[example_idx]['image_token_embeddings'])
        layer2errors_combined = {}
        for i in layer2errors_text.keys():
            combined_layer_errors = torch.cat((layer2errors_text[i][:image_token_pos], torch.tensor([layer2errors_image[i].sum()]), layer2errors_text[i][image_token_pos:]))
            layer2errors_combined[i] = combined_layer_errors
        all_weights = np.array([layer2errors_combined[layer].tolist() for layer in list(layer2errors_combined.keys())[1:]])
        all_weights_norm, log_info = self.llm_microscope.normalize_weights(all_weights, normalization_type, log_info, show_first, show_last)
        fig = self.llm_microscope.visualize_layers_text(all_weights_norm, words, show_first)

        # Assuming `fig` is the matplotlib figure you want to save
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
    
        # If save is True, save the figure to the specified path
        if save:
            # Ensure the saving_path directory exists
            os.makedirs(saving_path, exist_ok=True)
            
            # Create the full file path (you can customize the filename)
            file_name = f"{model_name}_example_{example_idx}.png"
            full_file_path = os.path.join(saving_path, file_name)
            
            # Save the figure to the file
            with open(full_file_path, 'wb') as f:
                f.write(buf.getbuffer())
        
        return buf.getvalue(), log_info