"""
Mechanistic Interpretability Analysis for In-Context Learning in Llama 3.2 1B

Analyzes how input, output, and query token representations evolve across layers
using PCA and other visualization techniques.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ICLExample:
    """Single input-output pair for in-context learning"""
    input_text: str
    output_text: str


class ResidualStreamExtractor:
    """Extract residual stream activations from Llama model"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        self.model.eval()
        
        self.n_layers = len(self.model.model.layers)
        print(f"Model loaded: {self.n_layers} layers")
        
        # Storage for activations
        self.activations = {}
        self.hooks = []
        
    def _get_final_token_position(self, text: str, start_idx: int, full_tokens: list) -> int:
        """
        Get the position of the final token for a given text within a prompt.
        Handles multi-token words by returning the last token's position.
        
        Args:
            text: The text to find
            start_idx: Start searching from this token index
            full_tokens: The full tokenized prompt
        """
        # Get tokens for just this text
        text_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Search for the text token sequence starting from start_idx
        for i in range(start_idx, len(full_tokens) - len(text_tokens) + 1):
            if full_tokens[i:i+len(text_tokens)] == text_tokens:
                # Return the position of the last token of this text
                return i + len(text_tokens) - 1
        
        # If exact match fails, try partial match (sometimes tokenization is context-dependent)
        # Just return the approximate position
        print(f"Warning: Could not find exact match for '{text}', using approximation")
        return start_idx
    
    def register_hooks(self, extract_pre_mlp: bool = False):
        """Register forward hooks to extract residual stream at each layer"""
        
        def hook_fn(layer_idx, extract_pre):
            def hook(module, input, output):
                # output is a tuple: (hidden_states, ...)
                hidden_states = output[0]
                
                key = f"layer_{layer_idx}_pre_mlp" if extract_pre else f"layer_{layer_idx}"
                self.activations[key] = hidden_states.detach()
            return hook
        
        # Hook after attention (pre-MLP) if requested
        if extract_pre_mlp:
            for idx, layer in enumerate(self.model.model.layers):
                hook = layer.self_attn.register_forward_hook(hook_fn(idx, True))
                self.hooks.append(hook)
        
        # Hook after MLP (end of layer) - always extract this
        for idx, layer in enumerate(self.model.model.layers):
            hook = layer.register_forward_hook(hook_fn(idx, False))
            self.hooks.append(hook)
    
    def clear_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
    
    def extract_residual_streams(
        self,
        examples: List[ICLExample],
        query_input: str,
        extract_pre_mlp: bool = False,
        generate_output: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Extract residual stream activations for ICL task.
        
        Returns dict with keys:
            - input_positions: list of token positions for inputs
            - output_positions: list of token positions for outputs
            - query_position: token position for query
            - layer_X: activation tensor of shape [batch=1, seq_len, hidden_dim]
            - generated_output: (optional) what the model generates
        """
        # Construct the ICL prompt
        prompt = self._construct_icl_prompt(examples, query_input)
        print(f"\nPrompt ({len(prompt)} chars):\n{prompt}\n")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        print(f"Tokenized length: {inputs['input_ids'].shape[1]} tokens")
        
        # Generate output first if requested (before hooks interfere)
        result = {}
        if generate_output:
            print("\n" + "="*60)
            print("GENERATING MODEL OUTPUT")
            print("="*60)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,  # Greedy decoding for reproducibility
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the generated output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the answer (everything after "Output:")
            if "Output:" in generated_text:
                parts = generated_text.split("Output:")
                model_answer = parts[-1].strip().split('\n')[0].strip()
            else:
                model_answer = generated_text[len(prompt):].strip()
            
            print(f"\n✓ Model's answer: '{model_answer}'")
            print(f"  Full generation:\n{generated_text[len(prompt):]}")
            print("="*60 + "\n")
            
            result['generated_output'] = model_answer
            result['full_generation'] = generated_text
        
        # Find token positions for each input, output, and query
        positions = self._get_token_positions(examples, query_input, prompt)
        
        # Register hooks
        self.register_hooks(extract_pre_mlp)
        
        # Forward pass (for activation extraction)
        print("Extracting activations...")
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # Store positions in activations dict
        result.update(self.activations)
        result.update(positions)
        
        # Clear hooks
        self.clear_hooks()
        
        return result
    
    def _construct_icl_prompt(self, examples: List[ICLExample], query_input: str) -> str:
        """Construct the ICL prompt with examples and query"""
        prompt_parts = []
        
        for ex in examples:
            prompt_parts.append(f"Input: {ex.input_text}")
            prompt_parts.append(f"Output: {ex.output_text}")
        
        prompt_parts.append(f"Input: {query_input}")
        prompt_parts.append("Output:")
        
        return "\n".join(prompt_parts)
    
    def _get_token_positions(
        self,
        examples: List[ICLExample],
        query_input: str,
        full_prompt: str
    ) -> Dict[str, List[int]]:
        """Get final token positions for all inputs, outputs, and query"""
        
        # Tokenize the full prompt once
        full_tokens = self.tokenizer.encode(full_prompt, add_special_tokens=True)
        
        print(f"Full tokenization ({len(full_tokens)} tokens):")
        print(f"Tokens: {full_tokens[:20]}...")  # Show first 20
        
        input_positions = []
        output_positions = []
        query_position = None
        
        # Build the prompt step by step to track positions
        current_pos = 0
        
        for i, example in enumerate(examples):
            # Find "Input: {text}" tokens
            input_line = f"Input: {example.input_text}"
            input_tokens = self.tokenizer.encode(input_line + "\n", add_special_tokens=False)
            
            # The input text position is at the end of the input line
            # Find where the actual input word ends
            just_input_tokens = self.tokenizer.encode(example.input_text, add_special_tokens=False)
            input_end_pos = current_pos + len(input_tokens) - 1 - 1  # -1 for newline, -1 for 0-indexing
            input_positions.append(input_end_pos)
            
            current_pos += len(input_tokens)
            
            # Find "Output: {text}" tokens
            output_line = f"Output: {example.output_text}"
            output_tokens = self.tokenizer.encode(output_line + "\n", add_special_tokens=False)
            
            # The output text position is at the end of the output
            just_output_tokens = self.tokenizer.encode(example.output_text, add_special_tokens=False)
            output_end_pos = current_pos + len(output_tokens) - 1 - 1  # -1 for newline, -1 for 0-indexing
            output_positions.append(output_end_pos)
            
            current_pos += len(output_tokens)
        
        # Now handle the query
        query_line = f"Input: {query_input}"
        query_tokens = self.tokenizer.encode(query_line + "\n", add_special_tokens=False)
        just_query_tokens = self.tokenizer.encode(query_input, add_special_tokens=False)
        query_position = current_pos + len(query_tokens) - 1 - 1
        
        print(f"\nToken positions found:")
        print(f"  Input positions: {input_positions}")
        print(f"  Output positions: {output_positions}")
        print(f"  Query position: {query_position}")
        
        # Validate positions are within bounds
        max_pos = max(input_positions + output_positions + [query_position])
        if max_pos >= len(full_tokens):
            print(f"WARNING: Position {max_pos} exceeds sequence length {len(full_tokens)}")
            print("Adjusting positions...")
            # Recalculate using a more robust method
            return self._get_token_positions_robust(examples, query_input, full_tokens)
        
        return {
            "input_positions": input_positions,
            "output_positions": output_positions,
            "query_position": query_position
        }
    
    def _get_token_positions_robust(
        self,
        examples: List[ICLExample],
        query_input: str,
        full_tokens: List[int]
    ) -> Dict[str, List[int]]:
        """Robust fallback method to find token positions by searching through tokens"""
        
        input_positions = []
        output_positions = []
        query_position = None
        
        current_search_start = 0
        
        for i, example in enumerate(examples):
            # Find input text
            input_tokens = self.tokenizer.encode(example.input_text, add_special_tokens=False)
            input_pos = self._get_final_token_position(
                example.input_text, current_search_start, full_tokens
            )
            input_positions.append(input_pos)
            current_search_start = input_pos + 1
            
            # Find output text
            output_tokens = self.tokenizer.encode(example.output_text, add_special_tokens=False)
            output_pos = self._get_final_token_position(
                example.output_text, current_search_start, full_tokens
            )
            output_positions.append(output_pos)
            current_search_start = output_pos + 1
        
        # Find query
        query_tokens = self.tokenizer.encode(query_input, add_special_tokens=False)
        query_position = self._get_final_token_position(
            query_input, current_search_start, full_tokens
        )
        
        print(f"\nRobust token positions found:")
        print(f"  Input positions: {input_positions}")
        print(f"  Output positions: {output_positions}")
        print(f"  Query position: {query_position}")
        
        return {
            "input_positions": input_positions,
            "output_positions": output_positions,
            "query_position": query_position
        }


class ICLVisualizer:
    """Visualize the residual stream evolution across layers"""
    
    def __init__(self, activations: Dict, n_components: int = 2):
        self.activations = activations
        self.n_components = n_components
        self.input_positions = activations["input_positions"]
        self.output_positions = activations["output_positions"]
        self.query_position = activations["query_position"]
        
        # Get number of layers
        self.n_layers = len([k for k in activations.keys() if k.startswith("layer_") and "pre_mlp" not in k])
    
    def plot_pca_evolution(self, save_path: str = "pca_evolution.png"):
        """Plot PCA of token positions across all layers"""
        
        n_rows = (self.n_layers + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for layer_idx in range(self.n_layers):
            layer_key = f"layer_{layer_idx}"
            # Get hidden states and ensure it's 2D [seq_len, hidden_dim]
            hidden_states = self.activations[layer_key].cpu().numpy()
            if hidden_states.ndim == 3:  # [batch, seq_len, hidden_dim]
                hidden_states = hidden_states[0]
            elif hidden_states.ndim == 1:  # Something went wrong, skip
                print(f"Warning: Layer {layer_idx} has unexpected shape {hidden_states.shape}")
                continue
            
            # Gather vectors for inputs, outputs, and query
            # Ensure we get the right shapes by using numpy array indexing
            input_vecs = hidden_states[np.array(self.input_positions)]  # [n_inputs, hidden_dim]
            output_vecs = hidden_states[np.array(self.output_positions)]  # [n_outputs, hidden_dim]
            query_vec = hidden_states[np.array([self.query_position])]  # [1, hidden_dim]
            
            # Debug: check shapes
            if layer_idx == 0:
                print(f"\nShape debugging (layer 0):")
                print(f"  hidden_states shape: {hidden_states.shape}")
                print(f"  input_positions: {self.input_positions}")
                print(f"  output_positions: {self.output_positions}")
                print(f"  query_position: {self.query_position}")
                print(f"  input_vecs shape: {input_vecs.shape}")
                print(f"  output_vecs shape: {output_vecs.shape}")
                print(f"  query_vec shape: {query_vec.shape}")
            
            # Combine all vectors for PCA
            all_vecs = np.vstack([input_vecs, output_vecs, query_vec])  # [n_inputs + n_outputs + 1, hidden_dim]
            
            # Fit PCA
            pca = PCA(n_components=self.n_components)
            transformed = pca.fit_transform(all_vecs)
            
            # Split back
            n_inputs = len(self.input_positions)
            n_outputs = len(self.output_positions)
            
            input_pca = transformed[:n_inputs]
            output_pca = transformed[n_inputs:n_inputs+n_outputs]
            query_pca = transformed[-1:]
            
            # Plot
            ax = axes[layer_idx]
            ax.scatter(input_pca[:, 0], input_pca[:, 1], 
                      c='blue', marker='o', s=100, label='Inputs', alpha=0.7)
            ax.scatter(output_pca[:, 0], output_pca[:, 1], 
                      c='red', marker='s', s=100, label='Outputs', alpha=0.7)
            ax.scatter(query_pca[:, 0], query_pca[:, 1], 
                      c='green', marker='*', s=200, label='Query', alpha=0.9)
            
            # Add arrows from inputs to outputs
            for i in range(min(n_inputs, n_outputs)):
                ax.arrow(input_pca[i, 0], input_pca[i, 1],
                        output_pca[i, 0] - input_pca[i, 0],
                        output_pca[i, 1] - input_pca[i, 1],
                        alpha=0.3, color='gray', head_width=0.05)
            
            ax.set_title(f"Layer {layer_idx} (Var: {pca.explained_variance_ratio_.sum():.2%})")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove extra subplots
        for idx in range(self.n_layers, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved PCA evolution plot to {save_path}")
        plt.close()
    
    def plot_cosine_similarity_heatmaps(self, save_path: str = "cosine_similarity.png"):
        """Plot cosine similarity between query and examples across layers"""
        
        similarities_to_inputs = []
        similarities_to_outputs = []
        
        for layer_idx in range(self.n_layers):
            layer_key = f"layer_{layer_idx}"
            hidden_states = self.activations[layer_key].cpu().numpy()
            if hidden_states.ndim == 3:
                hidden_states = hidden_states[0]
            
            query_vec = hidden_states[self.query_position]
            input_vecs = hidden_states[np.array(self.input_positions)]
            output_vecs = hidden_states[np.array(self.output_positions)]
            
            # Compute cosine similarities
            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
            
            sim_to_inputs = []
            for inp_vec in input_vecs:
                inp_norm = inp_vec / (np.linalg.norm(inp_vec) + 1e-8)
                sim_to_inputs.append(np.dot(query_norm, inp_norm))
            
            sim_to_outputs = []
            for out_vec in output_vecs:
                out_norm = out_vec / (np.linalg.norm(out_vec) + 1e-8)
                sim_to_outputs.append(np.dot(query_norm, out_norm))
            
            similarities_to_inputs.append(sim_to_inputs)
            similarities_to_outputs.append(sim_to_outputs)
        
        # Convert to arrays
        sim_inputs = np.array(similarities_to_inputs).T  # [n_examples, n_layers]
        sim_outputs = np.array(similarities_to_outputs).T
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.heatmap(sim_inputs, ax=ax1, cmap='coolwarm', center=0,
                   xticklabels=range(self.n_layers), 
                   yticklabels=[f"Ex {i+1}" for i in range(len(self.input_positions))],
                   cbar_kws={'label': 'Cosine Similarity'})
        ax1.set_title("Query-to-Input Similarity Across Layers")
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Example")
        
        sns.heatmap(sim_outputs, ax=ax2, cmap='coolwarm', center=0,
                   xticklabels=range(self.n_layers),
                   yticklabels=[f"Ex {i+1}" for i in range(len(self.output_positions))],
                   cbar_kws={'label': 'Cosine Similarity'})
        ax2.set_title("Query-to-Output Similarity Across Layers")
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Example")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved cosine similarity heatmaps to {save_path}")
        plt.close()
    
    def plot_norm_evolution(self, save_path: str = "norm_evolution.png"):
        """Plot how vector norms evolve across layers"""
        
        input_norms = []
        output_norms = []
        query_norms = []
        
        for layer_idx in range(self.n_layers):
            layer_key = f"layer_{layer_idx}"
            hidden_states = self.activations[layer_key].cpu().numpy()
            if hidden_states.ndim == 3:
                hidden_states = hidden_states[0]
            
            input_vecs = hidden_states[np.array(self.input_positions)]
            output_vecs = hidden_states[np.array(self.output_positions)]
            query_vec = hidden_states[self.query_position]
            
            input_norms.append(np.linalg.norm(input_vecs, axis=1))
            output_norms.append(np.linalg.norm(output_vecs, axis=1))
            query_norms.append(np.linalg.norm(query_vec))
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        input_norms = np.array(input_norms)  # [n_layers, n_examples]
        output_norms = np.array(output_norms)
        query_norms = np.array(query_norms)
        
        layers = range(self.n_layers)
        
        # Plot mean with std
        ax.plot(layers, input_norms.mean(axis=1), 'b-', label='Inputs (mean)', linewidth=2)
        ax.fill_between(layers, 
                        input_norms.mean(axis=1) - input_norms.std(axis=1),
                        input_norms.mean(axis=1) + input_norms.std(axis=1),
                        alpha=0.2, color='blue')
        
        ax.plot(layers, output_norms.mean(axis=1), 'r-', label='Outputs (mean)', linewidth=2)
        ax.fill_between(layers,
                        output_norms.mean(axis=1) - output_norms.std(axis=1),
                        output_norms.mean(axis=1) + output_norms.std(axis=1),
                        alpha=0.2, color='red')
        
        ax.plot(layers, query_norms, 'g-', label='Query', linewidth=2)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('L2 Norm')
        ax.set_title('Vector Norm Evolution Across Layers')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved norm evolution plot to {save_path}")
        plt.close()


def evaluate_icl_performance(examples: List[ICLExample], query_input: str, model_answer: str) -> Dict:
    """
    Evaluate how well the model performed on the ICL task.
    Returns a dict with evaluation metrics.
    """
    # Get expected outputs to compare against
    expected_outputs = [ex.output_text for ex in examples]
    
    # Simple evaluation: check if answer matches any expected pattern
    exact_match = model_answer.lower() in [out.lower() for out in expected_outputs]
    
    # For this specific task, we can check if it's a reasonable animal sound
    is_animal_sound = any(sound in model_answer.lower() for sound in ['baa', 'bleat', 'maa'])
    
    result = {
        'model_answer': model_answer,
        'query': query_input,
        'exact_match_to_examples': exact_match,
        'seems_correct': is_animal_sound or exact_match
    }
    
    return result


def main():
    """Run the mechanistic interpretability analysis"""
    
    # Define ICL task - simple pattern completion
    examples = [
        ICLExample("cat", "meow"),
        ICLExample("dog", "bark"),
        ICLExample("cow", "moo"),
        ICLExample("duck", "quack"),
        ICLExample("pig", "oink"),
    ]
    query_input = "horse"
    expected_answer = "neigh"
    
    print("="*60)
    print("Mechanistic Interpretability Analysis for ICL")
    print("="*60)
    print(f"\nTask: Animal sounds")
    print(f"Examples: {len(examples)}")
    for i, ex in enumerate(examples, 1):
        print(f"  {i}. {ex.input_text} → {ex.output_text}")
    print(f"\nQuery: {query_input}")
    print(f"Expected answer: {expected_answer}")
    
    # Extract residual streams
    extractor = ResidualStreamExtractor()
    activations = extractor.extract_residual_streams(examples, query_input, generate_output=True)
    
    print(f"\nExtracted activations for {extractor.n_layers} layers")
    print(f"Input positions: {activations['input_positions']}")
    print(f"Output positions: {activations['output_positions']}")
    print(f"Query position: {activations['query_position']}")
    
    # Evaluate performance
    if 'generated_output' in activations:
        evaluation = evaluate_icl_performance(examples, query_input, activations['generated_output'])
        print(f"\n{'='*60}")
        print("MODEL PERFORMANCE EVALUATION")
        print('='*60)
        print(f"Query: {evaluation['query']}")
        print(f"Model's answer: '{evaluation['model_answer']}'")
        print(f"Expected: '{expected_answer}'")
        print(f"Seems correct: {'✓ YES' if evaluation['seems_correct'] else '✗ NO'}")
        print('='*60 + "\n")
    
    # Visualize
    print("Generating visualizations...")
    visualizer = ICLVisualizer(activations)
    visualizer.plot_pca_evolution("pca_evolution.png")
    visualizer.plot_cosine_similarity_heatmaps("cosine_similarity.png")
    visualizer.plot_norm_evolution("norm_evolution.png")
    
    print("\n" + "="*60)
    print("Analysis complete! Check the generated plots.")
    print("="*60)


if __name__ == "__main__":
    main()
