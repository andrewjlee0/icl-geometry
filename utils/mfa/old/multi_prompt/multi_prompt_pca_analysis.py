"""
Multi-Prompt ICL Analysis with PCA Across Layers

Handles the conceptual challenge of doing PCA across multiple prompts by:
1. Per-task PCA (same task, different variations)
2. Global PCA (all prompts, see if tasks cluster)
3. Layer-by-layer PCA (show how all prompts evolve)
4. Individual prompt trajectories in shared PCA space
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from dataclasses import dataclass
import warnings
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings('ignore')


@dataclass
class ICLExample:
    """Single input-output pair for in-context learning"""
    input_text: str
    output_text: str


@dataclass
class PromptVariation:
    """A variation of the ICL prompt"""
    name: str
    examples: List[ICLExample]
    query: str
    expected_answer: str
    task_type: str


class MultiPromptPCAAnalyzer:
    """Multi-prompt analyzer with cross-layer PCA capabilities"""
    
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
        
    def _construct_prompt(self, examples: List[ICLExample], query: str) -> str:
        """Construct ICL prompt"""
        parts = [f"Input: {ex.input_text}\nOutput: {ex.output_text}" for ex in examples]
        parts.append(f"Input: {query}\nOutput:")
        return "\n".join(parts)
    
    def _generate_answer(self, prompt: str) -> str:
        """Generate model's answer"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Output:" in generated_text:
            parts = generated_text.split("Output:")
            answer = parts[-1].strip().split('\n')[0].strip()
        else:
            answer = generated_text[len(prompt):].strip()
        
        return answer
    
    def _get_token_positions(self, prompt: str, examples: List[ICLExample], query: str) -> Dict:
        """Get token positions"""
        full_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        
        def find_text_position(text: str, start_idx: int) -> int:
            text_tokens = self.tokenizer.encode(text, add_special_tokens=False)
            for i in range(start_idx, len(full_tokens) - len(text_tokens) + 1):
                if full_tokens[i:i+len(text_tokens)] == text_tokens:
                    return i + len(text_tokens) - 1
            return start_idx
        
        positions = {'input_positions': [], 'output_positions': [], 'query_position': None}
        
        current_start = 0
        for ex in examples:
            inp_pos = find_text_position(ex.input_text, current_start)
            positions['input_positions'].append(inp_pos)
            current_start = inp_pos + 1
            
            out_pos = find_text_position(ex.output_text, current_start)
            positions['output_positions'].append(out_pos)
            current_start = out_pos + 1
        
        query_pos = find_text_position(query, current_start)
        positions['query_position'] = query_pos
        
        return positions
    
    def _extract_all_layer_activations(self, prompt: str, positions: Dict) -> Dict[int, Dict[str, np.ndarray]]:
        """Extract activations at all positions across all layers"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        activations = {}
        hooks = []
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                hidden_states = output[0]
                if hidden_states.ndim == 3:
                    hidden_states = hidden_states[0]
                activations[layer_idx] = hidden_states.detach()
            return hook
        
        for idx, layer in enumerate(self.model.model.layers):
            hook = layer.register_forward_hook(make_hook(idx))
            hooks.append(hook)
        
        with torch.no_grad():
            _ = self.model(**inputs)
        
        for hook in hooks:
            hook.remove()
        
        # Organize by position type
        layer_activations = {}
        for layer_idx in range(self.n_layers):
            if layer_idx in activations:
                hidden = activations[layer_idx].cpu().numpy()
                layer_activations[layer_idx] = {
                    'inputs': hidden[np.array(positions['input_positions'])],
                    'outputs': hidden[np.array(positions['output_positions'])],
                    'query': hidden[positions['query_position']]
                }
        
        return layer_activations
    
    def analyze_prompt_variation_full(self, variation: PromptVariation) -> Dict:
        """Analyze a prompt variation with full activation extraction"""
        
        print(f"\n  Analyzing: {variation.name}")
        
        prompt = self._construct_prompt(variation.examples, variation.query)
        model_answer = self._generate_answer(prompt)
        is_correct = self._check_correctness(model_answer, variation.expected_answer, variation.task_type)
        
        positions = self._get_token_positions(prompt, variation.examples, variation.query)
        layer_activations = self._extract_all_layer_activations(prompt, positions)
        
        return {
            'variation_name': variation.name,
            'task_type': variation.task_type,
            'model_answer': model_answer,
            'expected_answer': variation.expected_answer,
            'is_correct': is_correct,
            'layer_activations': layer_activations
        }
    
    def _check_correctness(self, model_answer: str, expected: str, task_type: str) -> bool:
        """Check correctness"""
        model_lower = model_answer.lower().strip()
        expected_lower = expected.lower().strip()
        
        if model_lower == expected_lower or expected_lower in model_lower or model_lower in expected_lower:
            return True
        
        if task_type == "animal_sounds":
            sound_variations = {
                'baa': ['baa', 'bleat', 'maa'],
                'meow': ['meow', 'mew', 'miaow'],
                'bark': ['bark', 'woof', 'arf'],
            }
            for key, variations in sound_variations.items():
                if expected_lower in variations and model_lower in variations:
                    return True
        
        return False
    
    def plot_per_task_pca_evolution(self, results: List[Dict], save_path: str = "per_task_pca_evolution.png"):
        """
        PCA per task type: fit PCA on all prompts of same task, show evolution
        This makes sense conceptually because same task = same semantic space
        """
        
        # Group by task type
        by_task = {}
        for result in results:
            task_type = result['task_type']
            if task_type not in by_task:
                by_task[task_type] = []
            by_task[task_type].append(result)
        
        # Create subplots for each task
        n_tasks = len(by_task)
        n_cols = min(3, n_tasks)
        n_rows = (n_tasks + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_tasks == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_tasks > 1 else [axes]
        
        for idx, (task_type, task_results) in enumerate(by_task.items()):
            ax = axes[idx]
            
            # Collect all query activations across all layers for this task
            all_query_vecs = []
            layer_indices = []
            prompt_indices = []
            correctness = []
            
            for prompt_idx, result in enumerate(task_results):
                for layer_idx in sorted(result['layer_activations'].keys()):
                    query_vec = result['layer_activations'][layer_idx]['query']
                    all_query_vecs.append(query_vec)
                    layer_indices.append(layer_idx)
                    prompt_indices.append(prompt_idx)
                    correctness.append(result['is_correct'])
            
            if not all_query_vecs:
                continue
            
            # Fit global PCA for this task
            all_query_vecs = np.array(all_query_vecs)
            pca = PCA(n_components=2)
            transformed = pca.fit_transform(all_query_vecs)
            
            # Plot trajectories for each prompt
            n_prompts = len(task_results)
            colors = plt.cm.tab10(np.linspace(0, 1, n_prompts))
            
            for prompt_idx in range(n_prompts):
                mask = np.array(prompt_indices) == prompt_idx
                prompt_data = transformed[mask]
                prompt_layers = np.array(layer_indices)[mask]
                is_correct = task_results[prompt_idx]['is_correct']
                
                # Sort by layer
                sort_idx = np.argsort(prompt_layers)
                prompt_data = prompt_data[sort_idx]
                
                # Plot trajectory
                marker = 'o' if is_correct else 'x'
                label = f"{task_results[prompt_idx]['variation_name'][:15]} {'✓' if is_correct else '✗'}"
                
                ax.plot(prompt_data[:, 0], prompt_data[:, 1], 
                       marker=marker, markersize=4, alpha=0.7, 
                       color=colors[prompt_idx], label=label)
                
                # Mark start and end
                ax.scatter(prompt_data[0, 0], prompt_data[0, 1], 
                          s=100, marker='s', color=colors[prompt_idx], 
                          edgecolors='black', linewidths=2, alpha=0.8)
                ax.scatter(prompt_data[-1, 0], prompt_data[-1, 1], 
                          s=150, marker='*', color=colors[prompt_idx],
                          edgecolors='black', linewidths=2, alpha=0.8)
            
            ax.set_title(f"{task_type.replace('_', ' ').title()}\n"
                        f"Query Trajectories Across Layers\n"
                        f"Var explained: {pca.explained_variance_ratio_.sum():.2%}",
                        fontsize=11)
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)
        
        # Remove extra subplots
        for idx in range(len(by_task), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved per-task PCA evolution to {save_path}")
        plt.close()
    
    def plot_global_pca_snapshot(self, results: List[Dict], layer_indices: List[int] = None, 
                                 save_path: str = "global_pca_snapshot.png"):
        """
        Global PCA at specific layers: fit one PCA on all prompts/tasks at selected layers
        Shows if different tasks/prompts cluster differently
        """
        
        if layer_indices is None:
            # Default: early, middle, late layers
            layer_indices = [0, self.n_layers // 4, self.n_layers // 2, 
                           3 * self.n_layers // 4, self.n_layers - 1]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for plot_idx, layer_idx in enumerate(layer_indices):
            if plot_idx >= 6:
                break
            
            ax = axes[plot_idx]
            
            # Collect all query vectors at this layer
            all_vecs = []
            task_types = []
            correctness = []
            prompt_names = []
            
            for result in results:
                if layer_idx in result['layer_activations']:
                    query_vec = result['layer_activations'][layer_idx]['query']
                    all_vecs.append(query_vec)
                    task_types.append(result['task_type'])
                    correctness.append(result['is_correct'])
                    prompt_names.append(result['variation_name'])
            
            if len(all_vecs) < 2:
                continue
            
            # Fit PCA
            all_vecs = np.array(all_vecs)
            pca = PCA(n_components=2)
            transformed = pca.fit_transform(all_vecs)
            
            # Get unique task types for coloring
            unique_tasks = list(set(task_types))
            task_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_tasks)))
            task_to_color = {task: task_colors[i] for i, task in enumerate(unique_tasks)}
            
            # Plot each point
            for i, (vec_2d, task, is_correct, name) in enumerate(zip(transformed, task_types, correctness, prompt_names)):
                color = task_to_color[task]
                marker = 'o' if is_correct else 'x'
                size = 100 if is_correct else 80
                
                ax.scatter(vec_2d[0], vec_2d[1], 
                          c=[color], marker=marker, s=size, 
                          alpha=0.7, edgecolors='black', linewidths=1)
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = []
            for task in unique_tasks:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor=task_to_color[task],
                                             markersize=8, label=task.replace('_', ' ').title()))
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor='gray', markersize=8, label='Correct'))
            legend_elements.append(Line2D([0], [0], marker='x', color='w', 
                                         markerfacecolor='gray', markersize=8, label='Incorrect'))
            
            ax.legend(handles=legend_elements, fontsize=8, loc='best')
            ax.set_title(f"Layer {layer_idx}\nVariance: {pca.explained_variance_ratio_.sum():.2%}",
                        fontsize=11)
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Remove extra subplot
        if len(layer_indices) < 6:
            fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved global PCA snapshot to {save_path}")
        plt.close()
    
    def plot_layer_by_layer_pca(self, results: List[Dict], save_path: str = "layer_by_layer_pca.png"):
        """
        Layer-by-layer PCA: separate PCA for each layer
        Shows how representations evolve across layers
        """
        
        # Select layers to show (grid layout)
        n_layers_to_show = min(16, self.n_layers)
        layer_step = max(1, self.n_layers // n_layers_to_show)
        layers_to_show = list(range(0, self.n_layers, layer_step))[:n_layers_to_show]
        
        n_cols = 4
        n_rows = (len(layers_to_show) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        axes = axes.flatten()
        
        for plot_idx, layer_idx in enumerate(layers_to_show):
            ax = axes[plot_idx]
            
            # Collect vectors at this layer
            all_vecs = []
            colors_list = []
            markers_list = []
            
            for result in results:
                if layer_idx not in result['layer_activations']:
                    continue
                
                # Get input, output, and query vectors
                layer_data = result['layer_activations'][layer_idx]
                
                # Outputs (red)
                for out_vec in layer_data['outputs']:
                    all_vecs.append(out_vec)
                    colors_list.append('red')
                    markers_list.append('s')
                
                # Query (green if correct, orange if incorrect)
                all_vecs.append(layer_data['query'])
                colors_list.append('green' if result['is_correct'] else 'orange')
                markers_list.append('*')
            
            if len(all_vecs) < 2:
                ax.text(0.5, 0.5, f'Layer {layer_idx}\nNo data', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # PCA
            all_vecs = np.array(all_vecs)
            pca = PCA(n_components=2)
            transformed = pca.fit_transform(all_vecs)
            
            # Plot
            for i, (vec_2d, color, marker) in enumerate(zip(transformed, colors_list, markers_list)):
                size = 120 if marker == '*' else 60
                ax.scatter(vec_2d[0], vec_2d[1], c=color, marker=marker, 
                          s=size, alpha=0.6, edgecolors='black', linewidths=0.5)
            
            # Legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                      markersize=7, label='Outputs'),
                Line2D([0], [0], marker='*', color='w', markerfacecolor='green', 
                      markersize=10, label='Query (correct)'),
                Line2D([0], [0], marker='*', color='w', markerfacecolor='orange', 
                      markersize=10, label='Query (incorrect)')
            ]
            ax.legend(handles=legend_elements, fontsize=7, loc='best')
            
            ax.set_title(f"Layer {layer_idx}\nVar: {pca.explained_variance_ratio_.sum():.2%}",
                        fontsize=10)
            ax.set_xlabel("PC1", fontsize=9)
            ax.set_ylabel("PC2", fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Remove extra subplots
        for idx in range(len(layers_to_show), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved layer-by-layer PCA to {save_path}")
        plt.close()
    
    def plot_3d_trajectory_comparison(self, results: List[Dict], save_path: str = "3d_trajectory_comparison.png"):
        """
        3D trajectories: show how different prompts evolve through layer space
        Fit global PCA with 3 components
        """
        
        # Collect all query vectors across all layers and prompts
        all_vecs = []
        layer_indices = []
        prompt_indices = []
        
        for prompt_idx, result in enumerate(results):
            for layer_idx in sorted(result['layer_activations'].keys()):
                query_vec = result['layer_activations'][layer_idx]['query']
                all_vecs.append(query_vec)
                layer_indices.append(layer_idx)
                prompt_indices.append(prompt_idx)
        
        if len(all_vecs) < 3:
            print("Not enough data for 3D trajectory")
            return
        
        # Fit 3D PCA
        all_vecs = np.array(all_vecs)
        pca = PCA(n_components=3)
        transformed = pca.fit_transform(all_vecs)
        
        # Plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        n_prompts = len(results)
        colors = plt.cm.tab20(np.linspace(0, 1, n_prompts))
        
        for prompt_idx in range(n_prompts):
            mask = np.array(prompt_indices) == prompt_idx
            prompt_data = transformed[mask]
            prompt_layers = np.array(layer_indices)[mask]
            
            # Sort by layer
            sort_idx = np.argsort(prompt_layers)
            prompt_data = prompt_data[sort_idx]
            
            is_correct = results[prompt_idx]['is_correct']
            linestyle = '-' if is_correct else '--'
            alpha = 0.8 if is_correct else 0.5
            
            label = f"{results[prompt_idx]['variation_name'][:20]} {'✓' if is_correct else '✗'}"
            
            # Plot trajectory
            ax.plot(prompt_data[:, 0], prompt_data[:, 1], prompt_data[:, 2],
                   linestyle=linestyle, linewidth=2, alpha=alpha,
                   color=colors[prompt_idx], label=label)
            
            # Mark start
            ax.scatter(prompt_data[0, 0], prompt_data[0, 1], prompt_data[0, 2],
                      s=100, marker='o', color=colors[prompt_idx], 
                      edgecolors='black', linewidths=2)
            
            # Mark end
            ax.scatter(prompt_data[-1, 0], prompt_data[-1, 1], prompt_data[-1, 2],
                      s=200, marker='*', color=colors[prompt_idx],
                      edgecolors='black', linewidths=2)
        
        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('PC2', fontsize=11)
        ax.set_zlabel('PC3', fontsize=11)
        ax.set_title(f'3D Query Trajectories Across All Prompts\n'
                    f'Variance Explained: {pca.explained_variance_ratio_.sum():.2%}',
                    fontsize=13)
        ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D trajectory comparison to {save_path}")
        plt.close()
    
    def analyze_with_pca(self, variations: List[PromptVariation], prefix: str = "multi_pca"):
        """Full analysis with all PCA visualizations"""
        
        print("="*70)
        print(f"Multi-Prompt PCA Analysis: {len(variations)} variations")
        print("="*70)
        
        # Analyze all variations
        results = []
        for variation in variations:
            result = self.analyze_prompt_variation_full(variation)
            results.append(result)
            status = "✓" if result['is_correct'] else "✗"
            print(f"    {status} {variation.name}: {result['model_answer']}")
        
        # Generate all PCA plots
        print("\n" + "="*70)
        print("GENERATING PCA VISUALIZATIONS")
        print("="*70)
        
        self.plot_per_task_pca_evolution(results, f"{prefix}_per_task_evolution.png")
        self.plot_global_pca_snapshot(results, save_path=f"{prefix}_global_snapshot.png")
        self.plot_layer_by_layer_pca(results, f"{prefix}_layer_by_layer.png")
        self.plot_3d_trajectory_comparison(results, f"{prefix}_3d_trajectories.png")
        
        # Summary
        correct_count = sum(1 for r in results if r['is_correct'])
        print("\n" + "="*70)
        print(f"Analysis complete!")
        print(f"Accuracy: {correct_count}/{len(results)} = {correct_count/len(results):.1%}")
        print(f"\nGenerated files:")
        print(f"  - {prefix}_per_task_evolution.png")
        print(f"  - {prefix}_global_snapshot.png")
        print(f"  - {prefix}_layer_by_layer.png")
        print(f"  - {prefix}_3d_trajectories.png")
        print("="*70)
        
        return results


# Import functions from original script
def create_animal_sounds_variations() -> List[PromptVariation]:
    """Create variations of animal sounds task"""
    
    variations = []
    base_animals = [
        ("cat", "meow"), ("dog", "bark"), ("cow", "moo"),
        ("duck", "quack"), ("pig", "oink"), ("sheep", "baa"),
        ("horse", "neigh"), ("chicken", "cluck")
    ]
    
    variations.append(PromptVariation(
        name="Std_sheep", examples=[ICLExample(a, s) for a, s in base_animals[:5]],
        query="sheep", expected_answer="baa", task_type="animal_sounds"
    ))
    
    variations.append(PromptVariation(
        name="Alt_sheep", examples=[ICLExample(a, s) for a, s in [base_animals[i] for i in [1, 2, 4, 6, 7]]],
        query="sheep", expected_answer="baa", task_type="animal_sounds"
    ))
    
    variations.append(PromptVariation(
        name="Std_horse", examples=[ICLExample(a, s) for a, s in base_animals[:5]],
        query="horse", expected_answer="neigh", task_type="animal_sounds"
    ))
    
    variations.append(PromptVariation(
        name="Few_shot_3", examples=[ICLExample(a, s) for a, s in base_animals[:3]],
        query="sheep", expected_answer="baa", task_type="animal_sounds"
    ))
    
    return variations


def create_diverse_task_variations() -> List[PromptVariation]:
    """Create variations across different task types"""
    
    variations = []
    
    variations.append(PromptVariation(
        name="Animals", 
        examples=[ICLExample("cat", "meow"), ICLExample("dog", "bark"), ICLExample("cow", "moo")],
        query="sheep", expected_answer="baa", task_type="animal_sounds"
    ))
    
    variations.append(PromptVariation(
        name="Antonyms",
        examples=[ICLExample("hot", "cold"), ICLExample("big", "small"), ICLExample("fast", "slow")],
        query="light", expected_answer="dark", task_type="antonyms"
    ))
    
    variations.append(PromptVariation(
        name="Plurals",
        examples=[ICLExample("cat", "cats"), ICLExample("dog", "dogs"), ICLExample("box", "boxes")],
        query="wolf", expected_answer="wolves", task_type="plurals"
    ))
    
    return variations


def main():
    """Run multi-prompt PCA analysis"""
    
    print("="*70)
    print("MULTI-PROMPT PCA ANALYSIS")
    print("="*70)
    print("\nChoose analysis:")
    print("  1. Animal sounds variations (same task)")
    print("  2. Diverse tasks (different tasks)")
    print("  3. Both")
    
    choice = input("\nEnter choice (1/2/3) [default: 1]: ").strip() or "1"
    
    analyzer = MultiPromptPCAAnalyzer()
    
    if choice == "1":
        variations = create_animal_sounds_variations()
        prefix = "animal_pca"
    elif choice == "2":
        variations = create_diverse_task_variations()
        prefix = "diverse_pca"
    else:
        variations = create_animal_sounds_variations() + create_diverse_task_variations()
        prefix = "comprehensive_pca"
    
    analyzer.analyze_with_pca(variations, prefix=prefix)


if __name__ == "__main__":
    main()
