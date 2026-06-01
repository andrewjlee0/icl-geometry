"""
Multi-Prompt ICL Analysis

Tests multiple prompts to understand:
1. How robust the ICL mechanism is across different examples
2. Which layers consistently perform pattern matching
3. How performance correlates with representation geometry
4. What happens when the model fails vs succeeds
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
import json
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
    task_type: str  # e.g., "animal_sounds", "antonyms", etc.


class MultiPromptAnalyzer:
    """Analyze multiple prompt variations"""
    
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
        """Generate model's answer for a prompt"""
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
    
    def _extract_layer_activations(self, prompt: str, token_positions: List[int]) -> Dict[int, np.ndarray]:
        """Extract activations at specific token positions across all layers"""
        
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
        
        # Register hooks
        for idx, layer in enumerate(self.model.model.layers):
            hook = layer.register_forward_hook(make_hook(idx))
            hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # Clear hooks
        for hook in hooks:
            hook.remove()
        
        # Extract specific positions
        layer_activations = {}
        for layer_idx in range(self.n_layers):
            if layer_idx in activations:
                hidden = activations[layer_idx].cpu().numpy()
                layer_activations[layer_idx] = hidden[np.array(token_positions)]
        
        return layer_activations
    
    def _get_token_positions(self, prompt: str, examples: List[ICLExample], query: str) -> Dict[str, List[int]]:
        """Get token positions for inputs, outputs, and query"""
        
        full_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        
        def find_text_position(text: str, start_idx: int) -> int:
            text_tokens = self.tokenizer.encode(text, add_special_tokens=False)
            for i in range(start_idx, len(full_tokens) - len(text_tokens) + 1):
                if full_tokens[i:i+len(text_tokens)] == text_tokens:
                    return i + len(text_tokens) - 1
            return start_idx
        
        positions = {
            'input_positions': [],
            'output_positions': [],
            'query_position': None
        }
        
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
    
    def analyze_prompt_variation(self, variation: PromptVariation) -> Dict:
        """Analyze a single prompt variation"""
        
        print(f"\nAnalyzing: {variation.name}")
        print(f"  Task: {variation.task_type}")
        print(f"  Examples: {len(variation.examples)}")
        
        prompt = self._construct_prompt(variation.examples, variation.query)
        
        # Generate answer
        model_answer = self._generate_answer(prompt)
        is_correct = self._check_correctness(model_answer, variation.expected_answer, variation.task_type)
        
        print(f"  Query: {variation.query}")
        print(f"  Expected: {variation.expected_answer}")
        print(f"  Model: '{model_answer}'")
        print(f"  Correct: {'✓' if is_correct else '✗'}")
        
        # Get positions
        positions = self._get_token_positions(prompt, variation.examples, variation.query)
        
        # Extract activations for output and query tokens
        all_positions = positions['output_positions'] + [positions['query_position']]
        layer_activations = self._extract_layer_activations(prompt, all_positions)
        
        # Compute metrics
        metrics = self._compute_metrics(layer_activations, positions)
        
        return {
            'variation_name': variation.name,
            'task_type': variation.task_type,
            'model_answer': model_answer,
            'expected_answer': variation.expected_answer,
            'is_correct': is_correct,
            'positions': positions,
            'layer_activations': layer_activations,
            'metrics': metrics
        }
    
    def _check_correctness(self, model_answer: str, expected: str, task_type: str) -> bool:
        """Check if model's answer is correct"""
        model_lower = model_answer.lower().strip()
        expected_lower = expected.lower().strip()
        
        # Exact match
        if model_lower == expected_lower:
            return True
        
        # Partial match (answer contains expected)
        if expected_lower in model_lower or model_lower in expected_lower:
            return True
        
        # Task-specific checks
        if task_type == "animal_sounds":
            # Accept common variations
            sound_variations = {
                'baa': ['baa', 'bleat', 'maa'],
                'meow': ['meow', 'mew', 'miaow'],
                'bark': ['bark', 'woof', 'arf'],
                'moo': ['moo'],
                'quack': ['quack'],
                'oink': ['oink']
            }
            for key, variations in sound_variations.items():
                if expected_lower in variations and model_lower in variations:
                    return True
        
        return False
    
    def _compute_metrics(self, layer_activations: Dict[int, np.ndarray], positions: Dict) -> Dict:
        """Compute metrics across layers"""
        
        metrics = {
            'query_to_output_similarity': [],  # Cosine sim per layer
            'output_cluster_tightness': [],    # Variance of output embeddings
            'query_norm': [],                  # L2 norm of query embedding
        }
        
        n_outputs = len(positions['output_positions'])
        
        for layer_idx in sorted(layer_activations.keys()):
            vecs = layer_activations[layer_idx]
            
            if len(vecs) < n_outputs + 1:
                continue
            
            output_vecs = vecs[:n_outputs]
            query_vec = vecs[-1]
            
            # Query-to-output similarity
            query_norm_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
            similarities = []
            for out_vec in output_vecs:
                out_norm = out_vec / (np.linalg.norm(out_vec) + 1e-8)
                sim = np.dot(query_norm_vec, out_norm)
                similarities.append(sim)
            
            metrics['query_to_output_similarity'].append(np.mean(similarities))
            
            # Output cluster tightness (lower = tighter cluster)
            if len(output_vecs) > 1:
                output_center = output_vecs.mean(axis=0)
                distances = [np.linalg.norm(v - output_center) for v in output_vecs]
                metrics['output_cluster_tightness'].append(np.mean(distances))
            else:
                metrics['output_cluster_tightness'].append(0.0)
            
            # Query norm
            metrics['query_norm'].append(np.linalg.norm(query_vec))
        
        return metrics
    
    def analyze_multiple_prompts(self, variations: List[PromptVariation]) -> Dict:
        """Analyze multiple prompt variations"""
        
        print("="*70)
        print(f"Multi-Prompt Analysis: {len(variations)} variations")
        print("="*70)
        
        results = []
        for variation in variations:
            result = self.analyze_prompt_variation(variation)
            results.append(result)
        
        # Aggregate results
        aggregated = self._aggregate_results(results)
        
        return {
            'individual_results': results,
            'aggregated': aggregated
        }
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate metrics across all prompt variations"""
        
        # Separate by correctness
        correct_results = [r for r in results if r['is_correct']]
        incorrect_results = [r for r in results if not r['is_correct']]
        
        aggregated = {
            'total_prompts': len(results),
            'correct_count': len(correct_results),
            'incorrect_count': len(incorrect_results),
            'accuracy': len(correct_results) / len(results) if results else 0,
            'metrics_by_correctness': {
                'correct': self._average_metrics(correct_results) if correct_results else None,
                'incorrect': self._average_metrics(incorrect_results) if incorrect_results else None
            }
        }
        
        return aggregated
    
    def _average_metrics(self, results: List[Dict]) -> Dict:
        """Average metrics across multiple results"""
        
        if not results:
            return {}
        
        # Get all metric keys
        metric_keys = results[0]['metrics'].keys()
        
        averaged = {}
        for key in metric_keys:
            values = [r['metrics'][key] for r in results if key in r['metrics']]
            if values and len(values[0]) > 0:
                # Average across results, per layer
                values_array = np.array(values)
                averaged[key] = {
                    'mean': values_array.mean(axis=0).tolist(),
                    'std': values_array.std(axis=0).tolist()
                }
        
        return averaged
    
    def plot_accuracy_vs_similarity(self, analysis_results: Dict, save_path: str = "accuracy_vs_similarity.png"):
        """Plot how query-to-output similarity correlates with accuracy"""
        
        results = analysis_results['individual_results']
        
        # Extract data per layer
        n_layers = self.n_layers
        layer_accuracies = [[] for _ in range(n_layers)]
        layer_similarities = [[] for _ in range(n_layers)]
        
        for result in results:
            is_correct = 1 if result['is_correct'] else 0
            similarities = result['metrics']['query_to_output_similarity']
            
            for layer_idx, sim in enumerate(similarities):
                if layer_idx < n_layers:
                    layer_similarities[layer_idx].append(sim)
                    layer_accuracies[layer_idx].append(is_correct)
        
        # Compute average similarity for correct vs incorrect
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        correct_sims = []
        incorrect_sims = []
        
        for layer_idx in range(n_layers):
            if layer_accuracies[layer_idx]:
                correct_vals = [layer_similarities[layer_idx][i] 
                               for i, acc in enumerate(layer_accuracies[layer_idx]) if acc == 1]
                incorrect_vals = [layer_similarities[layer_idx][i] 
                                 for i, acc in enumerate(layer_accuracies[layer_idx]) if acc == 0]
                
                correct_sims.append(np.mean(correct_vals) if correct_vals else 0)
                incorrect_sims.append(np.mean(incorrect_vals) if incorrect_vals else 0)
        
        # Plot 1: Similarity by correctness across layers
        layers = range(len(correct_sims))
        ax1.plot(layers, correct_sims, 'g-o', label='Correct predictions', linewidth=2, markersize=6)
        ax1.plot(layers, incorrect_sims, 'r-x', label='Incorrect predictions', linewidth=2, markersize=6)
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Query-to-Output Similarity', fontsize=12)
        ax1.set_title('Similarity Patterns: Correct vs Incorrect', fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Per-layer accuracy
        layer_acc_means = []
        for layer_idx in range(n_layers):
            if layer_accuracies[layer_idx]:
                layer_acc_means.append(np.mean(layer_accuracies[layer_idx]))
            else:
                layer_acc_means.append(0)
        
        ax2.bar(range(len(layer_acc_means)), layer_acc_means, alpha=0.7, color='steelblue')
        ax2.axhline(y=analysis_results['aggregated']['accuracy'], color='red', 
                   linestyle='--', label=f"Overall Accuracy: {analysis_results['aggregated']['accuracy']:.2%}")
        ax2.set_xlabel('Layer', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Model Accuracy Across Dataset', fontsize=13)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved accuracy vs similarity plot to {save_path}")
        plt.close()
    
    def plot_metric_comparison(self, analysis_results: Dict, save_path: str = "metric_comparison.png"):
        """Compare metrics between correct and incorrect predictions"""
        
        agg = analysis_results['aggregated']['metrics_by_correctness']
        
        if not agg['correct'] or not agg['incorrect']:
            print("Not enough data for both correct and incorrect predictions")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics_to_plot = [
            ('query_to_output_similarity', 'Query-to-Output Similarity'),
            ('output_cluster_tightness', 'Output Cluster Tightness'),
            ('query_norm', 'Query Embedding Norm')
        ]
        
        for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
            if idx >= 4:
                break
            
            ax = axes[idx // 2, idx % 2]
            
            if metric_key in agg['correct']:
                correct_mean = agg['correct'][metric_key]['mean']
                correct_std = agg['correct'][metric_key]['std']
                
                incorrect_mean = agg['incorrect'][metric_key]['mean']
                incorrect_std = agg['incorrect'][metric_key]['std']
                
                layers = range(len(correct_mean))
                
                ax.plot(layers, correct_mean, 'g-o', label='Correct', linewidth=2)
                ax.fill_between(layers, 
                               np.array(correct_mean) - np.array(correct_std),
                               np.array(correct_mean) + np.array(correct_std),
                               alpha=0.2, color='green')
                
                ax.plot(layers, incorrect_mean, 'r-x', label='Incorrect', linewidth=2)
                ax.fill_between(layers,
                               np.array(incorrect_mean) - np.array(incorrect_std),
                               np.array(incorrect_mean) + np.array(incorrect_std),
                               alpha=0.2, color='red')
                
                ax.set_xlabel('Layer', fontsize=11)
                ax.set_ylabel(metric_label, fontsize=11)
                ax.set_title(metric_label, fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Remove extra subplot
        fig.delaxes(axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metric comparison plot to {save_path}")
        plt.close()
    
    def plot_per_prompt_heatmap(self, analysis_results: Dict, save_path: str = "per_prompt_heatmap.png"):
        """Show similarity heatmap for all prompts across layers"""
        
        results = analysis_results['individual_results']
        
        # Build matrix: [n_prompts, n_layers]
        similarity_matrix = []
        prompt_labels = []
        
        for result in results:
            similarities = result['metrics']['query_to_output_similarity']
            similarity_matrix.append(similarities)
            
            # Label with correctness
            correct_mark = "✓" if result['is_correct'] else "✗"
            label = f"{result['variation_name'][:20]} {correct_mark}"
            prompt_labels.append(label)
        
        similarity_matrix = np.array(similarity_matrix)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, max(8, len(results) * 0.4)))
        
        sns.heatmap(similarity_matrix, 
                   xticklabels=range(similarity_matrix.shape[1]),
                   yticklabels=prompt_labels,
                   cmap='RdYlGn',
                   center=0.5,
                   vmin=0,
                   vmax=1,
                   cbar_kws={'label': 'Query-to-Output Similarity'},
                   ax=ax)
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Prompt Variation', fontsize=12)
        ax.set_title('Query-to-Output Similarity Across All Prompts and Layers', fontsize=13)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-prompt heatmap to {save_path}")
        plt.close()
    
    def save_summary(self, analysis_results: Dict, save_path: str = "analysis_summary.txt"):
        """Save text summary of results"""
        
        with open(save_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MULTI-PROMPT ICL ANALYSIS SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            agg = analysis_results['aggregated']
            f.write(f"Total prompts analyzed: {agg['total_prompts']}\n")
            f.write(f"Correct predictions: {agg['correct_count']}\n")
            f.write(f"Incorrect predictions: {agg['incorrect_count']}\n")
            f.write(f"Overall accuracy: {agg['accuracy']:.2%}\n\n")
            
            f.write("="*70 + "\n")
            f.write("INDIVIDUAL RESULTS\n")
            f.write("="*70 + "\n\n")
            
            for i, result in enumerate(analysis_results['individual_results'], 1):
                f.write(f"{i}. {result['variation_name']}\n")
                f.write(f"   Task: {result['task_type']}\n")
                f.write(f"   Expected: {result['expected_answer']}\n")
                f.write(f"   Model: {result['model_answer']}\n")
                f.write(f"   Correct: {'✓ YES' if result['is_correct'] else '✗ NO'}\n\n")
        
        print(f"Saved summary to {save_path}")


def create_animal_sounds_variations() -> List[PromptVariation]:
    """Create variations of animal sounds task"""
    
    variations = []
    
    # Base animals
    base_animals = [
        ("cat", "meow"), ("dog", "bark"), ("cow", "moo"),
        ("duck", "quack"), ("pig", "oink"), ("sheep", "baa"),
        ("horse", "neigh"), ("chicken", "cluck")
    ]
    
    # Variation 1: Standard 5 examples, query sheep
    variations.append(PromptVariation(
        name="Standard_sheep",
        examples=[ICLExample(a, s) for a, s in base_animals[:5]],
        query="sheep",
        expected_answer="baa",
        task_type="animal_sounds"
    ))
    
    # Variation 2: Different 5 examples, query sheep
    variations.append(PromptVariation(
        name="Alt_examples_sheep",
        examples=[ICLExample(a, s) for a, s in [base_animals[i] for i in [1, 2, 4, 6, 7]]],
        query="sheep",
        expected_answer="baa",
        task_type="animal_sounds"
    ))
    
    # Variation 3: Query horse
    variations.append(PromptVariation(
        name="Standard_horse",
        examples=[ICLExample(a, s) for a, s in base_animals[:5]],
        query="horse",
        expected_answer="neigh",
        task_type="animal_sounds"
    ))
    
    # Variation 4: Query chicken
    variations.append(PromptVariation(
        name="Standard_chicken",
        examples=[ICLExample(a, s) for a, s in base_animals[:5]],
        query="chicken",
        expected_answer="cluck",
        task_type="animal_sounds"
    ))
    
    # Variation 5: 3 examples only
    variations.append(PromptVariation(
        name="Few_shot_3_sheep",
        examples=[ICLExample(a, s) for a, s in base_animals[:3]],
        query="sheep",
        expected_answer="baa",
        task_type="animal_sounds"
    ))
    
    # Variation 6: 7 examples
    variations.append(PromptVariation(
        name="Many_shot_7_sheep",
        examples=[ICLExample(a, s) for a, s in base_animals[:7]],
        query="sheep",
        expected_answer="baa",
        task_type="animal_sounds"
    ))
    
    return variations


def create_diverse_task_variations() -> List[PromptVariation]:
    """Create variations across different task types"""
    
    variations = []
    
    # Animal sounds
    variations.append(PromptVariation(
        name="Animal_sounds",
        examples=[
            ICLExample("cat", "meow"),
            ICLExample("dog", "bark"),
            ICLExample("cow", "moo"),
            ICLExample("duck", "quack"),
            ICLExample("pig", "oink"),
        ],
        query="sheep",
        expected_answer="baa",
        task_type="animal_sounds"
    ))
    
    # Antonyms
    variations.append(PromptVariation(
        name="Antonyms",
        examples=[
            ICLExample("hot", "cold"),
            ICLExample("big", "small"),
            ICLExample("fast", "slow"),
            ICLExample("happy", "sad"),
            ICLExample("up", "down"),
        ],
        query="light",
        expected_answer="dark",
        task_type="antonyms"
    ))
    
    # Plurals
    variations.append(PromptVariation(
        name="Plurals",
        examples=[
            ICLExample("cat", "cats"),
            ICLExample("dog", "dogs"),
            ICLExample("box", "boxes"),
            ICLExample("baby", "babies"),
            ICLExample("knife", "knives"),
        ],
        query="wolf",
        expected_answer="wolves",
        task_type="plurals"
    ))
    
    # Simple math
    variations.append(PromptVariation(
        name="Math_addition",
        examples=[
            ICLExample("2+3", "5"),
            ICLExample("4+5", "9"),
            ICLExample("1+8", "9"),
            ICLExample("6+2", "8"),
            ICLExample("3+7", "10"),
        ],
        query="5+4",
        expected_answer="9",
        task_type="math"
    ))
    
    # Rhyming
    variations.append(PromptVariation(
        name="Rhyming",
        examples=[
            ICLExample("cat", "hat"),
            ICLExample("dog", "log"),
            ICLExample("rain", "train"),
            ICLExample("light", "night"),
            ICLExample("moon", "spoon"),
        ],
        query="car",
        expected_answer="star",
        task_type="rhyming"
    ))
    
    return variations


def main():
    """Run multi-prompt analysis"""
    
    print("="*70)
    print("MULTI-PROMPT ICL ANALYSIS")
    print("="*70)
    print("\nThis script analyzes multiple prompt variations to understand:")
    print("  1. How robust ICL mechanisms are")
    print("  2. Which layers perform pattern matching")
    print("  3. Correlation between geometry and performance")
    print("="*70 + "\n")
    
    # Choose analysis type
    print("Choose analysis type:")
    print("  1. Animal sounds variations (same task, different examples/queries)")
    print("  2. Diverse tasks (different task types)")
    print("  3. Both (comprehensive analysis)")
    
    choice = input("\nEnter choice (1/2/3) [default: 1]: ").strip() or "1"
    
    analyzer = MultiPromptAnalyzer()
    
    if choice == "1":
        variations = create_animal_sounds_variations()
        prefix = "animal_variations"
    elif choice == "2":
        variations = create_diverse_task_variations()
        prefix = "diverse_tasks"
    else:
        variations = create_animal_sounds_variations() + create_diverse_task_variations()
        prefix = "comprehensive"
    
    # Run analysis
    results = analyzer.analyze_multiple_prompts(variations)
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    analyzer.plot_accuracy_vs_similarity(results, f"{prefix}_accuracy_vs_similarity.png")
    analyzer.plot_metric_comparison(results, f"{prefix}_metric_comparison.png")
    analyzer.plot_per_prompt_heatmap(results, f"{prefix}_heatmap.png")
    analyzer.save_summary(results, f"{prefix}_summary.txt")
    
    # Print summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nTotal prompts: {results['aggregated']['total_prompts']}")
    print(f"Accuracy: {results['aggregated']['accuracy']:.2%}")
    print(f"  Correct: {results['aggregated']['correct_count']}")
    print(f"  Incorrect: {results['aggregated']['incorrect_count']}")
    print(f"\nGenerated files:")
    print(f"  - {prefix}_accuracy_vs_similarity.png")
    print(f"  - {prefix}_metric_comparison.png")
    print(f"  - {prefix}_heatmap.png")
    print(f"  - {prefix}_summary.txt")
    print("="*70)


if __name__ == "__main__":
    main()
