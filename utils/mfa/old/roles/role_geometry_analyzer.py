"""
Role Geometry Analysis for In-Context Learning

Tests Hypothesis 1: Do models reorganize representations by abstract role?
- Early separation into role-specific regions (input vs output)
- Learned transformation mapping input-role → output-role
- Query trajectory from input region to output region

Key metrics:
1. Role Separability: Distance between input and output centroids
2. Within-role Cohesion: How tightly inputs/outputs cluster
3. Transformation Vector: Consistent direction from inputs → outputs
4. Query Trajectory: Does query move along the transformation?
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ICLExample:
    input_text: str
    output_text: str


class RoleGeometryAnalyzer:
    """Analyze geometric separation and transformation by role in ICL"""
    
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
    
    def construct_prompt(self, examples: List[ICLExample], query: str) -> str:
        """Construct ICL prompt"""
        parts = [f"Input: {ex.input_text}\nOutput: {ex.output_text}" for ex in examples]
        parts.append(f"Input: {query}\nOutput:")
        return "\n".join(parts)
    
    def extract_activations(self, prompt: str, examples: List[ICLExample], query: str) -> Dict:
        """Extract activations for all token positions across layers"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        full_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        
        # Find token positions
        positions = self._find_token_positions(prompt, examples, query)
        
        # Extract activations
        activations_by_layer = {}
        hooks = []
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                hidden_states = output[0]
                if hidden_states.ndim == 3:
                    hidden_states = hidden_states[0]
                activations_by_layer[layer_idx] = hidden_states.detach().cpu().numpy()
            return hook
        
        for idx, layer in enumerate(self.model.model.layers):
            hook = layer.register_forward_hook(make_hook(idx))
            hooks.append(hook)
        
        with torch.no_grad():
            _ = self.model(**inputs)
        
        for hook in hooks:
            hook.remove()
        
        return {
            'activations': activations_by_layer,
            'positions': positions,
            'prompt': prompt,
            'tokens': full_tokens
        }
    
    def _find_token_positions(self, prompt: str, examples: List[ICLExample], query: str) -> Dict:
        """Find positions of input tokens, output tokens, and query"""
        
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
            'query_position': None,
            'final_position': len(full_tokens) - 1
        }
        
        current_start = 0
        for ex in examples:
            inp_pos = find_text_position(ex.input_text, current_start)
            positions['input_positions'].append(inp_pos)
            current_start = inp_pos + 1
            
            out_pos = find_text_position(ex.output_text, current_start)
            positions['output_positions'].append(out_pos)
            current_start = out_pos + 1
        
        positions['query_position'] = find_text_position(query, current_start)
        
        return positions
    
    def compute_role_geometry_metrics(self, data: Dict) -> Dict:
        """Compute all role geometry metrics across layers"""
        
        activations = data['activations']
        positions = data['positions']
        
        metrics = {
            'role_separability': [],           # Distance between input & output centroids
            'within_input_similarity': [],     # Avg pairwise similarity among inputs
            'within_output_similarity': [],    # Avg pairwise similarity among outputs
            'between_role_similarity': [],     # Avg similarity between inputs & outputs
            'transformation_norm': [],         # Magnitude of transformation vector
            'query_to_input_centroid': [],     # Distance from query to input centroid
            'query_to_output_centroid': [],    # Distance from query to output centroid
            'query_projection_on_transform': [], # How much query aligns with transformation
            'final_to_output_centroid': [],    # Distance from final position to output centroid
            'transformation_vectors': [],      # The actual transformation vectors per layer
            'input_centroids': [],            # Input centroids per layer
            'output_centroids': [],           # Output centroids per layer
        }
        
        for layer_idx in sorted(activations.keys()):
            hidden = activations[layer_idx]
            
            # Extract role-specific representations
            input_vecs = hidden[positions['input_positions']]
            output_vecs = hidden[positions['output_positions']]
            query_vec = hidden[positions['query_position']]
            final_vec = hidden[positions['final_position']]
            
            # Compute centroids
            input_centroid = input_vecs.mean(axis=0)
            output_centroid = output_vecs.mean(axis=0)
            
            metrics['input_centroids'].append(input_centroid)
            metrics['output_centroids'].append(output_centroid)
            
            # 1. Role Separability: Distance between centroids
            separability = np.linalg.norm(output_centroid - input_centroid)
            metrics['role_separability'].append(separability)
            
            # 2. Within-role similarity (cohesion)
            if len(input_vecs) > 1:
                input_sims = self._pairwise_cosine_similarity(input_vecs)
                metrics['within_input_similarity'].append(input_sims.mean())
            else:
                metrics['within_input_similarity'].append(1.0)
            
            if len(output_vecs) > 1:
                output_sims = self._pairwise_cosine_similarity(output_vecs)
                metrics['within_output_similarity'].append(output_sims.mean())
            else:
                metrics['within_output_similarity'].append(1.0)
            
            # 3. Between-role similarity
            between_sims = []
            for iv in input_vecs:
                for ov in output_vecs:
                    between_sims.append(self._cosine_similarity(iv, ov))
            metrics['between_role_similarity'].append(np.mean(between_sims))
            
            # 4. Transformation vector
            transformation = output_centroid - input_centroid
            metrics['transformation_vectors'].append(transformation)
            metrics['transformation_norm'].append(np.linalg.norm(transformation))
            
            # 5. Query trajectory
            query_to_input = np.linalg.norm(query_vec - input_centroid)
            query_to_output = np.linalg.norm(query_vec - output_centroid)
            metrics['query_to_input_centroid'].append(query_to_input)
            metrics['query_to_output_centroid'].append(query_to_output)
            
            # 6. Query alignment with transformation
            query_direction = query_vec - input_centroid
            projection = np.dot(query_direction, transformation) / (np.linalg.norm(transformation) + 1e-8)
            metrics['query_projection_on_transform'].append(projection)
            
            # 7. Final position to output centroid
            final_to_output = np.linalg.norm(final_vec - output_centroid)
            metrics['final_to_output_centroid'].append(final_to_output)
        
        return metrics
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity between two vectors"""
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        return np.dot(v1_norm, v2_norm)
    
    def _pairwise_cosine_similarity(self, vecs: np.ndarray) -> np.ndarray:
        """Pairwise cosine similarity among vectors"""
        sims = []
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                sims.append(self._cosine_similarity(vecs[i], vecs[j]))
        return np.array(sims)
    
    def test_transformation_generalization(self, data: Dict, metrics: Dict, 
                                          expected_output: str) -> Dict:
        """Test if transformation vector can predict the output"""
        
        activations = data['activations']
        positions = data['positions']
        
        # Get output token embedding
        output_token_id = self.tokenizer.encode(expected_output, add_special_tokens=False)[0]
        output_embedding = self.model.model.embed_tokens.weight[output_token_id].detach().cpu().numpy()
        
        predictions = {
            'layer': [],
            'prediction_similarity': [],  # Similarity between (query + transform) and actual output embedding
            'direct_output_similarity': [], # Direct similarity between final position and output embedding
        }
        
        for layer_idx in sorted(activations.keys()):
            hidden = activations[layer_idx]
            query_vec = hidden[positions['query_position']]
            final_vec = hidden[positions['final_position']]
            transformation = metrics['transformation_vectors'][layer_idx]
            
            # Apply transformation to query
            predicted_output_vec = query_vec + transformation
            
            # Measure similarity to actual output embedding
            pred_sim = self._cosine_similarity(predicted_output_vec, output_embedding)
            predictions['prediction_similarity'].append(pred_sim)
            
            # Also measure direct similarity
            direct_sim = self._cosine_similarity(final_vec, output_embedding)
            predictions['direct_output_similarity'].append(direct_sim)
            
            predictions['layer'].append(layer_idx)
        
        return predictions
    
    def visualize_role_geometry(self, data: Dict, save_prefix: str = "role_geometry"):
        """Create comprehensive visualizations of role geometry"""
        
        metrics = self.compute_role_geometry_metrics(data)
        
        # 1. Core metrics plot
        self._plot_core_metrics(metrics, f"{save_prefix}_core_metrics.png")
        
        # 2. Query trajectory plot
        self._plot_query_trajectory(metrics, f"{save_prefix}_query_trajectory.png")
        
        # 3. Role cohesion plot
        self._plot_role_cohesion(metrics, f"{save_prefix}_role_cohesion.png")
        
        # 4. PCA visualization (multiple layers)
        self._plot_pca_evolution(data, f"{save_prefix}_pca_evolution.png")
        
        return metrics
    
    def _plot_core_metrics(self, metrics: Dict, save_path: str):
        """Plot role separability and transformation metrics"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        layers = range(len(metrics['role_separability']))
        
        # 1. Role Separability
        ax = axes[0, 0]
        ax.plot(layers, metrics['role_separability'], 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Euclidean Distance', fontsize=12)
        ax.set_title('Role Separability\n(Distance between Input & Output Centroids)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 2. Transformation Magnitude
        ax = axes[0, 1]
        ax.plot(layers, metrics['transformation_norm'], 'g-s', linewidth=2, markersize=6)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Vector Norm', fontsize=12)
        ax.set_title('Transformation Vector Magnitude', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 3. Query Projection on Transformation
        ax = axes[1, 0]
        ax.plot(layers, metrics['query_projection_on_transform'], 'r-^', linewidth=2, markersize=6)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Projection (unnormalized)', fontsize=12)
        ax.set_title('Query Alignment with Transformation\n(Positive = moving toward outputs)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 4. Final Position to Output Centroid
        ax = axes[1, 1]
        ax.plot(layers, metrics['final_to_output_centroid'], 'm-d', linewidth=2, markersize=6)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Euclidean Distance', fontsize=12)
        ax.set_title('Final Position Distance to Output Centroid', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved core metrics plot to {save_path}")
        plt.close()
    
    def _plot_query_trajectory(self, metrics: Dict, save_path: str):
        """Plot query's movement from input region to output region"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        layers = range(len(metrics['query_to_input_centroid']))
        
        # 1. Distances
        ax1.plot(layers, metrics['query_to_input_centroid'], 'b-o', 
                label='Query → Input Centroid', linewidth=2, markersize=6)
        ax1.plot(layers, metrics['query_to_output_centroid'], 'r-s', 
                label='Query → Output Centroid', linewidth=2, markersize=6)
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Euclidean Distance', fontsize=12)
        ax1.set_title('Query Position Relative to Role Centroids', fontsize=13)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 2. Relative position (which is it closer to?)
        relative_pos = []
        for i in range(len(layers)):
            # Positive = closer to output, Negative = closer to input
            diff = metrics['query_to_input_centroid'][i] - metrics['query_to_output_centroid'][i]
            relative_pos.append(diff)
        
        ax2.plot(layers, relative_pos, 'g-^', linewidth=2, markersize=6)
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=1.5, 
                   label='Equidistant (0)')
        ax2.fill_between(layers, 0, relative_pos, 
                        where=[x > 0 for x in relative_pos],
                        alpha=0.3, color='red', label='Closer to Outputs')
        ax2.fill_between(layers, 0, relative_pos,
                        where=[x < 0 for x in relative_pos],
                        alpha=0.3, color='blue', label='Closer to Inputs')
        ax2.set_xlabel('Layer', fontsize=12)
        ax2.set_ylabel('Distance Difference', fontsize=12)
        ax2.set_title('Query Trajectory\n(+ = toward outputs, - = toward inputs)', fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved query trajectory plot to {save_path}")
        plt.close()
    
    def _plot_role_cohesion(self, metrics: Dict, save_path: str):
        """Plot within-role vs between-role similarity"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        layers = range(len(metrics['within_input_similarity']))
        
        ax.plot(layers, metrics['within_input_similarity'], 'b-o', 
               label='Within Inputs', linewidth=2, markersize=6)
        ax.plot(layers, metrics['within_output_similarity'], 'r-s', 
               label='Within Outputs', linewidth=2, markersize=6)
        ax.plot(layers, metrics['between_role_similarity'], 'k--^', 
               label='Between Roles', linewidth=2, markersize=6, alpha=0.7)
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title('Role Cohesion: Within-Role vs Between-Role Similarity', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Add interpretation guide
        ax.text(0.02, 0.98, 
               'Good role separation:\n• High within-role similarity\n• Low between-role similarity',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved role cohesion plot to {save_path}")
        plt.close()
    
    def _plot_pca_evolution(self, data: Dict, save_path: str):
        """Visualize role geometry evolution using PCA"""
        
        activations = data['activations']
        positions = data['positions']
        n_layers = len(activations)
        
        # Select 4 evenly spaced layers to visualize
        layer_indices = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, layer_idx in enumerate(layer_indices):
            ax = axes[idx]
            hidden = activations[layer_idx]
            
            # Collect all relevant vectors
            input_vecs = hidden[positions['input_positions']]
            output_vecs = hidden[positions['output_positions']]
            query_vec = hidden[positions['query_position']].reshape(1, -1)
            final_vec = hidden[positions['final_position']].reshape(1, -1)
            
            # Combine for PCA
            all_vecs = np.vstack([input_vecs, output_vecs, query_vec, final_vec])
            
            # PCA
            pca = PCA(n_components=2)
            projected = pca.fit_transform(all_vecs)
            
            n_inputs = len(input_vecs)
            n_outputs = len(output_vecs)
            
            # Split back
            inputs_2d = projected[:n_inputs]
            outputs_2d = projected[n_inputs:n_inputs+n_outputs]
            query_2d = projected[n_inputs+n_outputs]
            final_2d = projected[n_inputs+n_outputs+1]
            
            # Plot
            ax.scatter(inputs_2d[:, 0], inputs_2d[:, 1], 
                      c='blue', s=100, alpha=0.6, label='Inputs', marker='o')
            ax.scatter(outputs_2d[:, 0], outputs_2d[:, 1], 
                      c='red', s=100, alpha=0.6, label='Outputs', marker='s')
            ax.scatter(query_2d[0], query_2d[1], 
                      c='green', s=200, alpha=0.8, label='Query', marker='^', 
                      edgecolors='black', linewidths=2)
            ax.scatter(final_2d[0], final_2d[1], 
                      c='purple', s=200, alpha=0.8, label='Final Position', marker='*',
                      edgecolors='black', linewidths=2)
            
            # Draw centroids
            input_centroid_2d = inputs_2d.mean(axis=0)
            output_centroid_2d = outputs_2d.mean(axis=0)
            ax.scatter(input_centroid_2d[0], input_centroid_2d[1],
                      c='darkblue', s=300, marker='X', edgecolors='black', linewidths=2,
                      label='Input Centroid', zorder=5)
            ax.scatter(output_centroid_2d[0], output_centroid_2d[1],
                      c='darkred', s=300, marker='X', edgecolors='black', linewidths=2,
                      label='Output Centroid', zorder=5)
            
            # Draw transformation arrow
            ax.arrow(input_centroid_2d[0], input_centroid_2d[1],
                    output_centroid_2d[0] - input_centroid_2d[0],
                    output_centroid_2d[1] - input_centroid_2d[1],
                    head_width=0.3, head_length=0.2, fc='black', ec='black',
                    linewidth=2, alpha=0.5, length_includes_head=True)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
            ax.set_title(f'Layer {layer_idx} Role Geometry', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PCA evolution plot to {save_path}")
        plt.close()
    
    def generate_report(self, data: Dict, metrics: Dict, save_path: str = "role_geometry_report.txt"):
        """Generate text report of findings"""
        
        with open(save_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ROLE GEOMETRY ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("HYPOTHESIS 1 TEST RESULTS\n")
            f.write("-"*70 + "\n")
            f.write("Does the model reorganize representations by abstract role?\n\n")
            
            # 1. Role Separability
            separability = metrics['role_separability']
            early_sep = np.mean(separability[:len(separability)//3])
            mid_sep = np.mean(separability[len(separability)//3:2*len(separability)//3])
            late_sep = np.mean(separability[2*len(separability)//3:])
            
            f.write(f"1. ROLE SEPARABILITY (Input vs Output Centroid Distance)\n")
            f.write(f"   Early layers (0-33%):  {early_sep:.3f}\n")
            f.write(f"   Middle layers (33-66%): {mid_sep:.3f}\n")
            f.write(f"   Late layers (66-100%): {late_sep:.3f}\n")
            
            if mid_sep > early_sep:
                f.write(f"   ✓ Separation INCREASES in middle layers (supports Hypothesis 1)\n")
            else:
                f.write(f"   ✗ Separation does not increase clearly\n")
            f.write("\n")
            
            # 2. Within-role vs Between-role similarity
            within_input = np.mean(metrics['within_input_similarity'])
            within_output = np.mean(metrics['within_output_similarity'])
            between_role = np.mean(metrics['between_role_similarity'])
            
            f.write(f"2. ROLE COHESION (Average across all layers)\n")
            f.write(f"   Within-input similarity:  {within_input:.3f}\n")
            f.write(f"   Within-output similarity: {within_output:.3f}\n")
            f.write(f"   Between-role similarity:  {between_role:.3f}\n")
            
            if within_input > between_role and within_output > between_role:
                f.write(f"   ✓ Within-role > Between-role (supports role clustering)\n")
            else:
                f.write(f"   ✗ No clear role-based clustering\n")
            f.write("\n")
            
            # 3. Query trajectory
            query_trajectory = metrics['query_to_output_centroid']
            early_dist = query_trajectory[0]
            late_dist = query_trajectory[-1]
            
            f.write(f"3. QUERY TRAJECTORY (Distance to Output Centroid)\n")
            f.write(f"   Early layers: {early_dist:.3f}\n")
            f.write(f"   Late layers:  {late_dist:.3f}\n")
            f.write(f"   Change:       {late_dist - early_dist:.3f}\n")
            
            if late_dist < early_dist:
                f.write(f"   ✓ Query moves TOWARD output region (supports transformation)\n")
            else:
                f.write(f"   ✗ Query does not clearly move toward outputs\n")
            f.write("\n")
            
            # 4. Transformation consistency
            transform_norms = metrics['transformation_norm']
            transform_variation = np.std(transform_norms) / np.mean(transform_norms)
            
            f.write(f"4. TRANSFORMATION CONSISTENCY\n")
            f.write(f"   Mean magnitude: {np.mean(transform_norms):.3f}\n")
            f.write(f"   Std deviation:  {np.std(transform_norms):.3f}\n")
            f.write(f"   Coefficient of variation: {transform_variation:.3f}\n")
            
            if transform_variation < 0.3:
                f.write(f"   ✓ Transformation is relatively consistent across layers\n")
            else:
                f.write(f"   ~ Transformation varies substantially across layers\n")
            f.write("\n")
            
            # Summary
            f.write("="*70 + "\n")
            f.write("SUMMARY\n")
            f.write("="*70 + "\n")
            f.write("Evidence for Hypothesis 1 (role-based geometry):\n\n")
            
            score = 0
            if mid_sep > early_sep:
                f.write("✓ Role separation emerges\n")
                score += 1
            if within_input > between_role and within_output > between_role:
                f.write("✓ Role-based clustering observed\n")
                score += 1
            if late_dist < early_dist:
                f.write("✓ Query moves toward output region\n")
                score += 1
            if transform_variation < 0.3:
                f.write("✓ Consistent transformation vector\n")
                score += 1
            
            f.write(f"\nEvidence score: {score}/4\n")
            
            if score >= 3:
                f.write("\nConclusion: STRONG support for Hypothesis 1\n")
            elif score >= 2:
                f.write("\nConclusion: MODERATE support for Hypothesis 1\n")
            else:
                f.write("\nConclusion: WEAK support for Hypothesis 1\n")
        
        print(f"Saved report to {save_path}")


def run_analysis(examples: List[ICLExample], query: str, expected_output: str,
                task_name: str = "test"):
    """Run complete role geometry analysis"""
    
    print("="*70)
    print(f"ROLE GEOMETRY ANALYSIS: {task_name}")
    print("="*70)
    
    analyzer = RoleGeometryAnalyzer()
    
    # 1. Extract activations
    print("\n1. Extracting activations...")
    prompt = analyzer.construct_prompt(examples, query)
    print(f"   Prompt:\n{prompt}\n")
    
    data = analyzer.extract_activations(prompt, examples, query)
    
    # 2. Compute metrics
    print("2. Computing role geometry metrics...")
    metrics = analyzer.compute_role_geometry_metrics(data)
    
    # 3. Test transformation generalization
    print("3. Testing transformation generalization...")
    predictions = analyzer.test_transformation_generalization(data, metrics, expected_output)
    
    # 4. Generate visualizations
    print("\n4. Generating visualizations...")
    analyzer.visualize_role_geometry(data, save_prefix=f"{task_name}_role_geometry")
    
    # 5. Generate report
    print("5. Generating report...")
    analyzer.generate_report(data, metrics, f"{task_name}_role_geometry_report.txt")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  - {task_name}_role_geometry_core_metrics.png")
    print(f"  - {task_name}_role_geometry_query_trajectory.png")
    print(f"  - {task_name}_role_geometry_role_cohesion.png")
    print(f"  - {task_name}_role_geometry_pca_evolution.png")
    print(f"  - {task_name}_role_geometry_report.txt")
    print("="*70)
    
    return data, metrics, predictions


def main():
    """Run example analyses"""
    
    print("ROLE GEOMETRY ANALYZER")
    print("Testing Hypothesis 1: Role-based geometric separation in ICL")
    print("\nChoose a task to analyze:")
    print("  1. Animal sounds")
    print("  2. Antonyms")
    print("  3. Country capitals")
    print("  4. Custom task")
    
    choice = input("\nEnter choice (1/2/3/4) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        examples = [
            ICLExample("cat", "meow"),
            ICLExample("dog", "bark"),
            ICLExample("cow", "moo"),
            ICLExample("duck", "quack"),
            ICLExample("pig", "oink"),
        ]
        query = "sheep"
        expected = "baa"
        task_name = "animal_sounds"
    
    elif choice == "2":
        examples = [
            ICLExample("hot", "cold"),
            ICLExample("big", "small"),
            ICLExample("fast", "slow"),
            ICLExample("happy", "sad"),
            ICLExample("up", "down"),
        ]
        query = "light"
        expected = "dark"
        task_name = "antonyms"
    
    elif choice == "3":
        examples = [
            ICLExample("France", "Paris"),
            ICLExample("Germany", "Berlin"),
            ICLExample("Italy", "Rome"),
            ICLExample("Spain", "Madrid"),
            ICLExample("Japan", "Tokyo"),
        ]
        query = "China"
        expected = "Beijing"
        task_name = "capitals"
    
    else:
        print("\nCustom task - enter your examples:")
        examples = []
        while True:
            inp = input(f"  Input {len(examples)+1} (or press Enter to finish): ").strip()
            if not inp:
                break
            out = input(f"  Output {len(examples)+1}: ").strip()
            examples.append(ICLExample(inp, out))
        
        query = input("Query: ").strip()
        expected = input("Expected output: ").strip()
        task_name = "custom"
    
    # Run analysis
    data, metrics, predictions = run_analysis(examples, query, expected, task_name)


if __name__ == "__main__":
    main()
