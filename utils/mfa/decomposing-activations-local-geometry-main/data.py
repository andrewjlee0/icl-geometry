import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ICLExample:
    """A single input-output demonstration."""
    input_token: str
    output_token: str


@dataclass  
class ICLPrompt:
    """Full ICL prompt with demonstrations and query."""
    demonstrations: List[ICLExample]
    query_input: str
    expected_output: str
    relation_name: str


def create_icl_tasks(n_prompts_per_task=8, n_demos_per_prompt=3) -> List[ICLPrompt]:
    """Create ICL tasks with multiple prompts per task, each with 3 demonstrations."""
    tasks = []
    
    # Example pools for each relation type
    # Each pool has (input, output) pairs and I'll sample different subsets
    
    # antonym_pairs = [
    #     ("hot", "cold"), ("big", "small"), ("fast", "slow"), ("up", "down"),
    #     ("left", "right"), ("old", "new"), ("rich", "poor"), ("dark", "light"),
    #     ("hard", "soft"), ("wet", "dry"), ("loud", "quiet"), ("early", "late"),
    #     ("good", "bad"), ("happy", "sad"), ("tall", "short"), ("thin", "thick"),
    # ]
    
    capital_pairs = [
        ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Egypt", "Cairo"),
        ("Cuba", "Havana"), ("Peru", "Lima"), ("Greece", "Athens"), ("Poland", "Warsaw"),
        ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Austria", "Vienna"), ("Spain", "Madrid"),
        ("China", "Beijing"), ("Russia", "Moscow"), ("Germany", "Berlin"), ("India", "Delhi"),
        ("Canada", "Ottawa"), ("Brazil", "Brasilia"), ("Australia", "Canberra"),
        ("Mexico", "Mexico City"), ("Kenya", "Nairobi"),
    ]
    
    past_tense_pairs = [
        ("run", "ran"), ("eat", "ate"), ("go", "went"), ("see", "saw"),
        ("take", "took"), ("give", "gave"), ("come", "came"), ("know", "knew"),
        ("think", "thought"), ("bring", "brought"), ("buy", "bought"), ("catch", "caught"),
        ("teach", "taught"), ("find", "found"), ("tell", "told"), ("sell", "sold"),
        ("swim", "swam"), ("drink", "drank"), ("begin", "began"),
        ("drive", "drove"), ("write", "wrote"),
    ]
    
    plural_pairs = [
        ("cat", "cats"), ("dog", "dogs"), ("car", "cars"), ("tree", "trees"),
        ("house", "houses"), ("bird", "birds"), ("book", "books"), ("chair", "chairs"),
        ("table", "tables"), ("phone", "phones"), ("lamp", "lamps"), ("door", "doors"),
        ("ball", "balls"), ("cup", "cups"), ("hat", "hats"), ("key", "keys"),
        ("box", "boxes"), ("baby", "babies"), ("city", "cities"),
        ("knife", "knives"), ("leaf", "leaves"),
    ]
    
    language_pairs = [
        ("France", "French"), ("Spain", "Spanish"), ("Germany", "German"), ("Italy", "Italian"),
        ("Portugal", "Portuguese"), ("Russia", "Russian"), ("Japan", "Japanese"), ("China", "Chinese"),
        ("Poland", "Polish"), ("Sweden", "Swedish"), ("Finland", "Finnish"), ("Turkey", "Turkish"),
        ("Greece", "Greek"), ("Denmark", "Danish"), ("Norway", "Norwegian"), ("Holland", "Dutch"),
        ("Ireland", "Irish"), ("Hungary", "Hungarian"), ("Czech Republic", "Czech"),
        ("Ukraine", "Ukrainian"), ("Iceland", "Icelandic"),
    ]
    
    gender_pairs = [
        ("king", "queen"), ("man", "woman"), ("boy", "girl"), ("father", "mother"),
        ("son", "daughter"), ("brother", "sister"), ("uncle", "aunt"), ("husband", "wife"),
        ("actor", "actress"), ("prince", "princess"), ("hero", "heroine"), ("waiter", "waitress"),
        ("god", "goddess"), ("host", "hostess"), ("lion", "lioness"), ("emperor", "empress"),
        ("nephew", "niece"), ("monk", "nun"), ("sir", "madam"),
        ("bull", "cow"), ("rooster", "hen"),
    ]
    
    comparative_pairs = [
        ("big", "bigger"), ("small", "smaller"), ("fast", "faster"), ("slow", "slower"),
        ("tall", "taller"), ("short", "shorter"), ("old", "older"), ("young", "younger"),
        ("hot", "hotter"), ("cold", "colder"), ("loud", "louder"), ("soft", "softer"),
        ("hard", "harder"), ("weak", "weaker"), ("strong", "stronger"), ("bright", "brighter"),
        ("easy", "easier"), ("heavy", "heavier"), ("happy", "happier"),
        ("pretty", "prettier"), ("high", "higher"),
    ]
    
    agent_pairs = [
        ("teach", "teacher"), ("write", "writer"), ("read", "reader"), ("play", "player"),
        ("sing", "singer"), ("dance", "dancer"), ("drive", "driver"), ("lead", "leader"),
        ("work", "worker"), ("build", "builder"), ("paint", "painter"), ("farm", "farmer"),
        ("hunt", "hunter"), ("bank", "banker"), ("deal", "dealer"), ("dream", "dreamer"),
        ("bake", "baker"), ("clean", "cleaner"), ("design", "designer"),
        ("research", "researcher"), ("program", "programmer"),
    ]
    
    # Create multiple prompts per task type
    # Each prompt uses 3 demos and a different query
    
    def make_prompts(pairs, relation_name, n_prompts=8, n_demos=3):
        """Create n_prompts different prompts from the pair pool."""
        prompts = []
        n_pairs = len(pairs)
        
        for i in range(n_prompts):
            # Select 3 demo pairs (non-overlapping with query)
            # Use different "starting points" to get variety
            start_idx = (i * 4) % n_pairs
            demo_indices = [(start_idx + j) % n_pairs for j in range(n_demos)]
            query_idx = (start_idx + 3) % n_pairs
            
            demos = [ICLExample(pairs[idx][0], pairs[idx][1]) for idx in demo_indices]
            query_input, expected_output = pairs[query_idx]
            
            prompts.append(ICLPrompt(
                demonstrations=demos,
                query_input=query_input,
                expected_output=expected_output,
                relation_name=relation_name
            ))
        
        return prompts
    
    # Generate prompts for each relation type
    # tasks.extend(make_prompts(antonym_pairs, "antonym", n_prompts=n_prompts_per_task, n_demos=n_demos_per_prompt))
    tasks.extend(make_prompts(capital_pairs, "capital", n_prompts=n_prompts_per_task, n_demos=n_demos_per_prompt))
    tasks.extend(make_prompts(past_tense_pairs, "past_tense", n_prompts=n_prompts_per_task, n_demos=n_demos_per_prompt))
    tasks.extend(make_prompts(plural_pairs, "plural", n_prompts=n_prompts_per_task, n_demos=n_demos_per_prompt))
    tasks.extend(make_prompts(language_pairs, "language", n_prompts=n_prompts_per_task, n_demos=n_demos_per_prompt))
    tasks.extend(make_prompts(gender_pairs, "gender", n_prompts=n_prompts_per_task, n_demos=n_demos_per_prompt))
    tasks.extend(make_prompts(comparative_pairs, "comparative", n_prompts=n_prompts_per_task, n_demos=n_demos_per_prompt))
    tasks.extend(make_prompts(agent_pairs, "agent_noun", n_prompts=n_prompts_per_task, n_demos=n_demos_per_prompt))
    
    return tasks


def format_prompt(task: ICLPrompt) -> str:
    """Format ICL task as a string prompt."""
    lines = []
    for demo in task.demonstrations:
        lines.append(f"{demo.input_token} -> {demo.output_token}")
    lines.append(f"{task.query_input} ->")
    return "\n".join(lines)


if __name__ == "__main__":
    n_prompts_per_task = 8
    n_demos_per_prompt = 3
    prompts = create_icl_tasks(n_prompts_per_task, n_demos_per_prompt)
    print('Number of prompts', len(prompts))

    for prompt in prompts:
        print(prompt)
        print()
        print(format_prompt(prompt))
        print()