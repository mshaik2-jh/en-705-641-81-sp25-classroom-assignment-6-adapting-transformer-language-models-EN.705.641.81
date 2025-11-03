import random
import openai
from datasets import load_dataset

# Load BoolQ dataset
dataset = load_dataset("boolq", split="validation")
random.seed(42)
client = openai.OpenAI(api_key= "removed for hw6")

# Separate yes/no examples
yes_examples = [ex for ex in dataset if ex["answer"] is True]
no_examples = [ex for ex in dataset if ex["answer"] is False]

def create_few_shot_prompt(target_example, n_shots=8):
    """
    Create a balanced few-shot prompt with n_shots demonstrations.
    Half yes, half no, interleaved.
    """
    n_per_label = n_shots // 2
    
    selected_yes = random.sample(yes_examples, n_per_label)
    selected_no = random.sample(no_examples, n_per_label)
    
    # Interleave yes/no
    demonstrations = []
    for y, n in zip(selected_yes, selected_no):
        demonstrations.append(y)
        demonstrations.append(n)
    
    # Format demonstrations
    demo_text = ""
    for demo in demonstrations:
        demo_text += f"Passage: {demo['passage']}\nQuestion: {demo['question']}\nAnswer: {'yes' if demo['answer'] else 'no'}\n\n"
    
    # Add the target example (without answer)
    demo_text += f"Passage: {target_example['passage']}\nQuestion: {target_example['question']}\nAnswer:"
    
    return demo_text

# Evaluation subset
eval_subset = random.sample(list(dataset), 30)
predictions = []

for example in eval_subset:
    prompt = create_few_shot_prompt(example)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    answer_text = response.choices[0].message.content.strip().lower()
    
    if "yes" in answer_text:
        pred = True
    elif "no" in answer_text:
        pred = False
    else:
        pred = None
    
    predictions.append({
        "passage": example["passage"],
        "question": example["question"],
        "true_answer": example["answer"],
        "predicted": pred
    })

# Compute accuracy
valid_preds = [p for p in predictions if p["predicted"] is not None]
accuracy = sum(p["true_answer"] == p["predicted"] for p in valid_preds) / len(valid_preds)

print(f"Evaluated {len(valid_preds)} examples. Accuracy: {accuracy:.2f}")
