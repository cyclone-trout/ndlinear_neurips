import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM
)
import torch
from safetensors.torch import load_file
import numpy as np
import evaluate
from transformers import GenerationConfig

parser = argparse.ArgumentParser(description="Load model weights from specified paths.")
parser.add_argument("--ckpt_path", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--max_input_length", type=str, required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "facebook/opt-1.3b"
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", use_fast=False)
tokenizer.padding_side = "left"

benchmark = args.dataset

if benchmark == "squad":
    raw_datasets = load_dataset("squad")
elif benchmark == "arc-easy":
    raw_datasets = load_dataset("ai2_arc", "ARC-Easy")
elif benchmark == "arc-challenge":
    raw_datasets = load_dataset("ai2_arc", "ARC-Challenge")
elif benchmark == "openbookqa":
    raw_datasets = load_dataset("openbookqa")
elif benchmark == "hellaswag":
    raw_datasets = load_dataset("hellaswag")
elif benchmark == "piqa":
    raw_datasets = load_dataset("piqa")
elif benchmark == "winogrande":
    raw_datasets = load_dataset("winogrande", 'winogrande_m')
elif benchmark == "copa":
    raw_datasets = load_dataset("super_glue", 'copa')
elif benchmark == "boolq":
    raw_datasets = load_dataset("super_glue", 'boolq')
elif benchmark == "cb":
    raw_datasets = load_dataset("super_glue", 'cb')
elif benchmark == "wic":
    raw_datasets = load_dataset("super_glue", 'wic')
elif benchmark == "rte":
    raw_datasets = load_dataset("super_glue", 'rte')
elif benchmark == "wsc":
    raw_datasets = load_dataset("super_glue", 'wsc')
else:
    raise ValueError("Invalid dataset.")

def build_input_prompt(example):
    if benchmark == "squad":
        context = example["context"]
        question = example["question"]
        return f"Context: {context}\nQuestion: {question}\nAnswer:"
    elif benchmark == "arc-easy" or benchmark == "arc-challenge":
        question = example["question"]
        choices = ""
        for label, choice in zip(example["choices"]["label"], example["choices"]["text"]):
            choices += f"{label}. {choice}\n"   
        return f"Question: {question}\nOptions:\n{choices}\nThe best option to the question is "
    elif benchmark == "openbookqa":
        question = example["question_stem"]
        choices = ""
        for label, choice in zip(example["choices"]["label"], example["choices"]["text"]):
            choices += f"{label}. {choice}\n"
        return f"Question: {question}\nOptions:\n{choices}\nThe best option to the question is "
    elif benchmark == "hellaswag":
        context = example["ctx"]
        return context
    elif benchmark == "piqa":
        goal = example["goal"]
        solution_1 = example["sol1"]
        solution_2 = example["sol2"]
        return f"Goal: {goal}\nPossible solutions:\n1. {solution_1}\n2. {solution_2}\nThe best solution is "
    elif benchmark == "winogrande":
        sentence = example["sentence"]
        option1 = example["option1"]
        option2 = example["option2"]
        return f"Sentence: {sentence}\nOptions:\n1. {option1}\n2. {option2}\nThe correct option that fits and completes the sentence is "
    elif benchmark == "copa":
        premise = example["premise"]
        question = example["question"]
        choice1 = example["choice1"]
        choice2 = example["choice2"]
        return f"Premise: {premise}\nQuestion: What is the {question}?\nOptions:\n1. {choice1}\n2. {choice2}\nThe most plausible option is "
    elif benchmark == "boolq":
        passage = example["passage"]
        question = example["question"]
        return f"Passage: {passage}\nQuestion: {question}\nAnswer (Yes or No): "
    elif benchmark == "cb":
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        return f"Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise entail the hypothesis? (Answer with 'Entailment', 'Contradiction', or 'Neutral'): "
    elif benchmark == "wic":
        word = example["word"]
        sentence1 = example["sentence1"]
        sentence2 = example["sentence2"]
        return f"Target Word: {word}\nSentence 1: {sentence1}\nSentence 2: {sentence2}\nDoes the target word have the same meaning in both sentences? (Answer with 'Yes' or 'No'): "
    elif benchmark == "rte":
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        return f"Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise entail the hypothesis? (Answer with 'Yes' or 'No'): "
    elif benchmark == "wsc":
        text = example["text"]
        span1 = example["span1_text"]
        span2 = example["span2_text"]
        return f"Text: {text}\nDoes '{span1}' refer to '{span2}' in the given context? (Answer with 'Yes' or 'No'): "
    else:
        raise ValueError("Invalid benchmark. Please choose from 'squad', 'arc-easy', 'arc-challenge', 'openbookqa'.")

def filter_long_inputs(example):
    input_text = build_input_prompt(example)
    input_length = len(tokenizer(input_text)["input_ids"])
    return input_length <= int(args.max_input_length)

if "validation" in raw_datasets:
    filtered_dataset = raw_datasets["validation"].filter(filter_long_inputs)
elif "train" in raw_datasets:
    filtered_dataset = raw_datasets["train"].filter(filter_long_inputs)
else:
    filtered_dataset = raw_datasets.filter(filter_long_inputs)

def compute_log_likelihood(input_text, answer_text):
    input_encoding = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
    answer_encoding = tokenizer(answer_text, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True).to(device)

    input_ids = input_encoding["input_ids"]
    answer_ids = answer_encoding["input_ids"]
    input_mask = input_encoding["attention_mask"]
    answer_mask = answer_encoding["attention_mask"]

    full_input_ids = torch.cat([input_ids, answer_ids], dim=1)
    full_attention_mask = torch.cat([input_mask, answer_mask], dim=1)

    with torch.no_grad():
        outputs = model(full_input_ids, attention_mask=full_attention_mask)

    logits = outputs.logits

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    log_likelihood = 0
    
    for i in range(answer_ids.shape[1]):
        log_likelihood += log_probs[0, input_ids.shape[1] + i - 1, answer_ids[0, i]].item()
    length_normalized_log_likelihood = log_likelihood

    return length_normalized_log_likelihood

model.eval()
model.generation_config = GenerationConfig.from_pretrained("facebook/opt-1.3b")

def extract_answer(text):
    return text.split("Answer:")[-1].strip() if "Answer:" in text else text.strip()

def generate_full_answer(batch):
    batch_input = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
    input_texts = [build_input_prompt(e) for e in batch_input]
    inputs = tokenizer(input_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=512,
        num_beams=1,
    )

    generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    final_answers = [extract_answer(text) for text in generated_texts]
    for input_prompt, prediction in zip(input_texts, final_answers):
        print("\n----------------------------")
        print(f"{input_prompt}\n")
        print(f"Model Prediction:\n{prediction}\n")
        print("----------------------------\n")
    return {"prediction_text": final_answers, "id": batch["id"]}

def generate_choice(batch):
    batch_input = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
    input_texts = [build_input_prompt(e) for e in batch_input]
    predicted_answers = []
    if benchmark == "arc-easy" or benchmark == "arc-challenge" or benchmark == "openbookqa":
        for i, input_text in enumerate(input_texts):
            choices = batch_input[i]["choices"]["text"]
            choice_labels = batch_input[i]["choices"]["label"]
            log_likelihoods = [compute_log_likelihood(input_text, choice) for choice in choices]
            best_choice_idx = np.argmax(log_likelihoods)
            predicted_answers.append(choice_labels[best_choice_idx])
    elif benchmark == "hellaswag":
        for i, input_text in enumerate(input_texts):
            endings = batch_input[i]["endings"]
            log_likelihoods = [compute_log_likelihood(input_text, ending) for ending in endings]
            best_ending_idx = np.argmax(log_likelihoods)
            predicted_answers.append(best_ending_idx)
    elif benchmark == "piqa":
        for i, input_text in enumerate(input_texts):
            choices = [batch_input[i]["sol1"], batch_input[i]["sol2"]]
            log_likelihoods = [compute_log_likelihood(input_text, choice) for choice in choices]
            best_choice_idx = np.argmax(log_likelihoods)
            predicted_answers.append(best_choice_idx)
    elif benchmark == "winogrande":
        for i, input_text in enumerate(input_texts):
            choices = [batch_input[i]["option1"], batch_input[i]["option2"]]
            log_likelihoods = [compute_log_likelihood(input_text, choice) for choice in choices]
            best_choice_idx = np.argmax(log_likelihoods)
            predicted_answers.append(best_choice_idx)
    elif benchmark == "copa":
        for i, input_text in enumerate(input_texts):
            choices = [batch_input[i]["choice1"], batch_input[i]["choice2"]]
            log_likelihoods = [compute_log_likelihood(input_text, choice) for choice in choices]
            best_choice_idx = np.argmax(log_likelihoods)
            predicted_answers.append(best_choice_idx)
    elif benchmark == "boolq" or benchmark == "wic" or benchmark == "wsc":
        for i, input_text in enumerate(input_texts):
            choices = ["No", "Yes"]
            log_likelihoods = [compute_log_likelihood(input_text, choice) for choice in choices]
            best_choice_idx = np.argmax(log_likelihoods)
            predicted_answers.append(best_choice_idx)
    elif benchmark == "rte":
        for i, input_text in enumerate(input_texts):
            choices = ["Yes", "No"]
            log_likelihoods = [compute_log_likelihood(input_text, choice) for choice in choices]
            best_choice_idx = np.argmax(log_likelihoods)
            predicted_answers.append(best_choice_idx)
    elif benchmark == "cb":
        for i, input_text in enumerate(input_texts):
            choices = ["Entailment", "Contradiction", "Neutral"]
            log_likelihoods = [compute_log_likelihood(input_text, choice) for choice in choices]
            best_choice_idx = np.argmax(log_likelihoods)
            predicted_answers.append(best_choice_idx)
    else:
        raise ValueError("Invalid benchmark.")
    for input_prompt, prediction in zip(input_texts, predicted_answers):
        print("\n----------------------------\n")
        print(f"{input_prompt}\n")
        print(f"Model Prediction:\n{prediction}\n")
        print("----------------------------\n")
    return {"prediction_text": predicted_answers}

eval_dataset = filtered_dataset

if benchmark == "squad":
    predictions = eval_dataset.map(generate_full_answer, batched=True, batch_size=48, load_from_cache_file=False)
    assert len(predictions) == len(eval_dataset), "Error: Number of predictions should match number of examples in the dataset!"
    formatted_predictions = [{"id": predictions[i]["id"], "prediction_text": predictions[i]["prediction_text"]} for i in range(len(predictions))]
    references = [{"id": eval_dataset[i]["id"], "answers": eval_dataset[i]["answers"]} for i in range(len(eval_dataset))]
    assert all(predictions[i]["id"] == references[i]["id"] for i in range(len(predictions))), "Error: Mismatch in IDs!"
    metric = evaluate.load("squad")
    metrics = metric.compute(predictions=formatted_predictions, references=references)
    print(f" F1 Score: {metrics['f1']:.2f}")
elif benchmark in ["arc-easy", "arc-challenge", "openbookqa"]:
    predictions = eval_dataset.map(generate_choice, batched=True, batch_size=48, load_from_cache_file=False)
    assert len(predictions) == len(eval_dataset), "Error: Number of predictions should match number of examples in the dataset!"
    formatted_predictions = [predictions[i]["prediction_text"] for i in range(len(predictions))]
    formatted_predictions = [ord(i) - 65 if i.isalpha() else i for i in formatted_predictions]
    references = [eval_dataset[i]["answerKey"] for i in range(len(eval_dataset))]
    references = [ord(i) - 65 if i.isalpha() else i for i in references]
    accuracy_metric = evaluate.load("accuracy")
    metrics = accuracy_metric.compute(
        predictions=formatted_predictions,
        references=references
    )
    print(f"\n Accuracy: {metrics['accuracy']:.2f}")
elif benchmark in ["hellaswag", "piqa", "winogrande", "copa", "boolq", "cb", "wic", "rte", "wsc"]:
    predictions = eval_dataset.map(generate_choice, batched=True, batch_size=48, load_from_cache_file=False)
    assert len(predictions) == len(eval_dataset), "Error: Number of predictions should match number of examples in the dataset!"
    formatted_predictions = [predictions[i]["prediction_text"] for i in range(len(predictions))]
    if benchmark in ["hellaswag", "piqa", "copa", "boolq", "cb", "wic", "rte", "wsc"]:
        references = [eval_dataset[i]["label"] for i in range(len(eval_dataset))]
    elif benchmark == "winogrande":
        references = [int(eval_dataset[i]["answer"]) - 1 for i in range(len(eval_dataset))]
    accuracy_metric = evaluate.load("accuracy")
    metrics = accuracy_metric.compute(
        predictions=formatted_predictions,
        references=references
    )
    print(f"\n Accuracy: {metrics['accuracy']:.2f}")
else:
    raise ValueError("Invalid benchmark. Please choose from 'squad', 'arc-easy', 'arc-challenge', 'openbookqa'.")