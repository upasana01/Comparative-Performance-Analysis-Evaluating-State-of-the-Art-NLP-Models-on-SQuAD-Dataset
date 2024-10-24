from datasets import load_dataset
import torch
from tqdm.auto import tqdm  
from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering
import evaluate

# Load certain rows of squad dataset
data = load_dataset('squad')

# Function to add the start and end index for answer context pair
def add_end_idx(answers, contexts):
    new_answers = []
    for answer, context in tqdm(zip(answers, contexts)):
        answer['text'] = answer['text'][0]
        answer['answer_start'] = answer['answer_start'][0]
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        else:
            for n in [1, 2]:
                if context[start_idx-n:end_idx-n] == gold_text:
                    answer['answer_start'] = start_idx - n
                    answer['answer_end'] = end_idx - n
        new_answers.append(answer)
    return new_answers

def prep_data(dataset):
    questions = dataset['question']
    contexts = dataset['context']
    answers = add_end_idx(dataset['answers'], contexts)
    return {
        'question': questions,
        'context': contexts,
        'answers': answers
    }

dataset = prep_data(data['train'].shuffle(seed=123).select(range(1000)))

# Tokenization
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
train = tokenizer(dataset['context'], dataset['question'],
                  truncation=True, padding='max_length',
                  max_length=512, return_tensors='pt')

def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in tqdm(range(len(answers))):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        shift = 1
        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - shift)
            shift += 1
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train, dataset['answers'])

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = SquadDataset(train)
dataloader_train = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=2,
                                               shuffle=True)

model = RobertaForQuestionAnswering.from_pretrained('roberta-base')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()

optimizer1 = torch.optim.AdamW(model.parameters(), lr=0.1, eps=0.01)

epochs = 1

for epoch in tqdm(range(1, epochs + 1)):
    model.train()
    loss_train_total = 0
    progress_bar = tqdm(dataloader_train, desc=f'Epoch {epoch}', leave=False, disable=False)
    for batch in progress_bar:
        try:
            model.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'start_positions': start_positions,
                'end_positions': end_positions
            }
            outputs = model(**inputs)
            loss = outputs.loss
            loss_train_total += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer1.step()
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})
        except Exception as e:
            print(f"Error: {e}")
            print("Batch contents:")
            print(batch)
        
    torch.save(model.state_dict(), f'finetuned_finBERT_epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    loss_train_avg = loss_train_total / len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')

model.eval()

def get_prediction(context, question):
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt').to(device)
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs[0])
    answer_end = torch.argmax(outputs[1]) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    return answer, answer_start, answer_end

def normalize_text(s):
    import string, re
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match(prediction, truth):
    return bool(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    common_tokens = set(pred_tokens) & set(truth_tokens)
    if len(common_tokens) == 0:
        return 0
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    return round(2 * (prec * rec) / (prec + rec), 2)

def question_answer(context, question, answer):
    prediction, _, _ = get_prediction(context, question)
    em_score = exact_match(prediction, answer)
    f1_score = compute_f1(prediction, answer)
    print(f'Question: {question}')
    print(f'Prediction: {prediction}')
    print(f'True Answer: {answer}')
    print(f'Exact match: {em_score}')
    print(f'F1 score: {f1_score}\n')

# Ensure the dataset is correctly structured
print(f"Dataset length: {len(dataset['context'])}")
print(f"Sample entry context: {dataset['context'][0]}")
print(f"Sample entry question: {dataset['question'][0]}")
print(f"Sample entry answer: {dataset['answers'][0]}")

# BLEU and ROUGE metric calculation
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

predictions = []
references = []

for idx, context in enumerate(dataset['context']):
    print(f"Processing entry {idx}")
    print(f"Processing Context {context}")
    question = dataset['question'][idx]
    true_answer = dataset['answers'][idx]['text'][0]
    try:
        prediction, _, _ = question_answer(context, question, true_answer)
        predictions.append(prediction)
        references.append([true_answer])
    except Exception as e:
        print(f"Error processing entry {idx}: {e}")

bleu_score = bleu.compute(predictions=predictions, references=references)
rouge_score = rouge.compute(predictions=predictions, references=references)

print(f'BLEU score: {bleu_score}')
print(f'ROUGE score: {rouge_score}')
