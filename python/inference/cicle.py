import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from crepes import WrapClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances

class CICLe:
    def __init__(self):
        # load pretrained model:
        with open('cicle.pkl', 'rb') as file:
            base = pkl.load( file)

        # set properties:
        self.phi        = lambda x: base['phi'].transform(x).toarray()
        self.classifier = base['classifier']
        self.id2label   = base['id2label']
        self.label2id   = base['label2id']
        self.texts      = base['texts']
        self.labels     = base['labels']

    def get_few_shot_examples(self, text, prediction_set, examples_per_class=2):
        examples = []

        for y in prediction_set:
            # get texts in current class:
            texts = self.texts[self.labels == y]

            # generate embeddings of texts in class:
            embeddings = self.phi([text] + texts.tolist())

            # calculate cosine-similaity:
            similarity = (1. - pairwise_distances(embeddings, metric='cosine'))[1:,0]

            # get closest sample of training data based on embeddings:
            for j in np.argsort(similarity)[::-1][:examples_per_class]:
                examples.append((texts[j], y, similarity[j]))

        # sort samples based on embedding from training data:
        examples.sort(key=lambda e: e[2], reverse=True)

        return examples

    def create_prompt(self, prompt, text, examples):
        # helper function replacing quotation marks in the text:
        replace_qm = lambda s: s.replace('"', "'").replace('\n\n', "\n").replace('\r\n', "\n")

        # create context:
        context  = [f'\n\n"{replace_qm(x)}" => {y}' for x, y, _ in examples]
        context += [f'\n\n"{replace_qm(text)}" => ']

        return [{"role": "user", "content": prompt + ''.join(context)}]

    def explain_prompts(self, prompt, chats, texts, examples, generate, max_tries):
        # This function is designed to take batches of inputs,
        # so all arguments should be lists of corresponding elements.

        # create prompts:
        prompts = []
        for examples_text, text, chat in zip(examples, texts, chats):
            for e in examples_text:
                if e[1] not in chat[-1]["content"]:
                    prompts.append(chat + [{
                        "role": "user",
                        "content": f"Provide a version of \"{text}\" that would change your assessment to {e[1]}. Change as few words as possible. Do not explain your output."
                    }])
                    break

        counterfactuals = [None] * len(chats)
        for _ in range(max_tries):
            new_chats = []
            indices   = []

            # Only process if a counterfactual hasn't been found yet:
            for i, cf in enumerate(counterfactuals):
                if cf is None:
                    new_chats.append(prompts[i])
                    indices.append(i)

            if len(new_chats) == 0: break

            # prompt for counterfactual:
            new_chats = generate(new_chats, 64, 1.)
            cf_texts  = [chat[-1]["content"].strip('"') for chat in new_chats]

            # prompt for validity:
            cf_preds  = generate([self.create_prompt(prompt, cf_text, examples[i]) for cf_text, i in zip(cf_texts, indices)], 5, None)
            cf_preds = [chat[-1]["content"] for chat in cf_preds]

            for j, i in enumerate(indices):
                if cf_preds[j] != chats[i][-1]["content"]:
                    counterfactuals[i] = cf_texts[j]

        return counterfactuals

    def __call__(self, texts, prompt, generate, batch_size):
        # predicts prediction sets:
        prediction_sets = [self.id2label[s.astype(bool)] for s in self.classifier.predict_set(self.phi(texts))]

        # initialize predictions:
        predictions = [{
            'title': x,
            'prediction': ps[0],
            'counterfactual': None,
            'probable_classes': list(ps)
        } if len(ps) == 1 else None for x, ps in zip(texts, prediction_sets)]
        indices = [i for i, p in enumerate(predictions) if p is None]

        # predict texts in batches:
        for i in tqdm(np.arange(0, len(indices), batch_size), desc='Predicting'):
            batch_indices = indices[i:i+batch_size]
            batch_X = [texts[i] for i in batch_indices]
            batch_ps = [prediction_sets[i] for i in batch_indices]

            # get 2 most similar texts in the training data:
            batch_examples = [self.get_few_shot_examples(x, ps) for x, ps in zip(batch_X, batch_ps)]

            # create prompts:
            batch_chats = [self.create_prompt(prompt, x, examples) for x, examples in zip(batch_X, batch_examples)]

            # predict class:
            batch_chats = generate(batch_chats, 5, None)

            # explain:
            batch_cfs = self.explain_prompts(prompt, batch_chats, batch_X, batch_examples, generate, max_tries=15)

            for j, x, cf, ps, chat in zip(batch_indices, batch_X, batch_cfs, batch_ps, batch_chats):
                predictions[j] = {
                    'title': x,
                    'prediction': chat[-1]["content"],
                    'counterfactual': cf,
                    'probable_classes': list(ps)
                }

        return pd.DataFrame(predictions)