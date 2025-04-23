import os
import tqdm
import random
import torch
import numpy as np
import itertools
import argparse
import transformers

from collections import defaultdict
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from utility import bucketize, compute_doc_score


transformers.logging.set_verbosity_error()


# Full Dataset Class - Version 2.0
class MentionPairDataset(Dataset):
    def __init__(self, hf_dataset_split, tokenizer, score = False):
        '''Initialize Entire data s.t. 
        during training, fetching data is just a simple lookup'''

        self.tokenizer = tokenizer
        self.score = score # Bool
        self.max_length = self.tokenizer.model_max_length    # 512 for BERT-base-uncased
        self.context_window = 122                            # window around each mention

        self.examples = []

        if score:
            print (f'Creating Dynamic Bucket Bins....')
            self.bins_s1, self.bins_s2, self.bins_s3 = self.create_bucket_bins(hf_dataset_split)

        progress = tqdm.tqdm(hf_dataset_split, desc = 'Processing dataset')
        for doc_idx, doc in enumerate(progress):
            words, spans = self.flatten_document(doc)   # spans: [(cluster_id, start, end)]
            
            positive_pairs = []
            negative_pairs = []

            cluster_map = defaultdict(list)

            # Group mentions by cluster ID
            for cid, s, e in spans:
                cluster_map[cid].append((s,e))

            # Populate positive_pairs
            for mentions in cluster_map.values():
                if len(mentions) >= 2:
                    # sample up to max_pairs_per_cluster from combinations
                    all_pairs = list(itertools.combinations(mentions, 2))
                    sampled = random.sample(all_pairs, min(len(all_pairs), 3))    # Pick 3 or fewer postitve pairs per cluster.
                    for m1, m2 in sampled:
                        positive_pairs.append((m1, m2, 1.0))

            # Populate negative_pairs
            for i in range(len(spans)):
                for j in range(i+1, len(spans)):
                    (cid1, s1, e1) = spans[i]
                    (cid2, s2, e2) = spans[j]

                    if cid1 != cid2:
                        negative_pairs.append(((s1, e1), (s2, e2), 0.0))
                    
            #print (f'[INFO] {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs found')

            # Downsample negatives
            sampled_negatives = random.sample(negative_pairs, min(len(positive_pairs), len(negative_pairs)))
            
            # comobine and shuffle
            final_pairs = positive_pairs + sampled_negatives
            random.shuffle(final_pairs)

            #print ('Length of final_pairs: ', len(final_pairs))
            
            for i, (m1, m2, label_val) in enumerate(final_pairs):
                if not self.score:
                    text = self.create_truncated_text(words, m1, m2)
                else:
                    doc_scores = compute_doc_score(doc)
                    text = self.create_truncated_text(words, m1, m2, doc_scores)
                #print (f'{count}/{len(final_pairs)} pairs processed')
                label = torch.tensor(label_val)
                self.examples.append((text, label))

                # Update progress bar
                progress.set_postfix(doc = doc_idx+1, pair = f'{i+1}/{len(final_pairs)}')
                
            '''if doc_idx + 1 == 3:  # Test with 3 documents.
                progress.update(1)
                progress.close()
                print ('Done!')
                break'''
            
        progress.close()
        print ('Done!')


    def __len__(self):
        return len(self.examples)


    def __getitem__(self, idx):
        return self.examples[idx]
    

    def create_bucket_bins(self, dataset):
        '''
            Dynamic binning based on actual dataset distribution
        '''
        score1_list = []
        score2_list = []
        score3_list = []

        for doc in dataset:
            s1, s2, s3 = compute_doc_score(doc)
            score1_list.append(s1)
            score2_list.append(s2)
            score3_list.append(s3)

        bins_s1 = np.quantile(score1_list, [0.25, 0.5, 0.75])
        bins_s2 = np.quantile(score2_list, [0.25, 0.5, 0.75])
        bins_s3 = np.quantile(score3_list, [0.25, 0.5, 0.75])

        return bins_s1, bins_s2, bins_s3


    def flatten_document(self, document):
        words, spans = [],[]

        for sent in document['sentences']:
            offset = len(words)
            words.extend(sent['words'])

            for span in sent['coref_spans']:
                cluster_id, start, end = span
                spans.append((cluster_id, offset + start, offset + end))  # inclusive span

        return words, spans    # full text, all coref spans
    
    
    def create_truncated_text(self, words, m1_span, m2_span, doc_scores = None):
        """
        Truncate around mentions and ensure the final tokenized output includes both mentions
        and fits within the model's max_length.
        """

        if self.score == False:
            extra_token_num = 2 # [CLS] and [SEP]
        else:
            extra_token_num = 5 # [CLS], [SEP] and [S1] [S2] [S3] = doc_score

        m1_start, m1_end = m1_span
        m2_start, m2_end = m2_span

        if m2_start < m1_start:
            (m1_start, m1_end), (m2_start, m2_end) = (m2_start, m2_end), (m1_start, m1_end)

        left_context_start = max(0, m1_start - self.context_window)
        left_context = words[left_context_start:m1_start]

        mention1 = words[m1_start:m1_end + 1]
        between_mentions = words[m1_end + 1:m2_start]
        mention2 = words[m2_start:m2_end + 1]

        right_context_end = min(len(words), m2_end + 1 + self.context_window)
        right_context = words[m2_end + 1:right_context_end]

        # Pre-tokenize all components
        get_tokens = lambda seq: self.tokenizer(seq, is_split_into_words=True, add_special_tokens=False)["input_ids"]

        left_tokens = get_tokens(left_context)
        right_tokens = get_tokens(right_context)
        mention1_tokens = get_tokens(mention1)
        mention2_tokens = get_tokens(mention2)
        between_tokens = get_tokens(between_mentions)

        # Add 4 for <m> and </m> tokens, which were added to tokenizer
        m_token_id = self.tokenizer.convert_tokens_to_ids("<m>")
        m_end_token_id = self.tokenizer.convert_tokens_to_ids("</m>")

        # Pre-compute mention-wrapped sequences
        mention1_tokens = [m_token_id] + mention1_tokens + [m_end_token_id]
        mention2_tokens = [m_token_id] + mention2_tokens + [m_end_token_id]

        while True:
            total_len = (
                len(left_tokens) + len(mention1_tokens) + 
                len(between_tokens) + len(mention2_tokens) +
                len(right_tokens) + extra_token_num + 4  # 4 for two <m> and two </m>
            )
            if total_len <= self.max_length:
                break

            # Trim between_mentions -> left context -> right context (token level now)
            if len(between_tokens) > 2:
                between_tokens = between_tokens[1:-1]
            elif len(left_tokens) > 0:
                left_tokens = left_tokens[1:]
            elif len(right_tokens) > 0:
                right_tokens = right_tokens[:-1]
            else:
                break

        #final_token_ids = (left_tokens + mention1_tokens + between_tokens + mention2_tokens + right_tokens)

        # Decode back to text - 
        #final_text = self.tokenizer.decode(final_token_ids, skip_special_tokens = True)
        #print (final_text)
        mention1_text = self.tokenizer.convert_ids_to_tokens(mention1_tokens[1:-1])
        mention2_text = self.tokenizer.convert_ids_to_tokens(mention2_tokens[1:-1])

        base_tokens = (
            self.tokenizer.convert_ids_to_tokens(left_tokens) +
            ["<m>"] + mention1_text + ["</m>"] +
            self.tokenizer.convert_ids_to_tokens(between_tokens) +
            ["<m>"] + mention2_text + ["</m>"] +
            self.tokenizer.convert_ids_to_tokens(right_tokens)
        )

        if not self.score:
            return " ".join(base_tokens)
        
        else:
            #assert doc_scores is not None, "doc_score must be provided when score = True"
            score1, score2, score3 = doc_scores
            s1_token = bucketize(score1, prefix = "S1", bins = self.bins_s1)
            s2_token = bucketize(score2, prefix = "S2", bins = self.bins_s2)
            s3_token = bucketize(score3, prefix = "S3", bins = self.bins_s3)

            return " ".join([s1_token, s2_token, s3_token] + base_tokens)


# -----------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=["train", "validation", "test"], default="train")
    parser.add_argument("--score", action="store_true", help="Whether to include doc-level score tokens")
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(['<m>', '</m>'], special_tokens=True)    # special tokens that encapsulate mentions

    # Add score bucket tokens (optinal)
    if args.score:
        score_tokens = [
            "<S1=low>", "<S1=mid>", "<S1=high>", "<S1=veryhigh>",
            "<S2=low>", "<S2=mid>", "<S2=high>", "<S2=veryhigh>",
            "<S3=low>", "<S3=mid>", "<S3=high>", "<S3=veryhigh>"]
        tokenizer.add_tokens(score_tokens, special_tokens=True)

    
    dataset = load_dataset("conll2012_ontonotesv5", "english_v4")[args.split] # Train/Validation/Test
    processed = MentionPairDataset(dataset, tokenizer, score=args.score)

    # Save examples
    if args.score:
        torch.save(processed, os.path.join(args.output_dir, f"mention_pairs_{args.split}_scored.pt"))
    else:
        torch.save(processed, os.path.join(args.output_dir, f"mention_pairs_{args.split}.pt"))

    # Save tokenizer
    if args.split == "train":
        if args.score:
            tokenizer.save_pretrained(os.path.join(args.output_dir, f"tokenizer_{args.split}_scored"))
        else:
            tokenizer.save_pretrained(os.path.join(args.output_dir, f"tokenizer_{args.split}"))


'''
During training/inference....To Load the saved tokenizer:
tokenizer = AutoTokenizer.from_pretrained("data/tokenizer_train_scored")
OR
tokenizer = AutoTokenizer.from_pretrained("data/tokenizer_train")

Then;
model = AutoModel.from_pretrained("bert-base-uncased")
mode.resize_token_embeddings(len(tokenizer))
'''