import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        seen_words=set()

        for sentence in positive:
            for word in sentence.split():
                seen_words.add(word)
        for sentence in negative:
            for word in sentence.split():
                seen_words.add(word)
        vocabulary=sorted(seen_words)

        word_to_id={
            word:idx+1
            for idx,word in enumerate(vocabulary)
        }
        combined=positive+negative
        encoded=[]
        for sentence in combined:
            encoding=[word_to_id[word] for word in sentence.split()]
            encoded.append(torch.tensor(encoding))

        return nn.utils.rnn.pad_sequence(encoded,batch_first=True)

    
    



        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        
