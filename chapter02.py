import re
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])

#An example on how to tokenize the words
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
print(result)
result = re.split(r'([,.]|\s)', text)
print(result)
result = [item for item in result if item.strip()]
print(result)
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

#on Edith Whorton Paragraph
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])

#Building vocabulary
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

#Converts text into tokens
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break   

#A class for vocaubulary storing
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab            #1
        self.int_to_str = {i:s for s,i in vocab.items()}        #2

    def encode(self, text):         #3
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):         #4
        text = " ".join([self.int_to_str[i] for i in ids]) 

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)    #5
        return text
    
     
    
#1 Stores the vocabulary as a class attribute for access in the encode and decode methods
#2 Creates an inverse vocabulary that maps token IDs back to the original text tokens
#3 Processes input text into token IDs
#4 Converts token IDs back into text
#5 Removes spaces before the specified punctuation

# Adding tokens for words which are unknown and are not in the vocabulary 
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}

# As an additional quick check, let’s print the last five entries of the updated vocabulary:
print(len(vocab.items()))
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# This will convert the text into token ID
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids) 
   
   
# Listing 2.4 A simple text tokenizer that handles unknown words 
    
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int            #1
                        else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)    #2
        return text


#    Listing 2.4 A simple text tokenizer that handles unknown words

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

#  let’s tokenize the sample text using the SimpleTokenizerV2 on the vocab we previously created in listing 2.2
tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))

#  detokenize the text for a quick sanity check:
print(tokenizer.decode(tokenizer.encode(text)))


#  more sophisticated tokenization scheme based on a concept called byte pair encoding (BPE). The BPE tokenizer was used to train LLMs such as GPT-2, GPT-3, and the original model used in ChatGPT.
from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

# The usage of this tokenizer is similar to the SimpleTokenizerV2 we implemented previously via an encode method:
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

 
strings = tokenizer.decode(integers)
print(strings)

text_abd=("This is a Akwirwwwe")

ind_tok_id = tokenizer.encode(text_abd, allowed_special={"<|endoftext|>"})
print(ind_tok_id)
dec_ind_tok_id = tokenizer.decode(ind_tok_id)
print(dec_ind_tok_id)

#Let’s implement a data loader that fetches the input–target pairs in figure 2.12 from the training dataset using a sliding window approach. To get started, we will tokenize the whole “The Verdict” short story using the BPE tokenizer:
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

#  we remove the first 50 tokens from the dataset for demonstration purposes, as it results in a slightly more interesting text passage in the next steps:
enc_sample = enc_text[50:]

# One of the easiest and most intuitive ways to create the input–target pairs for the next-word prediction task is to create two variables, x and y, where x contains the input tokens and y contains the targets, which are the inputs shifted by 1:
context_size = 4         #1
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")

# By processing the inputs along with the targets, which are the inputs shifted by one position, we can create the next-word prediction tasks (see figure 2.12), as follows:

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)
    
#  We’ve now created the input–target pairs that we can use for LLM training.
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired= enc_sample[i]
    print(tokenizer.decode(context), "----->", tokenizer.decode([desired]))

     
   
   
#    The code for the dataset class is shown in the following listing. 
   
   
#    Listing 2.5 A dataset for batched inputs and targets 
    
import torch
from torch.utils.data import Dataset, DataLoader
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)    #1

        for i in range(0, len(token_ids) - max_length, stride):     #2
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):    #3
        return len(self.input_ids)

    def __getitem__(self, idx):         #4
        return self.input_ids[idx], self.target_ids[idx]

     #1 Tokenizes the entire text
     #2 Uses a sliding window to chunk the book into overlapping sequences of max_length
     #3 Returns the total number of rows in the dataset
     #4 Returns a single row from the dataset

 
#Listing 2.6 A data loader to generate batches with input-with pairs
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")                         #1
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)   #2
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,     #3
        num_workers=num_workers     #4
    )

    return dataloader

 
   
# test the dataloader with a batch size of 1 for an LLM with a context size of 4 to develop an intuition of how the GPTDatasetV1 class from listing 2.5 and the create_ dataloader_v1 function from listing 2.6 work together:   
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)      #1
first_batch = next(data_iter)
print(first_batch)
#1 Converts dataloader into a Python iterator to fetch the next entry via Python’s built-in next() function

second_batch = next(data_iter)
print(second_batch)

dataloader = create_dataloader_v1(
    raw_text, batch_size=2, max_length=8, stride=5, shuffle=False)
data_iter = iter(dataloader)      #1
first_batch = next(data_iter)
print(first_batch)


 
   
   
# Let’s look briefly at how we can use the data loader to sample with a batch size greater than 1: 
   
   
    
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4,
    shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# Let’s see how the token ID to embedding vector conversion works with a hands-on example. Suppose we have the following four input tokens with IDs 2, 3, 5, and 1: 
input_ids = torch.tensor([2, 3, 5, 1])

#    For the sake of simplicity, suppose we have a small vocabulary of only 6 words (instead of the 50,257 words in the BPE tokenizer vocabulary), and we want to create embeddings of size 3 (in GPT-3, the embedding size is 12,288 dimensions): 
vocab_size = 6
output_dim = 3

# Using the vocab_size and output_dim, we can instantiate an embedding layer in PyTorch, setting the random seed to 123 for reproducibility purposes: 
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

 
   
   
# Now, let’s apply it to a token ID to obtain the embedding vector:  
print(embedding_layer(torch.tensor([3])))  
# We’ve seen how to convert a single token ID into a three-dimensional embedding vector. Let’s now apply that to all four input IDs (torch.tensor([2, 3, 5, 1])): 
print(embedding_layer(input_ids))

 
   
#    let’s create the initial positional embeddings to create the LLM inputs. 
#    Previously, we focused on very small embedding sizes for simplicity. Now, let’s consider more realistic and useful embedding sizes and encode the input tokens into a 256-dimensional vector representation, which is smaller than what the original GPT-3 model used (in GPT-3, the embedding size is 12,288 dimensions) but still reasonable for experimentation. Furthermore, we assume that the token IDs were created by the BPE tokenizer we implemented earlier, which has a vocabulary size of 50,257:     
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# Using the previous token_embedding_layer, if we sample data from the data loader, we embed each token in each batch into a 256-dimensional vector. If we have a batch size of 8 with four tokens each, the result will be an 8 × 4 × 256 tensor. 
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
   stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

# Let’s now use the embedding layer to embed these token IDs into 256-dimensional vectors: 
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

 
# we just need to create another embedding layer that has the same embedding dimension as the token_embedding_ layer:
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

# As we can see, the positional embedding tensor consists of four 256-dimensional vectors. We can now add these directly to the token embeddings, where PyTorch will add the 4 × 256–dimensional pos_embeddings tensor to each 4 × 256–dimensional token embedding tensor in each of the eight batches: 
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)

# As part of the input processing pipeline, input text is first broken up into individual tokens. These tokens are then converted into token IDs using a vocabulary. The token IDs are converted into embedding vectors to which positional embeddings of a similar size are added, resulting in input embeddings that are used as input for the main LLM layers.
