import fitz
from tqdm.auto import tqdm
from spacy.lang.en import English
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device_Type: {device}')
import gc
import textwrap
from sentence_transformers import util
from time import perf_counter as timer


class rag:
  def __init__(self, model_path, quantize = False):
    self.model_path = model_path
    self.quantize = quantize
    self.model, self.tokenizer = self.initialize_model(self.model_path,self.quantize)

  @staticmethod
  def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip()
    # Other potential text formatting functions can go here
    return cleaned_text

  def open_and_read_pdf(self,pdf_path:str) -> list[dict]:
    doc = fitz.open(pdf_path)  # open a document
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):  # iterate the document pages
        text = page.get_text()  # get plain text encoded as UTF-8
        text = self.text_formatter(text)
        pages_and_texts.append({"page_number": page_number - 0,
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "page_token_count": len(text) / 4,  # 1 token
                                "text": text})
    return pages_and_texts

  @staticmethod
  def sentence_converter(pages_and_texts: list[dict]) -> list[dict]:
    nlp = English()
    nlp.add_pipe("sentencizer")
    for item in tqdm(pages_and_texts):
        item["sentences"] = list(nlp(item["text"]).sents)
        # Make sure all sentences are strings
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]
        # Count the sentences
        item["page_sentence_count_spacy"] = len(item["sentences"])

    return pages_and_texts

  @staticmethod
  def split_list(input_list: list,
            slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of size slice_size (or as close as possible).

    For example, a list of 17 sentences would be split into two lists of [[10], [7]]
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]


  def chunking(self,pages_and_texts_sen: list[dict], num_sentence_chunk_size = 10) -> list[dict]:
  # Loop through pages and texts and split sentences into chunks
    for item in tqdm(pages_and_texts_sen):
        item["sentence_chunks"] = self.split_list(input_list=item["sentences"],
                                            slice_size=num_sentence_chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])

    return pages_and_texts_sen

  @staticmethod
  def convert_chunks(pages_and_texts_sen_chunk: list[dict]) -> list():
    # Split each chunk into its own item
    pages_and_chunks = []
    for item in tqdm(pages_and_texts_sen_chunk):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]

            # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo
            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            # Get stats about the chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters

            pages_and_chunks.append(chunk_dict)
    return pages_and_chunks

  def pages_to_chunks(self,pages_and_texts):
    pages_and_texts_sen = self.sentence_converter(pages_and_texts)
    pages_and_texts_sen_chunk = self.chunking(pages_and_texts_sen)
    pages_and_chunks = self.convert_chunks(pages_and_texts_sen_chunk)
    return pages_and_chunks


  @staticmethod
  def initialize_model(model_path, quantize):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if quantize == True:
      from transformers import BitsAndBytesConfig
      bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

      model = AutoModelForCausalLM.from_pretrained(
          model_path,
          quantization_config=bnb_config,
          device_map=device
      )
    else:
      model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16).to(device)

    return model, tokenizer

  def get_model_num_params(self):
      model = self.model
      return sum([param.numel() for param in model.parameters()])

  def get_model_mem_size(self):
    # Get model parameters and buffer sizes
    model = self.model
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])

    # Calculate various model sizes
    model_mem_bytes = mem_params + mem_buffers # in bytes
    model_mem_mb = model_mem_bytes / (1024**2) # in megabytes
    model_mem_gb = model_mem_bytes / (1024**3) # in gigabytes

    return {"model_mem_bytes": model_mem_bytes,
            "model_mem_mb": round(model_mem_mb, 2),
            "model_mem_gb": round(model_mem_gb, 2)}


  @staticmethod
  def generate_embedding(tokenizer, model: torch.nn.Module, text:str):
    inputs = tokenizer(text, return_tensors="pt",padding=True ,truncation=True).to(device)

    # Forward pass through the model to get hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract the hidden states
    hidden_states = outputs.hidden_states  # This is a tuple with the hidden states from all layers

    # Typically, the last hidden state is used as the embedding
    # hidden_states[-1] has the shape [batch_size, sequence_length, hidden_size]
    embedding = hidden_states[-1][:, 0, :]

    return embedding


  def create_embeddings(self,pages_and_chunks: list):
    # Create embeddings one by one on the GPU
    tokenizer = self.tokenizer
    model = self.model
    for item in tqdm(pages_and_chunks):
        item["embedding"] = self.generate_embedding(tokenizer, model, item["sentence_chunk"])
    return pages_and_chunks


  def process_pdf(self, pdf_path:str):
    pages_and_texts = self.open_and_read_pdf(pdf_path)
    pages_and_chunks = self.pages_to_chunks(pages_and_texts)
    return pages_and_chunks

  def embedding_merge(self,pages_and_chunks: list):
    # Merge all embeddings into one tensor
    embeddings = list()
    for i in pages_and_chunks:
      embeddings.append(i['embedding'])
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


  @staticmethod
  def print_wrapped(text, wrap_length=80):
      wrapped_text = textwrap.fill(text, wrap_length)
      print(wrapped_text)

  def retrieve_relevant_resources(self,
                                query: str,
                                embeddings: torch.tensor,
                                n_resources_to_return: int=5,
                                print_time: bool=True):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """
    tokenizer = self.tokenizer
    model = self.model
    # Embed the query
    query_embedding = self.generate_embedding(tokenizer, model, query)

    # Get dot product scores on embeddings
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time = timer()

    if print_time:
        print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

    scores, indices = torch.topk(input=dot_scores,
                                k=n_resources_to_return)

    return scores, indices

  def print_top_results_and_scores(self ,query: str,
                                  embeddings: torch.tensor,
                                  pages_and_chunks: list[dict],
                                  n_resources_to_return: int=5):
      """
      Takes a query, retrieves most relevant resources and prints them out in descending order.

      Note: Requires pages_and_chunks to be formatted in a specific way (see above for reference).
      """

      scores, indices = self.retrieve_relevant_resources(query=query,
                                                    embeddings=embeddings,
                                                    n_resources_to_return=n_resources_to_return)

      print(f"Query: {query}\n")
      print("Results:")
      # Loop through zipped together scores and indicies
      for score, index in zip(scores, indices):
          print(f"Score: {score:.4f}")
          # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
          self.print_wrapped(pages_and_chunks[index]["sentence_chunk"])
          # Print the page number too so we can reference the textbook further and check the results
          print(f"Page number: {pages_and_chunks[index]['page_number']}")
          print("\n")

  def prompt_formatter(self, query: str,
                      context_items: list[dict]) -> str:
    """
    Augments query with text-based context from context_items.
    """
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    # Create a base prompt with examples to help the model
    base_prompt = """Based on the following context items, please answer the query.
    Give yourself room to think by extracting relevant passages from the context before answering the query.
    Don't return the thinking, only return the answer.
    Make sure your answers are as explanatory as possible.

    \nNow use the following context items to answer the user query:
    {context}
    \nRelevant passages: 
    User query: {query}
    Answer:"""

    # Update base prompt with context items and query
    base_prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]

    # Apply the chat template
    prompt = self.tokenizer.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    return prompt
    
  def ask(self,query, embeddings, pages_and_chunks,
          temperature=0.7,
          max_new_tokens=1024,
          format_answer_text=True,
          return_answer_only=False):
      """
      Takes a query, finds relevant resources/context and generates an answer to the query based on the relevant resources.
      """

      # Get just the scores and indices of top related results
      scores, indices = self.retrieve_relevant_resources(query=query,
                                                    embeddings=embeddings)

      # Create a list of context items
      context_items = [pages_and_chunks[i] for i in indices]

      # Add score to context item
      for i, item in enumerate(context_items):
          item["score"] = scores[i].cpu() # return score back to CPU

      # Format the prompt with context items
      prompt = self.prompt_formatter(query=query,
                                context_items=context_items)

      # Tokenize the prompt
      input_ids = self.tokenizer(prompt, return_tensors="pt").to("cuda")

      # Generate an output of tokens
      outputs = self.model.generate(**input_ids,
                                  temperature=temperature,
                                  do_sample=True,
                                  max_new_tokens=max_new_tokens)

      # Turn the output tokens into text
      output_text = self.tokenizer.decode(outputs[0])

      if format_answer_text:
          # Replace special tokens and unnecessary help message
          output_text = output_text.replace(prompt, "").replace("", "").replace("", "").replace("Sure, here is the answer to the user query:\n\n", "")

      # Only return the answer without the context items
      if return_answer_only:
          return output_text

      return output_text, context_items