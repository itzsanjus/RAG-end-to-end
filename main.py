from rag_inference import rag
c = rag('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
pdf_path = '/content/kebo109.pdf'
processed_text = c.process_pdf(pdf_path)
processed_text_embed = c.create_embeddings(processed_text)
embeddings = c.embedding_merge(processed_text_embed)
query = 'What is Structure of Protiens?'
output_text, context_items = c.ask(query, embeddings, processed_text)