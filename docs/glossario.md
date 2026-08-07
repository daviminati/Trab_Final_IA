# Glossário Técnico - Projeto Integrador LLM

## Capítulo 1: Understanding Large Language Models

* **Large Language Model (LLM):** Modelo de aprendizado profundo treinado em grandes volumes de texto para compreender, gerar e manipular linguagem natural através da previsão autorregressiva de tokens.
* **IA vs. Machine Learning vs. Deep Learning:**
  * **IA (Inteligência Artificial):** Campo genérico voltado à criação de sistemas que simulam capacidades cognitivas humanas.
  * **ML (Machine Learning):** Subcampo da IA com algoritmos que aprendem padrões a partir de dados sem serem explicitamente programados.
  * **DL (Deep Learning):** Subcampo do ML fundamentado em redes neurais profundas (com múltiplas camadas) capazes de extrair representações hierárquicas de dados não estruturados.
* **Transformer:** Arquitetura de rede neural baseada puramente em mecanismos de atenção (*self-attention*), permitindo processamento em paralelo e eliminando a dependência de estruturas recorrentes (RNNs).
* **GPT (Generative Pre-trained Transformer):** Família de modelos baseada exclusivamente no *Decoder* da arquitetura Transformer, otimizada para a geração autorregressiva de texto.
* **Etapas de Treinamento de um LLM:** 
  1. *Pré-treinamento:* Aprende padrões e estruturas da linguagem em terabytes de dados não rotulados.
  2. *Fine-tuning de Instruções:* Ajusta o modelo para responder comandos e perguntas no formato de assistente.
  3. *Alinhamento (RLHF/DPO):* Refina as respostas para garantir segurança, utilidade e conformidade com preferências humanas.
* **Geração Autorregressiva (Autoregressive Generation):** Processo em que o modelo gera texto um token por vez, alimentando a própria saída de volta na entrada para prever o próximo token.

---

## Apêndice: Fundamentos do PyTorch

* **Tensor:** Estrutura de dados multidimensional fundamental no PyTorch, similar ao `ndarray` do NumPy, mas otimizada para aceleração via GPU/TPU e suporte a diferenciação automática.
* **Autograd:** Mecanismo de diferenciação automática do PyTorch que gera um grafo computacional dinâmico para calcular gradientes (`.backward()`) de forma eficiente.
* **`torch.nn.Module`:** Classe base para a construção de todas as camadas e arquiteturas de redes neurais no PyTorch.
* **Reshaping & Permute (`view`, `reshape`, `transpose`, `permute`):** Operações utilizadas para alterar a forma geométrica de tensores sem alterar seus dados subjacentes, essenciais para manipular dimensões de *batch* e *embeddings*.
