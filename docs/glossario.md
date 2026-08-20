# Glossário Técnico - Projeto Integrador LLM

## Capítulo 1: Compreendendo Grandes Modelos de Linguagem

* **Modelo de Linguagem Grande (LLM):** Modelo de aprendizado profundo treinado em grandes volumes de texto para compreender, gerar e manipular linguagem natural através da previsão autorregressiva de tokens.
* **IA vs. Aprendizado de Máquina vs. Aprendizado Profundo:**
  * **IA (Inteligência Artificial):** Campo genérico voltado à criação de sistemas que simulam capacidades cognitivas humanas.
  * **ML (Aprendizado de Máquina):** Subcampo da IA com algoritmos que aprendem padrões a partir de dados sem serem explicitamente programados.
  * **DL (Deep Learning):** Subcampo do ML fundamentado em redes neurais profundas (com múltiplas camadas) capazes de extrair representações hierárquicas de dados não estruturados.
* **Transformador:** Arquitetura de rede neural baseada puramente em mecanismos de atenção (*self-attention*), permitindo processamento em paralelo e eliminando a dependência de estruturas recorrentes (RNNs).
* **GPT (Transformador Pré-treinado Generativo):** Família de modelos baseada exclusivamente no Decoder da arquitetura Transformer, otimizada para a geração autorregressiva de texto.
* **Etapas de Treinamento de um LLM:**
  * **Pré-treinamento:** Aprende padrões e estruturas da linguagem em terabytes de dados não rotulados.
  * **Ajuste fino de Instruções:** Ajusta o modelo para responder comandos e perguntas no formato de assistente.
  * **Alinhamento (RLHF/DPO):** Refina as respostas para garantir segurança, utilidade e conformidade com preferências humanas.
* **Geração Autorregressiva (Geração Autoregressiva):** Processo em que o modelo gera texto um token por vez, alimentando a própria saída de volta na entrada para prever o próximo token.

---

### Apêndice: Fundamentos do PyTorch

* **Tensor:** Estrutura de dados multidimensional fundamental no PyTorch, similar ao `ndarray` do NumPy, mas otimizada para aceleração via GPU/TPU e suporte a diferenciação automática.
* **Autograd:** Mecanismo de diferenciação automática do PyTorch que gera um grafo computacional dinâmico para calcular gradientes (`.backward()`) de forma eficiente.
* **torch.nn.Module:** Classe base para a construção de todas as camadas e arquiteturas de redes neurais no PyTorch.
* **Remodelação e Permutação (view, reshape, transpose, permute):** Operações utilizadas para alterar a forma geométrica de tensores sem alterar seus dados subjacentes, essenciais para manipular dimensões de batch e embeddings.

---

## Capítulo 2: Trabalhando com Dados de Texto (Working with Text Data)

* **Token:** Unidade básica de texto gerada a partir da segmentação do texto bruto, podendo representar uma palavra completa, uma subpalavra ou um caractere isolado.
* **Token ID:** Valor inteiro, único e discreto associado a um token específico presente no vocabulário do modelo, utilizado como índice para mapear strings numéricas para o computador.
* **Vocabulário (Vocabulary):** Dicionário estruturado que delimita e estabelece o conjunto completo de mapeamentos únicos entre tokens legíveis por humanos e seus respectivos Token IDs.
* **Codificação por Pares de Bytes (Byte Pair Encoding - BPE):** Algoritmo de tokenização baseado em subpalavras que agrupa iterativamente os pares de caracteres/bytes mais frequentes, lidando com termos fora do vocabulário de forma otimizada.
* **Tokens Especiais (Special Tokens):** Tokens reservados com funções estruturais e de controle (ex: `<|unk|>` para caracteres desconhecidos e `<|endoftext|>` para marcar o encerramento de sequências).
* **Janela Deslizante (Sliding Window):** Técnica de amostragem de dados que desloca um bloco de tamanho fixo sobre um texto para extrair os pares de entrada ($X$) e alvo ($Y$, deslocado de uma posição) no treinamento auto-supervisionado.
* **Embedding de Token (Token Embedding):** Camada de mapeamento que converte um Token ID discreto em um vetor denso e contínuo de dimensão fixa ($d_{emb}$), capaz de capturar relações semânticas.
* **Embedding Positional (Positional Embedding):** Vetor denso somado ao embedding do token para codificar a ordem ou posição ordinal do token na sequência, compensando a invariância posicional da atenção.
* **Tamanho do Contexto (Context Length):** Quantidade máxima de tokens que o modelo consegue processar simultaneamente em uma única passada pela rede.
