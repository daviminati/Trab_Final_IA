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

## Capítulo 3: Codificação de Mecanismos de Atenção (Coding Attention Mechanisms)

* **Mecanismo de Atenção (Attention Mechanism):** Componente central da arquitetura Transformer que permite ao modelo estabelecer relações dinâmicas e ponderadas entre diferentes tokens de uma sequência, capturando dependências contextuais de longo alcance independentemente da distância entre eles.
* **Self-Attention (Autoatenção):** Mecanismo de atenção em que cada token da sequência interage e calcula pontuações de relevância com todos os outros tokens da mesma sequência, permitindo a construção de representações contextuais ricas.
* **Scaled Dot-Product Attention (Atenção por Produto Escalar com Escala):** Formulação matemática específica para calcular os pesos de atenção utilizando o produto escalar entre vetores de consulta (*Query*) e chave (*Key*), dividido pela raiz quadrada da dimensão dos embeddings ($\sqrt{d_k}$) para estabilizar os gradientes durante o treinamento.
* **Query (Consulta - $Q$):** Vetor de representação que atua como o elemento ativo no cálculo da atenção, representando o token atual que está "procurando" por informações relevantes em outros tokens.
* **Key (Chave - $K$):** Vetor de representação associado a cada token da sequência, utilizado para ser comparado com as consultas (*Queries*) através de produto escalar, determinando o grau de alinhamento ou relevância.
* **Value (Valor - $V$):** Vetor de representação que contém o conteúdo informativo real do token; após a ponderação pelos pesos de atenção, os valores são somados para produzir a saída contextualizada.
* **Causal Attention (Atenção Causal / Masked Attention):** Variante do mecanismo de atenção utilizada em modelos geradores baseados em *Decoder* (como o GPT), que aplica uma máscara triangular superior para impedir que tokens futuros influenciem tokens passados, preservando a natureza autorregressiva da geração de texto.
* **Máscara Causal (Causal Mask):** Matriz de mascaramento com valores infinitos negativos (ou zeros após a função softmax) aplicada antes do cálculo da atenção para zerar a probabilidade de conexões com tokens futuros.
* **Multi-Head Attention (Atenção Multi-Cabeça):** Extensão do mecanismo de atenção que executa múltiplos cálculos de atenção em paralelo (várias "cabeças"), permitindo que o modelo foque simultaneamente em diferentes subespaços de representação e tipos de relações contextuais (ex: sintáticas e semânticas).
* **Dimensão do Embedding ($d_{emb}$ / Output Dimension):** Tamanho ou número de dimensões dos vetores densos que representam cada token no espaço vetorial contínuo da rede neural.
* **Número de Heads ($h$):** Quantidade de subespaços paralelos independentes em uma camada de *Multi-Head Attention*, dividindo a dimensão total do embedding de forma proporcional.
* **Dimensão da Head ($d_h$):** A dimensionalidade de cada cabeça de atenção individual, calculada tipicamente como a dimensão total do embedding dividida pelo número de heads ($d_h = d_{emb} / h$).
* **Matriz de Atenção (Attention Matrix / Attention Weights):** Matriz estocástica resultante da aplicação da função *softmax* sobre as pontuações de atenção normalizadas, cujas linhas somam 1 e indicam o peso relativo que cada token atribui aos demais.
(Autor: Elian Duarte)

## Capítulo 4: Implementando um Modelo GPT do Zero para Gerar Texto

* **Bloco Transformer (Transformer Block):** Módulo fundamental e recorrente na arquitetura GPT que combina mecanismos de atenção (*Multi-Head Attention*), redes neurais feed-forward, normalização de camada e conexões residuais.
* **Normalização de Camada (LayerNorm / Layer Normalization):** Técnica de normalização aplicada aos recursos ou ativações de uma camada neural para ajustar sua média para 0 e variância para 1, estabilizando o treinamento.
* **Conexões Residuais / Atalhos (Residual Connections / Shortcut Connections):** Mecanismo que adiciona diretamente a entrada de um bloco ou camada à sua saída, facilitando o fluxo de gradientes em redes profundas.
* **Rede Neural Feed-Forward (Feed Forward Network - FFN):** Sub-rede totalmente conectada composta por duas camadas lineares separadas por ativação não linear, expandindo a dimensionalidade para enriquecer as representações.
* **GELU (Gaussian Error Linear Unit):** Função de ativação suave e não linear que pondera os inputs com base em sua magnitude, permitindo gradientes suaves para valores negativos.
* **Dropout:** Técnica de regularização que desativa aleatoriamente uma fração dos neurônios durante o treinamento para prevenir o sobreajuste (*overfitting*).
* **Logits:** Vetores de pontuações brutas e não normalizadas geradas na última camada linear do modelo para cada token da sequência.
* **Cabeça de Modelo de Linguagem (Language Model Head):** Camada linear final da arquitetura que mapeia a representação vetorial oculta de volta para o espaço dimensional do vocabulário do tokenizer.
* **Inferência e Geração Autorregressiva (Autoregressive Generation):** Processo sequencial em que o modelo prevê um token por vez e alimenta a saída de volta à entrada para gerar texto coerente.
