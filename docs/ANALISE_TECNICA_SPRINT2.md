# Análise Técnica - Sprint 2: Processamento de Dados de Texto para LLMs

## Respostas fundamentadas aos Experimentos e Teoria

1. *Por que um LLM não pode trabalhar diretamente com texto bruto?*
   Redes neurais realizam operações algébricas (multiplicação de matrizes e ajustes de pesos por gradiente). Texto puro é categórico e não possui propriedades matemáticas diretas.

2. *Qual é a função do vocabulário?*
   Mapear de forma finita todas as unidades textuais (palavras/subpalavras) para números inteiros (Token IDs) únicos.

3. *Qual é a diferença entre um token e um Token ID?*
   * Token: Fragmento de texto (ex: " LLM").
   * Token ID: Representação numérica inteira associada àquele token (ex: 23781).

4. *Por que Token IDs não são representação semântica?*
   IDs são apenas números inteiros ordinais e arbitrários. O ID 1000 não possui o dobro de significado do ID 500.

5. *Qual é a função dos embeddings?*
   Projetar os Token IDs discretos em vetores densos $d_{emb}$, permitindo que palavras semanticamente semelhantes fiquem próximas no espaço vetorial.

6. *Por que é necessário representar a posição dos tokens?*
   O mecanismo de atenção processa todos os tokens da sequência em paralelo. Sem o Positional Embedding, o modelo não saberia a ordem temporal das palavras.

7. *Relação entre tamanho do contexto e quantidade de amostras:*
   Aumentar o tamanho do contexto (max_length) ou o stride reduz a quantidade total de amostras extraídas do texto bruto. Experimento no notebook: contexto 8/stride 4 gerou 1285 amostras, enquanto contexto 64/stride 32 gerou apenas 159.

8. *Impacto da dimensão do embedding ($d_{emb}$):*
   Dimensões maiores aumentam a capacidade semântica, mas aumentam proporcionalmente o uso de memória e a quantidade de parâmetros da rede neural. Experimento no notebook: $d_{emb}=128$ gera 6,5M de parâmetros, enquanto $d_{emb}=768$ gera 39,3M na camada.

9. *Função do DataLoader:*
   Agrupar as amostras sequenciais do Dataset em matrizes/lotes (batches) e realizar o embaralhamento (shuffle) para otimização em GPU.

10. *Conexão com a próxima Sprint (Mecanismo de Atenção):*
    A matriz de saída [Batch Size, Context Length, Embedding Dimension] será a entrada direta dos vetores de *Query ($Q$), Key ($K$) e Value ($V$)* do mecanismo de Atenção na Sprint 3.
