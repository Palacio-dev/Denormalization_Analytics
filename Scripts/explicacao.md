# Análise Estatística do Experimento de Desnormalização de Esquemas

Este documento explica o que o script `analyze_llm_schema_experiment.py` faz e por quê, para servir de referência e, se for útil, como base para a seção de metodologia do artigo.

## 1. Desenho experimental

O experimento é um **fatorial completo 6 × 3 × 16**, com medidas repetidas dentro de schema:

- **6 templates de prompt**: RISEN, RISEN-few-shot, JUNIOR, MID-LEVEL, SENIOR, OBT
- **3 modelos de LLM**: Gemini 3.1-Pro, qwen3-coder:480b-cloud, qwen3.5:4b
- **16 schemas relacionais** de referência

Cada um dos 16 schemas é desnormalizado sob as 18 combinações de prompt × modelo, gerando **288 observações**. Como o mesmo schema aparece 18 vezes, essas observações **não são independentes** — é isso que motiva o uso de um modelo de efeitos mistos em vez de, por exemplo, uma ANOVA comum.

O script confere isso automaticamente no início (`validate_design`): conta quantos schemas/prompts/modelos foram detectados no CSV, avisa se não bate com o esperado (16/6/3), e reporta se alguma das 288 combinações está faltando.

## 2. As seis variáveis dependentes

| Outcome | O que mede | Como é calculado | Faixa | Direção |
|---|---|---|---|---|
| `completeness` | Cobertura de atributos | `atributos_gerados / atributos_de_referência` | ≥ 0, sem teto fixo (pode passar de 1 se o LLM gerar atributos a mais) | maior é melhor |
| `type_correctness` | Corretude de tipos | proporção de atributos **pareados** com tipo correto | [0, 1] | maior é melhor |
| `field_correctness` | Corretude de nomes de atributos | distância de Levenshtein **média** entre nomes pareados | ≥ 0, sem teto fixo | **menor é melhor** (única métrica invertida) |
| `bleu` | Similaridade textual | BLEU padrão | [0, 1] | maior é melhor |
| `rouge` | Similaridade textual | ROUGE padrão | [0, 1] | maior é melhor |
| `meteor` | Similaridade textual | METEOR padrão | [0, 1] | maior é melhor |

Nenhuma dessas seis é uma contagem bruta limitada por `total_attributes` — todas já chegam do CSV como razões, proporções ou distâncias médias prontas. Por isso todas são modeladas como variáveis contínuas (ver seção 3), e não como proporções binomiais.

## 3. Modelo estatístico primário

Para cada um dos 6 outcomes, o script ajusta o mesmo modelo:

```
outcome ~ Prompt * Model + (1 | Schema)
```

- **Efeitos fixos**: Prompt (6 níveis), Model (3 níveis) e a interação Prompt × Model — testam se o tipo de prompt e o modelo (e a combinação dos dois) afetam o outcome.
- **Efeito aleatório**: intercepto aleatório por Schema — absorve o fato de que alguns schemas são intrinsecamente "mais fáceis" ou "mais difíceis" de desnormalizar, e modela corretamente a dependência das 18 medidas repetidas dentro de cada schema.
- É um **Linear Mixed Model (LMM)**, ajustado por máxima verossimilhança (`reml=False`), via `statsmodels`. ML (em vez de REML) foi escolhido para manter os ajustes comparáveis entre outcomes; se em algum momento vocês quiserem estimativas de variância mais precisas para reportar no artigo, REML é a alternativa mais indicada (basta trocar `reml=False` por `reml=True`).

**Ajuste numérico:** o otimizador `lbfgs` converge bem, mas em parte dos ajustes ele produz uma Hessiana aproximada que não é invertível, e o `statsmodels` falha ao calcular os erros-padrão (`Singular matrix`) mesmo com o modelo já convergido. Por isso o script tenta uma cadeia de otimizadores (`lbfgs` → `bfgs` → `cg` → `powell` → `nm`) e usa o primeiro que funcionar — isso é só um detalhe numérico, não muda a especificação do modelo.

## 4. Diagnóstico de resíduos

Para cada LMM, o script salva dois gráficos (`residuals_<outcome>.png` e `qq_<outcome>.png`):

- **Resíduos vs. valores ajustados**: para checar não-linearidade ou heterocedasticidade.
- **QQ-plot normal**: para checar se os resíduos seguem distribuição normal, premissa do LMM.

Vale olhar esses gráficos antes de reportar os resultados — se algum outcome mostrar desvio forte de normalidade (por exemplo, `field_correctness`, que é uma distância e tende a ser assimétrica à direita), pode valer a pena transformar a variável (ex. `log1p`) antes de reportar no artigo.

## 5. Correção para comparações múltiplas (FDR)

Cada LMM testa vários coeficientes (Prompt, Model, e suas interações) — isso é multiplicidade de testes dentro de um único outcome, e há ainda 6 outcomes sendo testados. Para controlar a taxa de falsos positivos, o script agrupa os outcomes em duas **famílias** e aplica a correção de **Benjamini-Hochberg (FDR)** separadamente dentro de cada uma:

- **Família estrutural**: `completeness`, `field_correctness`, `type_correctness` → `structural_fdr_results.csv`
- **Família NLP**: `bleu`, `rouge`, `meteor` → `lmm_fdr_results.csv`

Cada arquivo tem uma linha por coeficiente testado (por outcome), com `p_value` (bruto) e `p_fdr` (corrigido). Os termos de variância do efeito aleatório (`Group Var`) são excluídos dessa correção — eles não são testes de hipótese sobre Prompt/Model.

## 6. Comparações pareadas exploratórias (pós-hoc)

Além do LMM (que é a análise primária), o script roda comparações pareadas simples via **teste de Wilcoxon** (não-paramétrico), pareando pelo Schema: para cada nível do outro fator (ex. fixando o Model), compara cada par de níveis do fator de interesse (ex. RISEN vs. JUNIOR), usando apenas os schemas presentes nas duas condições. Os p-valores são corrigidos por FDR dentro de cada combinação outcome × fator, e salvos em `posthoc_<outcome>_<fator>.csv`.

**Importante**: essas comparações são rotuladas como exploratórias no próprio script — a inferência principal do estudo deve vir do LMM (seção 3), que já modela a estrutura de dependência corretamente. O Wilcoxon pareado é útil para uma leitura rápida e intuitiva de "qual prompt ganha de qual", mas não controla pela interação com o outro fator.

## 7. Estatísticas descritivas e correlações

- `descriptive_statistics.csv`: média, desvio-padrão, mediana, mínimo e máximo de cada outcome, por combinação prompt × modelo.
- `correlations_structural.csv` / `correlations_nlp.csv`: correlação de Spearman entre os outcomes dentro de cada família (ex.: `completeness` está associado a `bleu`?).
- `interaction_<outcome>.png`: gráfico de interação (médias por prompt, uma linha por modelo) — boa visualização complementar ao LMM.

## 8. Checagens de qualidade dos dados

Antes de qualquer modelagem, o script agora faz três checagens automáticas:

1. **Separador decimal**: converte vírgula → ponto automaticamente se detectar células com vírgula decimal (evita que a coluna inteira vire texto).
2. **Espaços em branco**: remove espaços extras nos níveis categóricos (schema/prompt/modelo), evitando que, por exemplo, `"RISEN"` e `"RISEN "` sejam contados como níveis diferentes.
3. **Faixa de valores**: avisa se `bleu`, `rouge`, `meteor` ou `type_correctness` saírem do intervalo `[0, 1]`, o que indica célula corrompida/deslocada na fonte dos dados (não é algo que o script deveria corrigir sozinho — é um sinal para conferir a origem dos dados).

## 9. Arquivos gerados (`results/`)

| Arquivo | Conteúdo |
|---|---|
| `descriptive_statistics.csv` | Médias/DP/mediana por prompt×modelo, todos os outcomes |
| `correlations_structural.csv`, `correlations_nlp.csv` | Correlações de Spearman entre outcomes |
| `lmm_<outcome>.csv` | Resumo completo do LMM ajustado |
| `lmm_<outcome>_fixed_effects.csv` | Só os coeficientes fixos (Prompt, Model, interação) |
| `lmm_fdr_results.csv` | P-valores corrigidos (FDR), família NLP |
| `structural_fdr_results.csv` | P-valores corrigidos (FDR), família estrutural |
| `posthoc_<outcome>_<fator>.csv` | Comparações pareadas exploratórias (Wilcoxon) |
| `interaction_<outcome>.png` | Gráfico de interação prompt × modelo |
| `residuals_<outcome>.png`, `qq_<outcome>.png` | Diagnóstico de resíduos do LMM |
| `analysis_summary.txt` | Resumo textual da execução |

## 10. Observações para o artigo

- `field_correctness` tem escala invertida em relação aos outros cinco outcomes (menor = melhor, por ser uma distância de edição). Vale deixar isso explícito na legenda de qualquer tabela/gráfico que a inclua ao lado das demais métricas.
- `completeness` não tem teto natural em 1 — um valor acima de 1 significa que o schema gerado tem mais atributos do que o de referência, não necessariamente que está "mais completo que o máximo". Isso é uma limitação da métrica (contagem, não identidade dos atributos), não do script.
- O suporte a GLMM binomial (`bambi`/`PyMC`) ainda existe no script (`GLMM_OUTCOMES`/`GLMM_SUCCESS_COLUMNS`), mas está vazio — nenhum dos seis outcomes atuais é uma contagem bruta limitada por `total_attributes`. Ele fica disponível caso surja uma métrica desse tipo no futuro.