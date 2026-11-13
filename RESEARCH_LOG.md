# Relatório de Pesquisa Autônoma (Autoresearch): Evolução Completa do `latticesort` (`exp00` a `exp19`)

## 1. Metodologia e Arquitetura da Biblioteca
Iniciado sob o protocolo de laço fechado de `autoresearch.lol`, o projeto evoluiu de um experimento de três arquivos para a biblioteca completa `latticesort`:
* [harness.rs](file:///home/robson/.gemini/antigravity/scratch/metatron-sort-rs/src/harness.rs): avaliador determinístico com validação elemento a elemento contra `std::slice::sort_unstable` e cálculo da média geométrica de latência (`geomean_ms`) em 9 distribuições de $N = 1.000.000$ inteiros de 32 bits (mediana de 3 execuções por distribuição).
* [candidate.rs](file:///home/robson/.gemini/antigravity/scratch/metatron-sort-rs/src/candidate.rs): núcleo adaptativo de 32 bits (`lattice_sort_i32`), integrando detecção estrutural, agregação em rede de sentinela dupla, contagem por amplitude densa e cascata Radix híbrida MSB (Most Significant Byte, byte mais significativo) + LSD (Least Significant Digit, dígito menos significativo).
* [kernel_i64.rs](file:///home/robson/.gemini/antigravity/scratch/metatron-sort-rs/src/kernel_i64.rs), [slice_ext.rs](file:///home/robson/.gemini/antigravity/scratch/metatron-sort-rs/src/slice_ext.rs) e [parallel.rs](file:///home/robson/.gemini/antigravity/scratch/metatron-sort-rs/src/parallel.rs): extensão polimórfica de 64 bits e ponto flutuante (`LatticeSortExt`) e motor paralelo multi-core (`parallel_lattice_sort`).
* [results.tsv](file:///home/robson/.gemini/antigravity/scratch/metatron-sort-rs/results.tsv): registro histórico de todas as 20 mutações (`exp00` a `exp19`).

---

## 2. Ambiente de Benchmark e Reprodutibilidade
Para evitar distorções de interpretação entre medições históricas, todas as baterias de referência foram consolidadas no seguinte hardware:
* **Processador**: `13th Gen Intel(R) Core(TM) i7-1360P` (12 núcleos físicos / 16 threads lógicos: 4 núcleos de performance + 8 núcleos de eficiência).
* **Hierarquia de Cache**: L1 (Level 1 Cache, memória cache de primeiro nível do processador) de dados de `48 KiB` por núcleo P (`32 KiB` por núcleo E), L2 (Level 2 Cache, memória cache de segundo nível do processador) de `1,25 MiB` por núcleo P (`9 MiB` somados em 6 instâncias) e L3 (Level 3 Cache, memória cache de terceiro nível compartilhada) de `18 MiB`.
* **Sistema e Compilador**: Linux x86_64, `rustc` (Edition 2024), perfil `release` com `opt-level = 3`, `lto = true` e `codegen-units = 1`.
* **Variação de Frequência e Imunização**: No estado `exp12` (pré-imunização Red-Team), o algoritmo sondava apenas uma janela inicial contígua, registrando `1.060 ms` (`5.00x` a `5.68x` conforme o estado térmico de Turbo Boost da CPU (Central Processing Unit, unidade central de processamento)). A partir do `exp13` e `exp14`, a introdução da sondagem em 5 quartis e do descascamento de prefixo (*prefix peeling*) imunizou o algoritmo contra todos os 9 ataques adversariais de Red-Team ao custo de `~0,05 ms` na sondagem inicial, estabilizando a média geométrica imunizada em `4.79x` a `4.99x` para $N = 10^6$, `4.08x` para $N = 10^7$ e `7.68x` (paralelo) para $N = 10^9$.

---

## 3. Hipóteses Exploradas (`exp01` a `exp19`)

1. **Ressonância Orbital de Kirkwood e Razão Áurea ($\phi^{-1} \approx 0{,}6180339887$, `exp01`)**:
   * **Mecanismo**: substituição do passo fixo `len / k` por amostragem coprima escalada por $2^{32}/\phi$ (`2_654_435_761`) somada a avaliação aritmética sem desvios (*branchless*) de pivôs.
   * **Efeito medido**: reduziu `Wide K=13` de `36.375 ms` para `7.827 ms` e a média geométrica de `36.639 ms` para `17.817 ms`.

2. **Nucleação Cristalina por Filotaxia Áurea (`0x9E3779B9`, `exp02`)**:
   * **Mecanismo**: sondagem rápida de baixa cardinalidade com agregação direta na pilha indexada pelo multiplicador de Fibonacci `0x9E3779B9` e reconstrução via `slice::fill`.
   * **Efeito medido**: derrubou `Cyclic K=12, 13, 17` de `~8 ms` para `~0.9 ms` (`3.40x` a `3.65x` sobre `std::slice::sort_unstable`).

3. **Condensação Baricêntrica por Amplitude (`exp03`)**:
   * **Mecanismo**: varredura linear de extremos `(min_val, max_val)`; quando a amplitude $\Delta \le \min(n, 1.048.576)$, projeta cada inteiro diretamente no deslocamento $(x - E_{\min})$ sem comparações.
   * **Efeito medido**: acelerou `Random K=100` (`5.10x`), `Random K=1000` (`7.84x`) e `Pareto Power-Law` (`6.50x`), reduzindo a média geométrica para `2.917 ms` (`2.20x`).

4. **Colapso de Onda Solitária / Soliton (`exp04` e `exp12`)**:
   * **Mecanismo**: detecção de pontos de inflexão monotônicos seguida de fusão linear de duas frentes lendo a cauda descendente de trás para frente, sem etapa intermediária de inversão em memória.
   * **Efeito medido**: reduziu `Soliton Wave` de `60.063 ms` para `1.796 ms` (`9.08x`).

5. **Cascata de 5 Vias (`exp05`) vs. 3 Vias (`exp06`)**:
   * **Efeito medido**: a cascata de 5 vias (`exp05`) reduziu `Uniform 32-bit Random` de `69.385 ms` para `50.614 ms` (`KEEP`), enquanto 3 vias (`exp06`) degradou para `52.452 ms` (`DISCARD`).

6. **Projeção Espectral Tri-Shell (`11-11-10`, `exp07`) vs. Tetra-Shell (`8-8-8-8`, `exp08`, `exp10`–`exp12`)**:
   * **Mecanismo**: cálculo simultâneo dos 4 histogramas em uma única leitura desenrolada (`as_chunks::<4>()`) e dispersão em 256 fluxos que cabem integralmente em `18 KB` do cache L1 e em 256 entradas do TLB (Translation Lookaside Buffer, buffer de tradução de endereços de memória virtual), com salto de camadas degeneradas.
   * **Efeito medido**: reduziu `Uniform 32-bit Random` de `50.614 ms` para `5.224 ms` (`2.72x` sobre `std`).

7. **Teorema da Rede de Sentinela Dupla Auto-Verificável (`exp09`)**:
   * **Mecanismo matemático**: em uma tabela direta com função de dispersão $h(x)$, inicializa-se toda posição vazia $s \neq h(d_0)$ com $d_0$ e a posição $s_0 = h(d_0)$ com $d_1$ tal que $h(d_1) \neq s_0$. Como toda posição vazia contém um valor cujo *hash* difere do próprio índice da posição, a igualdade `keys[h(val)] == val` torna-se matematicamente impossível para posições vazias.
   * **Efeito medido**: reduziu `Wide K=13` de `2.456 ms` para `0.740 ms` (`3.86x`).

8. **Imunização Red-Team: Soliton de Vale, Descascamento de Prefixo e 5 Janelas de Quartil (`exp13` e `exp14`)**:
   * **Mecanismo**: adição de fusão para ondas em formato de vale (decrescente seguido de crescente), ordenação local de sufixo quando um prefixo já está ordenado (*prefix peeling*), expansão da rede de sentinela dupla para `1.024` posições (`K <= 384`) e amostragem de entropia distribuída em 5 quartis (`0%, 25%, 50%, 75%, 100%`).
   * **Efeito medido**: aprovou 9 de 9 ataques adversariais no binário `adversarial` (`Iceberg Attack` em `0.77 ms`, `Valley Soliton` em `13.5x`, `Mid-Array Entropy Trap` em `3.09x`).

9. **Cascata Híbrida MSB + L2 em Dois Níveis para 10 Milhões e 1 Bilhão de Elementos (`exp15` e `exp18`)**:
   * **Mecanismo**: para `len >= 1_500_000`, executa particionamento MSB no Byte 3 em 256 baldes; quando um balde individual excede `262_144` elementos (`1 MB`, como ocorre em $N = 10^9$), aplica uma segunda divisão de 256 vias no Byte 2, reduzindo cada sub-balde de 1 bilhão de elementos para `~61 KB` (100% residente no cache L2).
   * **Efeito medido**: média geométrica de `4.08x` em $N = 10.000.000$ (`bench_10m`) e `2.81x` single-core (`7.966 s` vs `22.371 s`) em $N = 1.000.000.000$ (`bench_1b`).

10. **Polimorfismo IEEE 754 (`exp16`), Paralelismo Multi-Core (`exp17`) e Calibração de Nanosegundos (`exp19`)**:
    * **Mecanismo**: bijeção de bit de sinal para `f32`/`f64` e kernel de 8 passes para `i64`/`u64` (`LatticeSortExt`); coleta paralela de histogramas MSB e ordenação concorrente de 256 baldes disjuntos via `std::thread::scope` (`parallel_lattice_sort`); e ajuste de `SMALL_SORT_CUTOFF = 384` no ponto de cruzamento exato entre redes de ordenação in-place e Radix LSD em pilha (`stack_scratch = [0_i32; 2048]`).
    * **Efeito medido**: `3.71x` em `f32`, `7.68x` em 1 bilhão de elementos paralelos (`2.913 s` vs `22.371 s`) e paridade de `1.00x` a `1.06x` em listas minúsculas ($N \le 384$).

---

## 4. Histórico Completo de Experimentos (`results.tsv`)

| ID | Nome | Inspiração / Mecanismo | Geomean / Tempo (ms) | Std (ms) | Speedup | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `exp00` | Metatron 13-Ring Scratch | Cubo de Metatron (12 pivôs) | 36.639 | 6.053 | 0.17x | BASELINE |
| `exp01` | Kirkwood Branchless Metatron-13 | Lacunas de Kirkwood + árvore branchless | 17.817 | 5.849 | 0.33x | KEEP |
| `exp02` | Phyllotaxis Crystal Nucleation | Filotaxia áurea (`0x9E3779B9`) | 7.746 | 5.983 | 0.77x | KEEP |
| `exp03` | Barycentric Horizon Condensation | Horizonte de eventos ($E - E_{\min}$) | 2.917 | 6.404 | 2.20x | KEEP |
| `exp04` | Soliton Standing-Wave Collapse | Ondas solitárias bimodais | 2.101 | 6.368 | 3.03x | KEEP |
| `exp05` | Feigenbaum 5-Way Cascade | Constante de Feigenbaum ($\delta \approx 4{,}669$) | 1.936 | 5.993 | 3.09x | KEEP |
| `exp06` | Plastic 3-Way Cascade | Constante plástica ($\rho \approx 1{,}325$) | 1.924 | 6.133 | 3.19x | DISCARD |
| `exp07` | Tri-Shell 11-11-10 Spectral | Camadas quânticas `11-11-10` | 1.387 | 5.563 | 4.01x | KEEP |
| `exp08` | Tetra-Shell 8-8-8-8 L1 Spectral | Simetria 4D residente em L1 | 1.267 | 5.312 | 4.19x | KEEP |
| `exp09` | Self-Verifying Dual-Sentinel Lattice | Sentinela dupla $h(d_0) \neq s$ | 1.160 | 5.661 | 4.88x | KEEP |
| `exp10` | CMOV Soliton + 4x Unrolled Spectral | `as_chunks::<4>()` + CMOV (Conditional Move, instrução de movimentação condicional sem desvio) | 1.161 | 5.355 | 4.61x | PARTIAL |
| `exp11` | Speculative Soliton + 4x Scatter | Predição especulativa + scatter 4x | 1.084 | 5.442 | 5.02x | KEEP |
| `exp12` | Zero-Reverse Soliton + 4x Hist | Fusão reversa direta + scatter escalar | 1.060 | 5.302 | 5.00x | KEEP |
| `exp13` | Red-Team Immunized Generalizer | Valley Soliton + Prefix Peeling + 128-Nuclei | 1.085 | 5.225 | 4.82x | KEEP |
| `exp14` | Front 1: 1024-Lattice + Stack 2048 | 5 quartis + 1024 slots + stack buffer 2048 | 1.158 | 5.550 | 4.79x | KEEP |
| `exp15` | Front 2: MSB-First L2 Cascade ($10^7$) | Partição MSB p/ 256 blocos de 156 KB em L2 | 13.800 | 56.320 | 4.08x | KEEP |
| `exp16` | Front 3: `LatticeSortExt` (6 tipos) | Bijeção de sinal + Octa-Shell 64-bit | 6.750 | 25.040 | 3.71x | KEEP |
| `exp17` | Front 4: `parallel_lattice_sort` ($10^7$) | Histograma MSB paralelo + 256 blocos L2 | 35.430 | 186.910 | 5.28x | KEEP |
| `exp18` | Front 5: 1-Billion 2-Level Cascade ($10^9$) | Sub-baldes de 61 KB no Byte 2 (`4 GB`) | 2913.000 | 22371.000 | **7.68x** | **KEEP** |
| `exp19` | Front 6: Small-Slice Crossover (`384`) | Cruzamento L1 ($1024+5N$ vs $N\log_2 N$) | **1.119** | **5.589** | **4.99x** | **KEEP** |
