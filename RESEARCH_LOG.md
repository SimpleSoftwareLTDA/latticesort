# Relatório de Pesquisa Autônoma (Autoresearch): Ordenação Inspirada em Geometrias e Fenômenos Naturais

## 1. Metodologia Autoresearch
Seguindo o protocolo de laço fechado de `autoresearch.lol`, o repositório foi estruturado em três componentes isolados:
* [harness.rs](file:///home/robson/.gemini/antigravity/scratch/metatron-sort-rs/src/harness.rs): avaliador imutável com validação estrita elemento a elemento contra `std::slice::sort_unstable` e cálculo da média geométrica de latência (`geomean_ms`) em 9 distribuições de $N = 1.000.000$ inteiros de 32 bits (mediana de 3 execuções por distribuição).
* [candidate.rs](file:///home/robson/.gemini/antigravity/scratch/metatron-sort-rs/src/candidate.rs): arquivo alvo modificável onde cada hipótese geométrica e física foi implementada e testada.
* [results.tsv](file:///home/robson/.gemini/antigravity/scratch/metatron-sort-rs/results.tsv): registro histórico acoplado aos commits do Git, retendo (`KEEP`) apenas mutações que reduziram a média geométrica ou estabeleceram avanços reais, e descartando (`DISCARD`) mutações inferiores.

---

## 2. Hipóteses Exploradas (Geometria, Constantes da Natureza e Física)

1. **Ressonância Orbital de Kirkwood e Razão Áurea ($\phi^{-1} \approx 0{,}6180339887$, `exp01`)**:
   * **Inspiração**: lacunas de Kirkwood no cinturão de asteroides, onde períodos em razões racionais simples entram em ressonância destrutiva, enquanto passos irracionais ou coprimos distribuem órbitas uniformemente.
   * **Mecanismo**: substituição do passo fixo `len / k` por amostragem coprima escalada por $2^{32}/\phi$ (`2_654_435_761`) somada a avaliação aritmética sem desvios (*branchless*) de pivôs.
   * **Efeito medido**: reduziu `Wide K=13` de `36.375 ms` para `7.827 ms` e a média geométrica de `36.639 ms` para `17.817 ms`.

2. **Nucleação Cristalina por Filotaxia Áurea (`0x9E3779B9`, `exp02`)**:
   * **Inspiração**: formação de núcleos cristalinos em soluções supersaturadas e empacotamento espiral de sementes na botânica regido pelo ângulo áureo.
   * **Mecanismo**: sondagem rápida de 64 pontos; se a entropia de amostra indica baixa cardinalidade, agrupa os elementos diretamente em uma grade de 128 posições na pilha indexada pelo multiplicador de Fibonacci `0x9E3779B9` e reconstrói os blocos contínuos via `slice::fill`.
   * **Efeito medido**: derrubou `Cyclic K=12, 13, 17` de `~8 ms` para `~0.9 ms` (`3.40x` a `3.65x` mais rápido que `std::slice::sort_unstable`) e a média geométrica para `7.746 ms`.

3. **Condensação Baricêntrica no Horizonte de Eventos (`exp03`)**:
   * **Inspiração**: discos de acreção gravitacional onde partículas dentro de um horizonte de energia $\Delta = E_{\max} - E_{\min}$ ocupam órbitas discretas relativas ao estado fundamental $E_{\min}$.
   * **Mecanismo**: varredura linear de extremos `(min_val, max_val)`; quando a amplitude $\Delta \le \min(n, 1.048.576)$, projeta cada inteiro diretamente no seu deslocamento orbital $(x - E_{\min})$ sem comparações.
   * **Efeito medido**: acelerou `Random K=100` (`5.10x`), `Random K=1000` (`7.84x`) e `Pareto Power-Law` (`6.50x`), reduzindo a média geométrica para `2.917 ms` (`2.20x` acima do `std`).

4. **Colapso de Onda Solitária / Soliton (`exp04` e `exp12`)**:
   * **Inspiração**: ondas solitárias (descritas na física de fluidos por equações de onda de canal raso) compostas por uma frente ascendente e uma cauda descendente coerente.
   * **Mecanismo**: detecção de pontos de inflexão monotônicos em tempo constante seguida de fusão linear lendo a cauda descendente de trás para frente, sem etapa intermediária de inversão em memória.
   * **Efeito medido**: reduziu `Soliton Wave` de `60.063 ms` para `1.796 ms` (`9.08x` mais rápido que `std::slice::sort_unstable`).

5. **Cascata de Bifurcação de Feigenbaum ($\delta \approx 4{,}6692$, `exp05`) vs. Constante Plástica ($\rho \approx 1{,}3247$, `exp06`)**:
   * **Inspiração**: constante universal de duplicação de período em sistemas dinâmicos caóticos ($\delta \approx 4{,}6692 \rightarrow 5$ ramos / 4 pivôs) comparada à razão plástica ternária (3 ramos / 2 pivôs).
   * **Efeito medido**: a cascata de 5 vias (`exp05`) reduziu `Uniform 32-bit Random` de `69.385 ms` para `50.614 ms` (`KEEP`), enquanto a redução para 3 vias (`exp06`) degradou para `52.452 ms` pelo aumento da profundidade $\log_3 n$ (`DISCARD`).

6. **Projeção Espectral Atômica Tri-Shell (`11-11-10`, `exp07`) vs. Tetra-Shell Hiper-Octaédrica (`8-8-8-8`, `exp08` e `exp10`)**:
   * **Inspiração**: decomposição de níveis eletrônicos em camadas espectrais com salto de camadas degeneradas (entropia zero quando todos os elementos compartilham o mesmo estado na camada).
   * **Mecanismo**: cálculo simultâneo dos 4 histogramas em uma única leitura desenrolada (`as_chunks::<4>()`) e dispersão em 256 fluxos que cabem integralmente no cache de dados L1 (Level 1 Cache, memória cache de primeiro nível do processador) e no TLB (Translation Lookaside Buffer, buffer de tradução de endereços de memória virtual), evitando cópia final graças ao número par de camadas.
   * **Efeito medido**: reduziu `Uniform 32-bit Random` de `50.614 ms` para `8.567 ms` (`exp07`) e depois para `5.224 ms` (`exp12`, `2.72x` mais rápido que `std::slice::sort_unstable`).

7. **Teorema da Rede Cristalina de Sentinela Dupla Auto-Verificável (`exp09`)**:
   * **Inspiração**: redes quase-cristalinas sem defeitos topológicos.
   * **Mecanismo matemático**: em uma tabela direta de 512 posições com função de dispersão $h(x)$, inicializa-se toda posição vazia $s \neq h(d_0)$ com $d_0$ e a posição $s_0 = h(d_0)$ com $d_1$ tal que $h(d_1) \neq s_0$. Como toda posição vazia contém um valor cujo *hash* difere do próprio índice da posição, a igualdade `keys[h(val)] == val` torna-se matematicamente impossível para posições vazias. Isso elimina a verificação de vetor booleano `used[s]` e elimina colisões via seleção harmônica de multiplicador na sondagem inicial.
   * **Efeito medido**: reduziu `Wide K=13` (amplitude de $10^9$) de `2.456 ms` para `0.740 ms` (`3.86x` mais rápido que `std::slice::sort_unstable`).

---

## 3. Histórico Completo de Experimentos (`results.tsv`)

| ID | Nome | Inspiração | Geomean (ms) | Std (ms) | Speedup | Status |
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
| `exp12` | Zero-Reverse Soliton + 4x Hist | Fusão reversa direta + scatter escalar | **1.060** | **5.302** | **5.00x** | **KEEP** |

---

## 4. Benchmark Final (`exp12`, $N = 1.000.000$)

| Dataset ($N = 1.000.000$) | Candidate (`exp12`) | `std::slice::sort_unstable` | Speedup |
| :--- | :---: | :---: | :---: |
| 1. Cyclic $K=12$ (não-13) | **0.747 ms** | 2.998 ms | **4.01x** |
| 2. Cyclic $K=13$ (base 13) | **0.754 ms** | 2.759 ms | **3.66x** |
| 3. Cyclic $K=17$ (primo) | **0.764 ms** | 2.939 ms | **3.85x** |
| 4. Wide $K=13$ (amplitude $10^9$) | **0.740 ms** | 2.855 ms | **3.86x** |
| 5. Random $K=100$ | **0.741 ms** | 4.543 ms | **6.13x** |
| 6. Random $K=1000$ | **0.761 ms** | 6.583 ms | **8.65x** |
| 7. Soliton Wave (bimodal) | **1.796 ms** | 16.307 ms | **9.08x** |
| 8. Pareto Power-Law | **1.004 ms** | 6.884 ms | **6.86x** |
| 9. Uniform 32-bit Random | **5.224 ms** | 14.223 ms | **2.72x** |
| **MÉDIA GEOMÉTRICA (SCORE)** | **1.060 ms** | **5.302 ms** | **5.00x** |
