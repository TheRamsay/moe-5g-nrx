## Compute-Aware Mixture-of-Experts for efficient 5G Neural Receiver
### Popis
- cílem je natrénovat neurální přijímač pro fyzicko vrstvu 5G, který bude výpočetně efektivní
- současné SOTA modely dosahují dobré přenosti, ale jsou statické a výpočet náročné
- naše práce je MoE architektura která bude mít heterogenní experty, výpočetní náročnost se bude přizpůsobovat podle kvality signálu
- pro čistý signál se použije jednoduchý model, pro signál který obsahuje hodně interference z prostředí se použije komplexnější model
- **vstup:** OFDM resource grid, 2D tenzor (čas x frekvence) který obsahuje reálnou a imaginární složku
- **výstup:** LLR (log likelihood ratio) odhady pro soft bit dekódování (to už zajištuje knihovna, není součást modelu)
### Dataset
- budeme generovat
- využijeme [NVIDIA Sionnna](https://developer.nvidia.com/sionna), 
- simulace standardního 5G slotu (14 symbolů, 3.5 Ghz, 16-QAM)
- pro trénink se využiji standardizované stochastické modely od 3GPP (UMA a TDL-C), dle [Morais et al., 2025](https://arxiv.org/html/2512.12449v1) nejlépe generalizují
- testování na OOD robustnost, použijí se deterministická data z DeepMIMO, které simulují reálné prostředí pomocí 3D ray-tracingu
- pokud bude velký rozdíl výsledků, vyzkoušíme fine-tuning na malém množství OOD vzorků
### Metriky a Evaluace
- BLER (Block Error Rate) - hlavní ukazatel spolehlivosti, cílem je dosáhnout porovnatelných výsledků se statickým modelem
- Average FLOPs - průměrný počet operací na slot, porovnat různé SNR
- BLER vs FLOPs tradefoff
- utilizace jednotlivých expertů
### Architektura modelu
- pro vzorek se udělá hrubý odhad kanálu pomocí Least Squares metody (Sionna)
- vstupem modelu je pak konkatenovaný vstupní tenzor s tímto odhadem 
- první vrstva modelu bude sdílená, slouží k extrakci feature a implicitnímu odhadu SNR/kanálu
- potom tam budou heterogenní experti, v základu použijeme tři, eventuelně můžeme vyzkoušet více
- router bude směrovat pakety na základě příznaků ze společné části
- při tréninku soft-gating přes gumbel softmax
- inference hard-gating (top 1) routing pro výpočetní efektivitu
- trénink vyzkoušet variantu 1) expert pretraining + router training a 2) joint training všeho dohromady
### Related Work
- [Wiesmayr et al. (2024)](https://arxiv.org/pdf/2409.02912): definice moderního neurálního přijímače, základní baseline, je to dense model -> výpočetně náročný
- [van Bolderik et al. (2024) - MEAN](https://ieeexplore.ieggee.org/document/10767787) - Proof-of-Concept aplikace MoE v 5G, pouze homogenní experti, spoléhají na exaktní znalost SNR (nereálné)
- [Song et al. (2025)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11162260) - koncept "channel-aware gating" v bezdrátových sítích, náš router se inspiruje
- [van Bolderik et al. (2026) - LOREN](https://arxiv.org/pdf/2602.10770) - autoři MEAN, důraz na efektivní využití paměti, různé LoRA adaptéry pro různé konfigurace sítě
- https://arxiv.org/pdf/2504.19660 - přehledová studie na různé use-casy neurálních přijímačů + MoE, hezké intro do problematiky, pros, cons, popis MoE
### Mozne vylepseni pokud bude cas
- 