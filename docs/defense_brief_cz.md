# Příprava na konzultaci — Compute-Aware MoE pro 5G neurální přijímač

**Na zítřejší konzultaci u učitele.** Učitel viděl checkpoint report
(až po asym warm-start, 0,910 BLER / 61 % FLOPs). Tato konzultace je o všem,
co přišlo od té doby. Začni §0, abys ho zorientoval, pak zbytek na Q&A přípravu.

---

## 0. Co je nového od checkpoint reportu

Checkpoint skončil u: **asym warm-start, 0,910 BLER, 61 % FLOPs, jeden seed.**
Od té doby proběhlo ~10 experimentů. Celý přehled:

| Co | Výsledek |
|---|---|
| **Alpha sweep** (4 hodnoty α) | exp26 (α=2e-3) je koleno Pareta: **0,902 BLER / 56 % FLOPs** |
| **3-seed potvrzení** | Bimodální: 2/3 seedů reprodukují (s67, s42 → 0,902); **1/3 kolabuje** (s32 → 0,958) |
| **Stabilizační pokusy** × 2 | large-warmup → 100 % large; β-warmup → horší BLER. **Oba selhaly.** |
| **Ablace náhodného routeru** | Router input = šum → **BLER propadne o 6,6 pb** + kolaps na small. Channel featury jsou zásadní. |
| **Ablace 2 expertů** (bez nano) | **0,7 pb horší BLER + 9 pb více FLOPs.** Nano si svůj slot zaslouží. |
| **DeepMIMO OOD eval** | Všechny modely selhávají (~0,99 BLER) na ray-traced ASU campus datech. |
| **Few-shot fine-tune** (500 kroků, 2k OOD vzorků) | Žádné zlepšení. Poctivý scope: pouze syntetické kanály. |
| **SNR-oracle baseline** | Ručně psaná kaskáda s true SNR: 0,900 / 49 %. exp26 na Pareto frontéře bez oracle. |
| **Explicitní SNR-input ablace** (exp38) | Přidání channel statistik do routeru → **100% kolaps na large**. Implicitní stem featury stačí. |

---

## 1. Elevator pitch (30 sekund)

Postavili jsme 5G neurální přijímač, který **přizpůsobuje výpočet každému vzorku**.
Místo jednoho velkého modelu spuštěného na každém přijatém OFDM slotu routujeme
každý slot přes Mixture of Experts se třemi různými velikostmi (nano / small / large).
Malý router se podívá na kvalitu kanálu a vybere nejlevnějšího experta, který
daný slot dokáže dekódovat.

**Hlavní výsledek:** náš nejlepší model (`exp26`) dosahuje výsledku dense baseline
v rámci **0,1 procentního bodu BLER** při **56 % FLOPs**. Routovací chování se
router naučil sám z FLOPs penalizace v trénovací ztrátě — bez per-sample SNR
labelů.

---

## 2. Problém (30 sekund)

5G neurální přijímač dekóduje OFDM symboly na bity. Dense baseline (Wiesmayr et al.
2024) aplikuje stejný výpočet na každý slot bez ohledu na to, jak snadný nebo těžký
je. Většina slotů je snadná (vysoké SNR, přímá viditelnost) a šlo by je dekódovat
malou sítí. Jen ty těžké (oblast waterfall, nízké SNR) potřebují celý přijímač.

Statické dense přijímače tedy plýtvají výpočtem. Chceme výpočet, který se škáluje
s **obtížností kanálu**.

---

## 3. Architektura (1 minuta, případně nákres na tabuli)

Tři komponenty v sérii:

1. **Sdílený stem** — malé MLP, které zpracovává přijatý signál + LS odhad kanálu.
   Vždy běží (285M FLOPs). Produkuje reprezentaci featur, kterou používá router
   I vybraný expert.

2. **Channel-aware router** — tiny MLP, které bere poolované stem featury
   (mean+max pool přes frekvenci a čas) a vydává pravděpodobnost nad 3 experty.
   Trénován Gumbel-Softmaxem; **hard top-1 za inference** (takže běží jen JEDEN
   expert — úspory jsou reálné, ne amortizované).

3. **3 heterogenní experti** — stejná rodinná architektura, různá kapacita:
   - **nano** (90k params, 320M FLOPs celkem, 20 % z large)
   - **small** (168k params, 695M FLOPs celkem, 43 % z large)
   - **large** (450k params, 1604M FLOPs celkem, 100 %)

Ztráta: `BCE + γ·channel_MSE + α·E[FLOPs ratio] + β·load_balance`.
α je FLOPs penalizace — zvýšením se obchoduje BLER za rychlost. β brání
kolapsu na jednoho experta.

---

## 4. Cesta (část, kterou je dobré vyprávět jako příběh)

Zajímavá část projektu je **recept**, protože jsme před nalezením funkčního řešení
narazili na dva protichůdné módy selhání:

**Fáze 1 — společný trénink od nuly:** Všichni experti náhodná inicializace.
FLOPs penalizace nastoupí brzy, když žádný z nich ještě není dobrý, a router
se naučí „large je drahý, opusť ho." Výsledek: BLER 0,926 / 48 % FLOPs.
Levné, ale BLER trpí.

**Fáze 2 — plný warm-start:** Každý expert inicializován z předtrénovaného dense
checkpointu odpovídající velikosti. Router teď vidí, že warm-large je od kroku 1
striktně lepší než warm-nano/small, uzamkne se na large, nikdy neexploruje.
Výsledek: BLER 0,879 / **100 % FLOPs** — v podstatě doladěný dense large.

**Anti-collapse experimenty (5 pokusů, všechny selhaly):** silnější β,
β=2,0 vynucené rovnoměrné rozložení, kapacitní omezení, Switch-Transformer
pomocná ztráta — vše buď kolabuje, nebo ničí BLER.

**Fáze 3 — asymetrický warm-start (oprava):** stem + nano + small warm-startovány
z dense checkpointů, ale **large zůstane s náhodnou inicializací**. Large teď
nemá žádnou počáteční výhodu; router používá jen nano a small po ~6–8k kroků.
Pak large „procitne", jakmile se dotrénuje natolik, aby byl užitečný, a router
ho objeví. **Všichni tři experti aktivní** do kroku 10–12k.

Toto je recept, na kterém je postaveno vše ostatní.

---

## 5. Hlavní výsledek a proč mu věřit

Po tom, co asymetrický warm-start zafungoval, jsme spustili **sweep přes 4 hodnoty
FLOPs penalizace α**, abychom našli nejlepší operační bod.

| Run | α | Avg BLER | FLOPs % | Routing l/n/s | Verdikt |
|---|---:|---:|---:|---|---|
| Dense large | — | 0,901 | 100 % | — | reference |
| exp24 | 5e-4 | 0,898 | 100 % | 100/0/0 | α příliš slabé → kolaps na large |
| exp25 | 1e-3 | 0,907 | 56 % | 44/12/44 | dominováno |
| **exp26** | **2e-3** | **0,902** | **56 %** | 44/15/40 | **koleno Pareta** |
| exp27 | 5e-3 | 0,911 | 60 % | 37/0/63 | α příliš silné → nano vyhladověno |

**exp26 je 0,1 pb BLER od dense_large při 56 % FLOPs.** Striktní Pareto zlepšení.

---

## 6. Ablace (důkaz, že designové volby jsou zásadní)

To jsou otázky, které by položil ostrý recenzent. Odpověděli jsme na obě.

### „Používá router skutečně channel featury, nebo by mohl být náhodný?"
Nahradili jsme router input `torch.randn` (stejný tréninkový recept).
Výsledek: **BLER propadne o 6,6 pb** A router kolabuje na small (large se
nikdy nepoužije). **Channel-aware featury jsou zásadní** — přesně ústřední
tvrzení projektu.

### „Potřebujete skutečně tři experty, nebo by stačili dva?"
Odebrali jsme nano, trénovali jen s {small, large}. Výsledek: **0,7 pb horší BLER
+ 9 pb více FLOPs.** Nano není dekorativní — absorbuje beznadějné vzorky s nízkým SNR,
na kterých by small plýtval výpočtem.

---

## 7. Poctivá slabá místa (a jak je přiznáváme)

### Stabilita seedů
Znovu spustili α=2e-3 se seedy 32 a 42 vedle našeho hlavního seedu 67:
- s67: avg 0,902 ✓
- s42: avg 0,902 ✓
- **s32: avg 0,958, large kolaboval** ✗

**2 ze 3 seedů reprodukují; 1 uvázne ve fázi-2 atraktoru.** Vyzkoušeli jsme dva
stabilizační recepty (large-warmup, β-warmup); **oba selhaly** (large-warmup
překoriguje na 100 % large; β-warmup dává horší průměrný BLER). Reportujeme
transparentně.

### OOD generalizace
Testovali jsme na **DeepMIMO ray-traced datech** (ASU campus, 3,5 GHz). Všechny
tři naše modely — včetně dense large — katastrofálně selhávají (~0,99 BLER).
Pak jsme zkusili **few-shot fine-tune** (500 kroků, 2k OOD vzorků).
**Negativní výsledek:** dense_large 0,990 → 0,9901, exp26 0,992 → 0,9915.
500 kroků / 2k vzorků je **nedostatečných** pro překlenutí mezery syntetický vs.
ray-traced. Užitečné vedlejší zjištění: žádné katastrofální zapomínání na
in-distribution datech a routing se také neadaptoval (nano-default na OOD zachován).

### SNR-oracle baseline
Postavili jsme ručně psanou kaskádu, která používá **skutečné SNR** k výběru
nejlevnějšího experta, který splní toleranci BLER. S oracle SNR tato kaskáda
dosahuje **0,900 BLER při 49 % FLOPs**, čímž mírně dominuje exp26 (0,902 / 56 %).
**exp26 je na Pareto frontiéře bez oracle přístupu.** Zkoušeli jsme také
nakrmit router přímo signálovými statistikami (channel power, channel
variance — SNR proxy) → exp38 → 100% kolaps na large. Mezera vůči oracle
tedy zůstává, ale syrové statistiky nejsou způsob, jak ji uzavřít.

---

## 9. Co jsme udělali vs. předchozí práce

- **Wiesmayr et al. 2024** — definuje dense NRX architekturu, kterou používáme
  jako baseline.
- **MEAN (van Bolderik 2024)** — také MoE pro 5G NRX, ale s **homogenními experty**
  (stejný výpočet) a per-SNR specializací, nikoliv výpočetní efektivitou. Trénován
  jen na CDL-C. **Ortogonální příspěvek** — oni cílí na robustnost, my na výpočet.
- **Song et al. 2025** — channel-aware gating v bezdrátových sítích. My tuto myšlenku
  instanciujeme na NRX s explicitní FLOPs penalizací.

---

## 10. Pravděpodobné otázky učitele a krátké odpovědi

**Q: Proč heterogenní experti a ne jen jeden expert s proměnnou kapacitou?**
O: Heterogenní nám umožňuje skutečně přeskočit výpočet. Jeden přizpůsobitelný
expert by pořád musel na každém vzorku něco spustit.

**Q: Proč nepoužívat SNR přímo jako vstup routeru?**
O: Za inference nemáme ground-truth SNR. ZKOUŠELI jsme nakrmit router
signálovými statistikami korelujícími s SNR (channel power, channel
variance) — exp38, dnes. Výsledek: kolaps na 100 % large. Implicitní stem
featury překonávají explicitní syrové statistiky. Správně trénovaný
SNR-estimátor jako modul (ne syrové stats) by se mohl chovat jinak — to je
otevřený směr.

**Q: Jak víte, že router jen nepamatuje profil (UMa vs TDLC)?**
O: Dva důkazy: (a) per-SNR breakdowny ukazují, že router přechází uvnitř každého
profilu (small → large v oblasti waterfall), ne konstantní volbu per profil;
(b) ablace s náhodným routerem ukazuje, že bez channel featur model úplně
kolabuje — router je tedy skutečně POUŽÍVÁ.

**Q: Stačí jeden seed?**
O: Ne, a víme o tom. Spustili jsme 3 seedy; 2 reprodukují, 1 kolabuje. Vyzkoušeli
jsme dva stabilizační recepty, oba selhaly. Bimodální distribuci reportujeme
poctivě a doporučujeme best-of-N seedy.

**Q: Ukazujete jen syntetické Sionna výsledky. Co skutečné kanály?**
O: Testovali jsme na DeepMIMO ray-traced ASU Campus. Všechny modely — včetně
dense — selhávají bez OOD adaptace. Krátký few-shot fine-tune (500 kroků,
2k vzorků) nestačil na recovery. Poctivé scope statement: tato práce je pro
syntetické 3GPP kanály; překlenutí na ray-traced vyžaduje více dat a tréninkového
času, než jsme měli.

**Q: Jaký je příspěvek, když MEAN už udělal MoE pro NRX?**
O: Tři věci, které MEAN nemá:
  1. **Heterogenní velikosti expertů** — skutečná výpočetní heterogenita, ne jen
     specializace.
  2. **FLOPs penalizace v lossu** — emergentní compute-aware routing.
  3. **Pareto analýza** + ablace + oracle baseline — kompletní end-to-end
     charakterizace místo jediného čísla.

**Q: Co byste dělali dál? (před finálním deadlinem, 13 dní)**
O: Tři konkrétní experimenty motivované tím, co jsme právě zjistili:
  1. **Proper benchmark inference latence s efektivním dispatch.** Současný
     naivní top-1 dispatch (3 sekvenční sub-batche per forward pass)
     nepřevádí úsporu FLOPs do wall-clock. Implementovat production-grade
     dispatch (Mixtral, vLLM kernely) a re-benchmark exp26 vs dense_large
     při různých velikostech batche.
  2. **High-resolution per-SNR evaluace.** Současných 7 binů per profil
     nám dává 2 body v TDLC waterfall (14 dB a 18 dB). Resample po 1-2 dB
     krocích v rozsahu 10-20 dB, abychom přesně charakterizovali, kde se
     exp26 odchyluje od dense_large — a zda je BLER mezera uniformní nebo
     koncentrovaná v konkrétních SNR oblastech.
  3. **Interpretovatelnost routeru — co router *vidí*?** Máme předběžnou
     PCA stem featur na clusteru, ale neposunuli jsme to dál. Konkrétní
     plán: (a) PCA/UMAP stem featur obarvených podle vybraného experta
     a podle skutečného SNR, (b) saliency mapy přes vstupní grid ukazující,
     které subnosné/symboly řídí routovací rozhodnutí, (c) analýza
     aktivací per-expert featur pro to, na co se každý expert specializuje.

  Mimo tyto tři, dlouhodobější otevřené problémy zůstávají: **seed-stabilní
  tréninkový recept** (náš 1/3 kolabuje) a **OOD robustnost** (DeepMIMO).

---

## 11. Status (kde jsme vs. deadline)

- **Experimenty:** v podstatě hotové. Headline + 3 ablace + 3-seed +
  oracle baseline + OOD + few-shot OOD = **~10 různých experimentů
  hotových s výsledky**.
- **Dokumentace:** `docs/final_draft.md` je kompletní narativní draft. Oficiální
  `docs/checkpoint_report.md` stále potřebuje LaTeX přepis — skutečný zbývající
  bottleneck.
- **Poster:** nezačatý.
- **Čas do deadline:** 13 dní.

Máme dost prostoru. Riziko je **provedení writeup**, ne samotná práce.

---

## 12. Jeden slide / jedno číslo na každé tvrzení

Pokud máš čas jen na jeden bullet na tvrzení:

- **Compute-aware MoE funguje:** exp26 dosahuje dense BLER při 56 % FLOPs.
- **Router skutečně používá channel featury:** ablace s náhodným routerem
  ztrácí 6,6 pb BLER.
- **3 experti > 2 experti:** odebrání nano stojí 0,7 pb BLER a 9 pb FLOPs.
- **Asym warm-start je jediný recept, který funguje:** Fáze 1 a Fáze 2
  obě čistě selhávají; charakterizovali jsme obě.
- **Seed-stabilní? Většinou:** 2/3 seedů reprodukují; 1 kolabuje. Reportováno.
- **OOD generalizace?** Pouze syntetický scope; krátký fine-tune nestačí.
- **Překonáni oraclem?** Mírně (49 % vs. 56 % FLOPs při stejném BLER).
  Zkoušeli jsme nakrmit router syrovými SNR-proxy stats (exp38) → kolaps.
  Implicitní featury překonávají explicitní syrové stats.
- **Explicitní SNR proxy zkoušeny (exp38):** 100% kolaps na large.
  Implicitní stem featury jsou dostačující.

To je celý projekt.
