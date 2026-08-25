# Sazeti izvestaj projekta: Reinforcement Learning za BipedalWalker-v3 i BipedalWalkerHardcore-v3

## 1. Cilj projekta

Cilj projekta je bio:

1. da se uporede `PPO`, `SAC` i `TD3` na standardnom `BipedalWalker-v3`
2. da se proveri koliko se ti rezultati prenose na `BipedalWalkerHardcore-v3`
3. da se razvije jaci custom pristup za `hardcore` koristeci sekvencijalni model i dodatne trening trikove

Glavno pitanje projekta je bilo da li observation history, `LSTM` i anti-stall logika mogu da poprave rezultate na `hardcore` zadatku.

## 2. Tehnicka postavka

- Okruzenja: `BipedalWalker-v3` i `BipedalWalkerHardcore-v3`
- Observation space: `24`
- Action space: `4`
- Biblioteke: `Gymnasium`, `Stable-Baselines3`, `PyTorch`

U projektu postoje dve glavne grane:

- standardni `SB3` pipeline za `PPO`, `SAC` i `TD3`
- custom `hardcore` port sa sopstvenim `SAC/TD3` agentima

Custom `hardcore` port koristi:

- observation history
- `LSTM` ili `Transformer` backbone
- `frame_skip`
- `fall_penalty`
- periodicnu checkpoint evaluaciju
- anti-stall logiku tokom treninga

Tehnicki, observation history se pravi u `Gymnasium` wrapper-u koji slaze poslednjih `12` opservacija u sekvencu, a zatim se ta sekvenca prosledjuje `PyTorch` `torch.nn.LSTM` modulu; poslednje skriveno stanje LSTM-a koristi se kao ulaz u actor i critic mreze custom `SAC` agenta.

## 3. Eksperimentalni protokol

- Svi glavni prikazani run-ovi koriste jedan training seed: `42`
- Standardni `BipedalWalker-v3` eksperimenti su vodjeni sa `300,000` timesteps
- `Hardcore` SB3 baseline run-ovi imaju sopstvene trening budzete po algoritmu
- Evaluacija je radjena deterministicki
- Broj evaluacionih epizoda nije svuda isti i zato je naveden u tabelama

## 4. Rezultati na standardnom BipedalWalker-v3

Svi sledeci rezultati su dobijeni na `BipedalWalker-v3` sa `300,000` timesteps, `seed=42` i `5` evaluacionih epizoda.

| Algoritam | Mean reward | Std reward | Mean ep length | Kratak zakljucak |
|---|---:|---:|---:|---|
| `TD3` | `299.54` | `0.46` | `797.0` | najbolji mean reward i najstabilniji rezultat |
| `SAC` | `282.08` | `0.91` | `1181.6` | vrlo jak i veoma ujednacen hod |
| `PPO` | `125.28` | `131.29` | `989.6` | uci, ali je izrazito nestabilan |

Iz logova i videa za standardni zadatak mogu se izvesti sledeci kvalitativni zakljucci:

- `PPO` pokazuje najnestabilniji korak i najvecu varijansu izmedju epizoda
- `SAC` ima najujednaceniji hod
- `TD3` pokazuje najbolju ravnotezu i najbolju ukupnu kontrolu

## 5. Rezultati na BipedalWalkerHardcore-v3

### 5.1. SB3 hardcore baseline

| Algoritam | Trening budzet | Eval epizode | Mean reward | Std reward | Kratak zakljucak |
|---|---:|---:|---:|---:|---|
| `SB3 PPO` | `100,000,000` timesteps | `5` | `-98.81` | `20.90` | bolji od random baseline-a, ali ne resava zadatak |
| `SB3 TD3` | `300,000` timesteps | `5` | `-116.88` | `24.80` | prakticno neuspesan baseline |

Zakljucak je da dobri rezultati sa standardnog okruzenja ne prenose automatski na `hardcore`.

### 5.2. Custom hardcore port

| Eksperiment | Trening budzet | Eval epizode | Mean raw reward | Std reward | Kratak zakljucak |
|---|---:|---:|---:|---:|---|
| `Custom SAC-LSTM` legacy | `414,952` logovanih env koraka, `667` epizoda | `1` | `-31.64` | `0.00` | prvi signal da custom pristup pomaze u odnosu na SB3 baseline |
| `SAC-LSTM + anti-stall` breakthrough | `927,369` logovanih env koraka, do `ep3200` | `20` | `40.01` | `67.98` | prvi pozitivan multi-episode prosek |
| `SAC-LSTM + anti-stall` overall eval | `2,363,796` logovanih env koraka na celoj liniji | `5` | `160.90` | `113.97` | najjaci evaluacioni paket u repou |
| `SAC-LSTM + anti-stall` best periodic checkpoint | ista resumed linija | periodicne eval provere | `239.39` | `0.00` | najbolji zabelezen periodicni `raw_mean`, na `ep6000` |

Prakticno, glavni napredak u projektu nije dosao iz cistog povecanja timesteps budzeta, nego iz promene trening protokola i arhitekture:

- observation history
- `LSTM` backbone
- custom `SAC`
- checkpoint selekcija
- anti-stall trening logika

## 6. Sta je konkretno donela anti-stall izmena

Problem koji se pojavljivao u treningu bio je da agent:

- stane
- ostane u losem lokalnom optimumu
- prezivljava bez korisnog napretka

Uvedena je dodatna anti-stall logika tokom treninga, sa ciljem da se kazni neaktivnost i podstakne nastavak kretanja.

Praktican efekat te izmene je:

- vise pokusaja kretanja
- manje stagnacije
- bolji prolaz kroz prepreke
- prvi stvarni pozitivni pomaci na `hardcore` zadatku

## 7. Poredjenje sa javnim leaderboard-om

Napravljen je i presentation-friendly grafikon koji poredi najbolje `hardcore` rezultate iz projekta sa javnim `BipedalWalker-v3` leaderboard-om.

Najbolje reference iz tog poredjenja su:

- `SAC-LSTM + anti-stall (best episode)` sa `239.39`, priblizno oko `#47`
- `SAC-LSTM + anti-stall (overall)` sa `160.90`, priblizno oko `#64`
- `SAC-LSTM + anti-stall (breakthrough)` sa `40.01`, priblizno oko `#76`

Ovo poredjenje treba citati oprezno, jer javni CSV nije dedicated `BipedalWalkerHardcore-v3` leaderboard, nego `BipedalWalker-v3` leaderboard. Zato je taj grafikon koristan kao spoljna referenca, ali nije strogo apples-to-apples rangiranje za `hardcore`.

## 8. Glavni cinjenicni zakljucci

1. Na standardnom `BipedalWalker-v3` svi algoritmi mogu da nauce zadatak, ali su `SAC` i `TD3` znacajno bolji i stabilniji od `PPO`.
2. Na `BipedalWalkerHardcore-v3` standardni `SB3` baseline-i nisu dali upotrebljivo resenje.
3. Custom `SAC-LSTM` port je dao bolje rezultate od `SB3` baseline-a na `hardcore`.
4. Anti-stall logika je napravila prvi stvarni breakthrough: prelaz sa negativnih rezultata na pozitivan multi-episode prosek.
5. Najjaci rezultat u repou je resumed `SAC-LSTM + anti-stall` linija.
6. Najpostenija formulacija trenutnog statusa nije "problem je potpuno resen", nego "ostvaren je jak parcijalni uspeh, ali resenje jos nije dovoljno konzistentno preko vise seed-ova".

## 9. Ogranicenja

- Glavni eksperimenti nisu provereni preko vise training seed-ova
- Broj evaluacionih epizoda nije uniforman izmedju svih run-ova
- Neki custom checkpoint-i su evaluirani na malom broju epizoda
- Javni leaderboard koji je koriscen za spoljnu referencu nije `hardcore` leaderboard

## 10. Zavrsni zakljucak

Najvazniji rezultat projekta je da kombinacija custom `SAC` pristupa, observation history, `LSTM` backbone-a i anti-stall trening logike znacajno popravlja performanse na `BipedalWalkerHardcore-v3` u odnosu na standardne baseline-e.

Na standardnom zadatku najbolji rezultat je dao `TD3`, dok je na `hardcore` zadatku najvise obecao custom `SAC-LSTM + anti-stall` pristup. Projekat je zato dosao do jasnog i cinjenicno podrzanog zakljucka: za `hardcore` varijantu nisu dovoljni samo standardni RL baseline-i, nego je potrebno koristiti i sekvencijalni model, jacu kontrolu treninga i eksplicitnu borbu protiv stagnacije tokom ucenja.
