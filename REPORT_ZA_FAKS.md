# Izvestaj projekta: Reinforcement Learning za BipedalWalker-v3 i BipedalWalkerHardcore-v3

## 1. Tema i motivacija

Tema projekta je primena reinforcement learning-a na problem upravljanja
bipedalnim robotom u Gymnasium okruzenjima:

- `BipedalWalker-v3`
- `BipedalWalkerHardcore-v3`

Osnovna ideja je da agent ne dobija direktnu instrukciju "kako da hoda", nego
samostalno uci politiku upravljanja motorima na osnovu:

- trenutne opservacije stanja
- izabrane akcije
- dobijenog reward-a

Ovaj problem je interesantan zato sto spaja:

- continuous control
- nelinearnu dinamiku
- potrebu za koordinacijom vise aktuatora
- delom parcijalno opservabilan zadatak, posebno u `hardcore` rezimu

Prakticni cilj projekta je dvostruk:

1. implementirati i testirati standardni trening pipeline za `PPO`, `SAC` i `TD3`
2. razviti namenski `hardcore` port sa custom `SAC/TD3` agentima i
   sekvencijalnim modelima (`LSTM` i `Transformer`)

## 2. Glavna istrazivacka pitanja

Projekat pokusava da odgovori na sledeca pitanja:

1. Kako se ponasaju `PPO`, `SAC` i `TD3` na standardnom `BipedalWalker-v3`
   zadatku?
2. Zasto isti algoritmi cesto ne prenose dobre performanse na
   `BipedalWalkerHardcore-v3`?
3. Da li observation history i sekvencijalne mreze mogu da pomognu kod
   `hardcore` varijante?
4. Da li je za tezu varijantu zadatka potrebno pratiti vise checkpoint-a umesto
   da se veruje samo poslednjem modelu?

## 3. Pregled projekta i arhitekture

Projekat sada ima dve glavne grane.

### 3.1. Standardni SB3 pipeline

Ovu granu cine:

- `train_bipedal_walker.py`
- `train_bipedal_hardcore.py`
- `bipedal_workflow.py`

Ona koristi `Stable-Baselines3` implementacije:

- `PPO`
- `SAC`
- `TD3`

Glavni tok rada je:

1. CLI parsira argumente
2. bira se algoritam
3. pravi se env
4. trenira se model
5. model se cuva
6. radi se evaluacija
7. po potrebi se racuna random baseline
8. po potrebi se snimaju video epizode

### 3.2. Namenski hardcore port

Ovu granu cine:

- `train_bipedal_hardcore_port.py`

Ona ne koristi gotove `SB3` agente za `hardcore`, nego uvodi:

- custom `SAC` i `TD3`
- observation history
- `LSTM` i `Transformer` actor-critic mreze
- frame skip i reward shaping
- episode-based trening
- periodicne evaluacije i vise tipova checkpoint-a

## 4. Formulacija problema

U reinforcement learning postavci, agent na svakom koraku `t` dobija stanje
`s_t`, bira akciju `a_t`, okruzenje vraca reward `r_t` i novo stanje
`s_{t+1}`.

Cilj je maksimizacija epizodnog povrata:

\[
G = \sum_{t=0}^{T-1} r_t
\]

Ako se kroz vise epizoda testira naucena politika, prosecna performansa je:

\[
\overline{G} = \frac{1}{N}\sum_{i=1}^{N} G_i
\]

gde je:

- `N` broj evaluacionih epizoda
- `G_i` povrat `i`-te epizode

Standardna devijacija reward-a se racuna kao:

\[
\sigma_G = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(G_i - \overline{G})^2}
\]

Ova metrika je veoma vazna zato sto govori ne samo koliko je agent dobar, nego i
koliko je stabilan.

## 5. Okruzenje

Koriscena su sledeca Gymnasium okruzenja:

- `BipedalWalker-v3`
- `BipedalWalkerHardcore-v3`

Osnovne karakteristike:

- observation space: vektor dimenzije `24`
- action space: vektor dimenzije `4`
- akcije su kontinualne i ogranicene na interval `[-1, 1]`

Agent upravlja motorima oba kuka i oba kolena. Zato problem nije samo "idi
napred", nego:

- odrzi ravnotezu
- koordinisi levu i desnu nogu
- izbegni pad
- kod `hardcore` moda savladaj i prepreke

## 6. Zasto je hardcore mod znacajno tezi

`BipedalWalkerHardcore-v3` uvodi mnogo tezu distribuciju staza:

- neravnine
- jame
- prepreke
- zahtevniju dinamiku kretanja

Prakticno, to znaci da agent mora da donosi odluke koje zavise ne samo od
trenutnog stanja, nego i od skorije istorije kretanja. Zbog toga se problem u
praksi moze posmatrati kao delom parcijalno opservabilan.

Ako model vidi samo jedno stanje `s_t`, onda cesto ne zna dovoljno o:

- prethodnom zamahu tela
- prethodnim kontaktima stopala sa tlom
- trendu promene ugla i brzine tela

Zato je observation history jedna od kljucnih ideja custom `hardcore` porta.

## 7. Algoritmi korisceni u projektu

U projektu postoje dve klase implementacija:

- standardni `SB3` algoritmi
- custom `hardcore` algoritmi

### 7.1. PPO

`PPO` (`Proximal Policy Optimization`) je on-policy actor-critic algoritam koji
kontrolise koliko nova politika sme da odstupi od stare.

Clipped ciljna funkcija je:

\[
L^{CLIP}(\theta) =
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta)\hat{A}_t,\;
clip(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t
\right)
\right]
\]

gde je odnos verovatnoca:

\[
r_t(\theta) =
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{old}}(a_t \mid s_t)}
\]

PPO u projektu sluzi kao jaka referenca za standardni pipeline, ali nije glavni
fokus custom `hardcore` porta.

### 7.2. SAC

`SAC` (`Soft Actor-Critic`) je off-policy algoritam za continuous control koji
optimizuje i reward i entropiju politike.

U regularizovanom obliku cilj politike moze da se pise kao:

\[
J(\pi) =
\mathbb{E}
\left[
\sum_t
\big(
r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot \mid s_t))
\big)
\right]
\]

U custom portu critic target je:

\[
y_t =
r_t +
\gamma (1-d_t)
\left(
\min(Q_1'(s_{t+1}, a_{t+1}), Q_2'(s_{t+1}, a_{t+1}))
- \alpha \log \pi(a_{t+1}\mid s_{t+1})
\right)
\]

gde je:

\[
a_{t+1} \sim \pi(\cdot \mid s_{t+1})
\]

Critic loss je:

\[
\mathcal{L}_{Q_i} =
\mathbb{E}
\left[
\big(Q_i(s_t, a_t) - y_t\big)^2
\right]
\]

Actor loss je:

\[
\mathcal{L}_{\pi} =
\mathbb{E}
\left[
\alpha \log \pi(a_t \mid s_t)
- \min(Q_1(s_t, a_t), Q_2(s_t, a_t))
\right]
\]

Ova postavka je posebno zanimljiva za `hardcore` jer entropijska regularizacija
moze da pomogne kod istrazivanja u teskom i sparse-like okruzenju.

### 7.3. TD3

`TD3` (`Twin Delayed Deep Deterministic Policy Gradient`) unapredjuje `DDPG`
preko tri kljucne ideje:

1. dva critic-a
2. delayed actor updates
3. target policy smoothing

Target akcija se formira kao:

\[
a_{t+1}' =
clip\big(
\mu_{\theta'}(s_{t+1}) + \epsilon,\;
a_{min}, a_{max}
\big)
\]

gde je:

\[
\epsilon \sim clip(\mathcal{N}(0,\sigma^2), -c, c)
\]

TD target je:

\[
y_t =
r_t +
\gamma (1-d_t)
\min(Q_1'(s_{t+1}, a_{t+1}'), Q_2'(s_{t+1}, a_{t+1}'))
\]

Critic loss je:

\[
\mathcal{L}_{Q_i} =
\mathbb{E}
\left[
\big(Q_i(s_t, a_t) - y_t\big)^2
\right]
\]

Actor loss je:

\[
\mathcal{L}_{\mu} =
-\mathbb{E}[Q_1(s_t, \mu(s_t))]
\]

TD3 je jak za continuous control, ali u `hardcore` modu moze biti osetljiviji na
istrazivanje od `SAC`.

## 8. Observation history i parcijalna opservabilnost

Standardni feed-forward model koristi samo trenutno stanje `s_t`.
Custom `hardcore` port uvodi istoriju opservacija:

\[
H_t = [o_{t-h+1}, o_{t-h+2}, \ldots, o_t]
\]

gde je:

- `o_t \in \mathbb{R}^{24}`
- `H_t \in \mathbb{R}^{h \times 24}`
- `h` duzina istorije

U projektu su podrazumevane vrednosti:

- `SAC`: `h = 12`
- `TD3`: `h = 6`

Ova istorija se zatim prosledjuje sekvencijalnom encoder-u umesto jedne
trenutne observation vrednosti.

## 9. Sekvencijalni modeli: LSTM i Transformer

### 9.1. LSTM

`LSTM` (`Long Short-Term Memory`) prima sekvencu ulaza i racuna:

\[
(h_t, c_t) = LSTM(x_t, h_{t-1}, c_{t-1})
\]

Poslednje skriveno stanje koristi se kao sazet prikaz istorije:

\[
z_T = h_T
\]

U projektu `LSTM` sluzi da model zapamti kratkorocne i srednjorocne obrasce
kretanja, kao sto su:

- ritam koraka
- trend promene nagiba tela
- smena kontakta nogu sa tlom

### 9.2. Transformer

Kod `Transformer` pristupa sekvenca se prvo embeduje:

\[
e_t = W_{emb}x_t + p_t
\]

gde je `p_t` poziciono kodiranje.

Zatim se koristi attention:

\[
Attention(Q, K, V) =
softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

U ovoj implementaciji naglasak je na "last-step" citanju, tj. poslednji element
sekvence koristi se kao query nad celom istorijom.

To znaci da model moze adaptivno da proceni:

- koji prethodni trenuci su trenutno najvazniji
- koji delovi istorije nose najvise korisne informacije

## 10. Hardcore wrapper: frame skip i reward shaping

Custom `hardcore` port koristi namenski wrapper.

### 10.1. Frame skip

Ako je `k` broj ponavljanja iste akcije, onda u wrapper koraku koristimo:

\[
a_t = a_{t,0} = a_{t,1} = \cdots = a_{t,k-1}
\]

U projektu je podrazumevano:

\[
k = 2
\]

To efektivno smanjuje frekvenciju kontrole i cesto olaksava agentu da nauci
stabilniji hod.

### 10.2. Reward shaping

Ako robot padne, terminalna kazna se menja na:

\[
\bar{r}_{t,j} =
\begin{cases}
-10, & \text{ako je robot pao} \\
r_{t,j}, & \text{inace}
\end{cases}
\]

Zatim se kroz frame skip sabira shaped reward:

\[
\tilde{r}_t = \sum_{j=0}^{k-1} \bar{r}_{t,j}
\]

Zbog toga projekat sada razlikuje:

- `raw reward`
  originalni reward iz okruzenja
- `shaped reward`
  reward koji zaista koristi custom trening

Ovo je vazno jer oblikovanje reward-a moze ubrzati ucenje, ali za objektivnu
procenu i dalje je bitno gledati originalni env skor.

## 11. Replay buffer i off-policy ucenje

I `SAC` i `TD3` koriste replay buffer:

\[
\mathcal{D} =
\{(s_t, a_t, r_t, s_{t+1}, d_t)\}
\]

Mini-batch se nasumicno uzorkuje:

\[
(s, a, r, s', d) \sim U(\mathcal{D})
\]

Prednosti replay buffer-a:

- smanjuje korelaciju uzastopnih primera
- omogucava visestruko koriscenje iskustva
- cini off-policy algoritme efikasnijim od cisto rollout pristupa

## 12. Soft update target mreza

I `SAC` i `TD3` koriste target mreze koje se ne osvezavaju naglo, nego meko:

\[
\theta' \leftarrow \tau \theta + (1-\tau)\theta'
\]

gde je:

- `\theta` parametar glavne mreze
- `\theta'` parametar target mreze
- `\tau` mali koeficijent, u projektu podrazumevano `0.01`

To smanjuje oscilacije target vrednosti i poboljsava stabilnost treninga.

## 13. Episode-based trening protokol custom porta

Za razliku od klasicnog SB3 toka po `timesteps`, custom `hardcore` port koristi
trening po epizodama.

Jedna epizoda ide ovako:

1. reset env-a
2. rollout do kraja epizode ili do `max_steps = 750`
3. svaka tranzicija se upisuje u replay buffer
4. posle epizode se radi onoliko gradient update-a koliko je epizoda imala
   wrapper koraka

Ako epizoda ima `T` wrapper koraka, onda je broj update-a:

\[
N_{update} = T
\]

Prvih `50` epizoda po default-u sluze kao inicijalna exploration faza pre punog
ritma ucenja.

Po default-u:

- evaluacija ide na svakih `200` epizoda
- svaka evaluacija koristi `20` epizoda
- finalna evaluacija koristi `100` epizoda

## 14. Checkpoint evaluacija i selekcija modela

Jedna od najvaznijih novina u projektu je checkpoint evaluacija.

Custom port cuva:

- `epN.pt`
  checkpoint posle evaluacije na epizodi `N`
- `best_raw.pt`
  najbolji checkpoint po originalnom reward-u
- `best_shaped.pt`
  najbolji checkpoint po shaped reward-u
- `last.pt`
  poslednji checkpoint treninga

Ako je prosecni raw reward pri evaluaciji:

\[
\overline{G}_{raw}^{(N)} = \frac{1}{N}\sum_{i=1}^{N} G_{raw}^{(i)}
\]

onda se `best_raw` osvezava kada vazi:

\[
\overline{G}_{raw}^{(N)} > \overline{G}_{raw,best}
\]

Analogno, za shaped reward:

\[
\overline{G}_{shaped}^{(N)} = \frac{1}{N}\sum_{i=1}^{N} G_{shaped}^{(i)}
\]

Checkpoint selekcija je kriticna kod `hardcore` okruzenja zato sto poslednji
model nije obavezno i najbolji model.

## 15. Implementacioni detalji iz konkretnog koda

### 15.1. Standardni pipeline

U `bipedal_workflow.py` se sada vide:

- standardni `SB3` trening helper-i
- normalizacija za neke `PPO` hardcore postavke
- env helper-i kao sto su `frame_skip`, observation history i `fall_penalty`
- evaluacija sa odvojenim `raw` i `shaped` reward metrikama

Drugim recima, standardni pipeline vise nije samo osnovni wrapper oko SB3
algoritama, nego ima i `hardcore-aware` pomocne funkcije.

### 15.2. Namenski hardcore port

U `train_bipedal_hardcore_port.py` su implementirane najvaznije nove komponente:

- observation history wrapper
- custom `SAC`
- custom `TD3`
- `LSTM` actor-critic
- `Transformer` actor-critic
- replay buffer
- `TD3` exploration noise
- periodična evaluacija i checkpoint sistem

To znaci da projekat sada ima i bibliotekarsku i istrazivacku granu.

## 16. Eksperimentalni rezimi koji vec postoje u projektu

Ovde je vazno razlikovati:

- rezultate koji su vec ranije dobijeni i sacuvani
- novu implementaciju koja je sada spremna za ozbiljne eksperimente

### 16.1. Rezultati za obicni `BipedalWalker-v3`

U notebook-u `result_analsys.ipynb` navedeni su sledeci rezultati:

| Algoritam | Mean reward | Std reward | Priblizan rank po mean reward |
|---|---:|---:|---:|
| TD3 | 299.54 | 0.46 | 29 |
| SAC | 282.08 | 0.91 | 37 |
| PPO | 125.28 | 131.29 | 68 |

Tumacenje:

- `TD3` je imao najbolji prosecan rezultat
- `SAC` je vrlo jak i stabilan
- `PPO` je radio, ali uz veoma veliku varijansu

### 16.2. Legacy hardcore summary rezultati

U `artifacts/summaries/` ranije su postojali sledeci `hardcore` eksperimenti:

| Algoritam | Timesteps | Mean reward | Std reward | Mean ep length |
|---|---:|---:|---:|---:|
| PPO | 5,000,000 | -48.74 | 18.89 | 2000 |
| TD3 | 300,000 | -116.88 | 24.80 | 2000 |

Ovi rezultati su korisni kao dokaz zasto je bio potreban novi `hardcore` port:

- `PPO` i `TD3` u starom toku nisu resili problem
- `hardcore` je ostao duboko ispod nule
- standardni pipeline nije bio dovoljan

### 16.3. Status novog custom hardcore porta

Nova implementacija `train_bipedal_hardcore_port.py` je:

- sintaksno proverena
- smoke-testirana za `SAC + LSTM`
- smoke-testirana za `TD3 + LSTM`
- smoke-testirana za `SAC + Transformer`
- proverena za checkpoint save/load i poseban `test-100` mod
- proverena za snimanje `mp4` videa iz checkpoint evaluacije
- dopunjena anti-stall eksperimentom

Pored same implementacije, lokalno je sada slozen i uredjen report paket pod:

- `artifacts/archive/reports/runs/INDEX.md`

Najbitniji trenutno spakovani eksperimenti su:

| Eksperiment | Mean reward | Kratak zakljucak |
|---|---:|---|
| SB3 PPO Hardcore baseline | -98.81 | bolji od random baseline-a, ali ne resava zadatak |
| SB3 TD3 Hardcore baseline | -116.88 | prakticno neuspesan baseline |
| Custom SAC + LSTM best raw checkpoint | -22.45 | najjaci rezultat u repou, stabilan ali jos negativan |
| Custom SAC + LSTM anti-stall transfer check | -74.50 | anti-stall pomaze treningu, ali slabije transferuje na cist env |

To znaci da custom port vise nije samo "spreman za eksperimente", nego vec ima
konkretne srednje-duge rezultate koji se mogu analizirati i porediti.

Bitna akademska ograda i dalje ostaje:

- najbolji custom checkpoint je i dalje ispod nule
- `hardcore` jos nije potpuno resen
- ali metodologija sada daje vidljiv napredak u odnosu na legacy baseline

## 17. Metrike pracenja u projektu

U projektu se sada koriste sledece metrike:

### 17.1. Raw epizodni reward

\[
G_{raw} = \sum_{t=0}^{T-1} r_t
\]

To je najvernija metrika uspeha u samom env-u.

### 17.2. Shaped epizodni reward

\[
G_{shaped} = \sum_{t=0}^{T-1} \tilde{r}_t
\]

To je metrika po kojoj custom port stvarno uci.

### 17.3. Moving average preko poslednjih 100 epizoda

Ako su poslednji reward-i `G_{t-99}, ..., G_t`, onda je:

\[
MA_{100}(t) = \frac{1}{100}\sum_{i=t-99}^{t} G_i
\]

Ova metrika sluzi za pracenje stabilnijeg trenda ucenja.

### 17.4. Improvement vs random baseline

Za standardni pipeline ostaje korisna i metrika:

\[
\text{improvement\_vs\_random} =
\overline{G}_{model} - \overline{G}_{random}
\]

## 18. Zasto je uveden novi custom hardcore port

Glavni razlog je sledeci:

- stari `hardcore` eksperimenti nisu davali zadovoljavajuce rezultate
- uspesni radovi za isti problem cesto koriste observation history
- sekvencijalni modeli (`LSTM`, `Transformer`) bolje hvataju dinamiku kroz vreme
- checkpoint evaluacija je neophodna jer je trening nestabilan

Drugim recima, problem nije bio samo u "jos vise koraka treninga", nego i u
samoj strukturi ulaza, modela i protokola evaluacije.

## 19. Predlozeni prvi ozbiljan eksperiment

Najrazumniji prvi ozbiljan pokusaj u novom portu je:

```powershell
.\.venv\Scripts\python.exe train_bipedal_hardcore_port.py --algo sac --backbone lstm
```

Razlog:

- `SAC` bolje istrazuje od `TD3`
- `LSTM` je prema referentnim radovima i intuiciji vrlo jak izbor za ovaj tip
  zadatka
- podrazumevani `history_length = 12` daje modelu duzu istoriju stanja

## 20. Ogranicenja trenutne verzije projekta

Najvaznija ogranicenja su:

- nema jos poredjenja preko vise seed-ova za custom port
- nema jos pune tabele rezultata `LSTM` vs `Transformer`
- shaping moze poboljsati ucenje, ali uvodi razliku izmedju trening signala i
  finalne env metrike
- anti-stall eksperiment pokazuje da bolji trening signal ne znaci automatski i
  bolji transfer na cist `Hardcore`
- `hardcore` ostaje tesko i nestabilno okruzenje cak i sa unapredjenom
  metodologijom

## 21. Predlog za dalji rad

Najvazniji naredni koraci su:

1. nastaviti najjaci `custom SAC + LSTM` run i posebno pratiti `best_raw`
   checkpoint-e umesto oslanjanja na `last`
2. uraditi vise seed-ova za `custom SAC + LSTM`
3. probati inicijalizaciju iz vec istreniranog standardnog policy-ja i zatim
   fine-tune na `Hardcore`
4. zadrzati anti-stall kao pomocni trening rezim, ali evaluaciju raditi i bez
   anti-stall pravila
5. napraviti learning curves za `raw` i `shaped` reward
6. uraditi eksplicitno poredjenje observation history duzina `6` i `12`
7. po potrebi dodatno tunirati:
   `alpha`, `gamma`, `tau`, `batch_size`, `eval_frequency`

## 22. Zavrsni zakljucak

Projekat je iz jednostavnog SB3 eksperimenta prerastao u ozbiljniji RL sistem sa
dve grane:

- standardnom biblioteckom granom za bazicne eksperimente
- namenskim `hardcore` portom za tezu i istrazivacki zanimljiviju varijantu

Najvazniji doprinos nove verzije projekta nije samo "jos jedan skript", nego
promena same metodologije:

- observation history umesto samo jednog stanja
- sekvencijalni modeli umesto cistog feed-forward pristupa
- custom `SAC/TD3` agenti umesto samo gotovih biblioteckih wrapper-a
- checkpoint evaluacija umesto oslanjanja na poslednji model
- odvojeno pracenje `raw` i `shaped` reward-a

Na osnovu dosadasnjih rezultata moze se jasno zakljuciti da je `Hardcore` mod
znatno tezi od standardnog `BipedalWalker-v3` zadatka i da trazi specijalizovan
pristup. Upravo zato je razvijen novi custom `hardcore` port.

Trenutno najjaci rezultat u repou je `custom SAC + LSTM best_raw` checkpoint sa
spoljnom evaluacijom oko `-22.45`, sto je znacajan pomak u odnosu na legacy
`PPO` i `TD3` baseline-e, ali jos nije potpuno resenje zadatka.

Konacan odgovor na pitanje "da li je problem resen" je zato i dalje: ne u
potpunosti. Ali sada odgovor vise nije "nema napretka", nego:

- postoji jasan inzenjerski napredak
- postoji najbolji checkpoint koji je mnogo jaci od starih baseline-a
- postoji uredno spakovan skup artefakata i videa za dalju analizu

Sa metodoloske, implementacione i eksperimentalne strane, projekat je sada
znatno zreliji, jasnije strukturiran i mnogo profesionalnije pripremljen za
ozbiljan nastavak rada nego prethodna verzija.
