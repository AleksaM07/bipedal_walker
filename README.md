# Bipedal Walker v3

Ovo je mali reinforcement learning projekat u kome treniramo agenta da hoda u okruzenju `BipedalWalker-v3`.

- okruzenje pokazuje robotu trenutno stanje
- model bira akciju za motore
- okruzenje vrati reward i sledece stanje
- to se ponavlja mnogo puta
- cilj je da model nauci da pravi akcije koje vode ka boljem ukupnom reward-u

Drugim recima: pravimo "mozak" koji vremenom uci kako da ne padne odmah i kako da se krece sto bolje.

## Sta je ovde najbitnije

Ako hoces da koristis projekat bez previse teorije, bitni su ti uglavnom ovi fajlovi:

- `setup_env.ps1`
- `train_bipedal_walker.py`
- `train_bipedal_hardcore.py`
- `train_bipedal_hardcore_port.py`
- `bipedal_workflow.py`
- `hardcore_port/`
- `requirements.txt`

Najbitniji tok rada je:

`train_bipedal_walker.py` -> izabrani algoritam (`ppo` / `sac` / `td3`) -> `bipedal_workflow.py`

To znaci:

- `train_bipedal_walker.py` cita argumente iz terminala
- bira koji algoritam hoces da koristis
- `bipedal_workflow.py` odradi pravi posao: trening, cuvanje, evaluaciju, random baseline i opcionalno video

Za namenski `hardcore` rad sada postoji i poseban tok:

`train_bipedal_hardcore_port.py` -> `hardcore_port/` -> custom `SAC` / `TD3` + `LSTM` / `Transformer`

## Kako pokrenuti projekat

Prvo napravi virtuelno okruzenje i instaliraj pakete:

```powershell
./setup_env.ps1
```

Posle toga mozes da pokrenes trening, na primer:

```powershell
.\.venv\Scripts\python.exe train_bipedal_walker.py --algo ppo --timesteps 50000 --eval-episodes 5
```

Ako hoces kompletan trening i video:

```powershell
.\.venv\Scripts\python.exe train_bipedal_walker.py --algo sac --timesteps 300000 --record-video --video-episodes 1
```

Ako hoces brzi eksperiment i kraci feedback loop:

```powershell
.\.venv\Scripts\python.exe train_bipedal_walker.py --algo ppo --timesteps 300000 --train-envs 4 --eval-episodes 3 --skip-random-baseline
```

Ako hoces namenski `hardcore` port sa custom agentom i sekvencijalnim modelom:

```powershell
.\.venv\Scripts\python.exe train_bipedal_hardcore_port.py --algo sac --backbone lstm
```

Ako hoces `TD3 + LSTM` verziju:

```powershell
.\.venv\Scripts\python.exe train_bipedal_hardcore_port.py --algo td3 --backbone lstm
```

Ako hoces `SAC + Transformer` verziju:

```powershell
.\.venv\Scripts\python.exe train_bipedal_hardcore_port.py --algo sac --backbone transformer
```

Ako hoces evaluaciju najboljeg checkpoint-a na 100 epizoda:

```powershell
.\.venv\Scripts\python.exe train_bipedal_hardcore_port.py --mode test-100 --algo sac --backbone lstm --checkpoint best_raw
```

Napomene za brzinu:

- `--train-envs` najvise pomaze PPO-u, jer moze da skuplja iskustvo iz vise env-ova
- `--skip-random-baseline` skracuje kraj eksperimenta kada ti baseline trenutno nije bitan
- manje `--eval-episodes` znaci brzu evaluaciju
- bez `--record-video` ceo tok je jos brzi
- u nekim ogranicenim Windows okruzenjima kod moze automatski da predje sa `SubprocVecEnv` na `DummyVecEnv`
- `--hardcore` sada koristi pravi Gymnasium env `BipedalWalkerHardcore-v3`, a ne samo flag nad obicnim env-om

Video ce biti sacuvan pod:

```text
artifacts/videos/
```

Kada je `--video-episodes 1`, skripta sada po default-u snima:

- najbolju evaluacionu epizodu
- najgoru evaluacionu epizodu

Ako stavis na primer `--video-episodes 20`, onda pored `best` i `worst`
snima jos 20 dodatnih epizoda radi sireg pregleda ponasanja agenta.

Model ce biti sacuvan pod:

```text
artifacts/models/
```

## Sta dobijas kao izlaz

Skripta sada stampa kratak, citljiv tekstualni rezime u terminalu, a puni
JSON summary cuva u:

```text
artifacts/summaries/
```

U summary-ju su najbitnije stvari:

- `saved_model_path`
  putanja do sacuvanog modela
- `eval_mean_reward`
  prosecni reward preko evaluacionih epizoda
- `eval_std_reward`
  koliko su rezultati bili nestabilni
- `eval_rewards`
  reward po svakoj epizodi
- `eval_episode_lengths`
  koliko je trajala svaka epizoda
- `best_eval_episode`
  najbolja pojedinacna evaluaciona epizoda sa reward-om, duzinom i seed-om
- `worst_eval_episode`
  najgora pojedinacna evaluaciona epizoda sa reward-om, duzinom i seed-om
- `random_baseline`
  rezultat potpuno random agenta
- `beats_random_baseline`
  da li je istrenirani model bolji od random igranja
- `improvement_vs_random`
  za koliko je model bolji ili gori od random baseline-a
- `diagnostics`
  kratke napomene ako izgleda da je politika "stuck" ili neuverljiva
- `videos`
  odvojene informacije za `best`, `worst` i dodatne snimke
- `video_files`
  objedinjena lista svih snimljenih videa ako je video bio trazen
- `video_error`
  poruka o gresci ako video nije uspeo

## Report bundle

Za pregledniji lokalni paket rezultata postoji i curator skripta:

```powershell
.\.venv\Scripts\python.exe scripts\curate_report_runs.py
```

Ona pakuje najbitnije eksperimente u:

```text
artifacts/report_runs/
```

Svaki report folder dobija:

- `policy/`
- `videos/`
- `summary/`
- `analysis.md`

To je zgodno kada hoces da za fakultetski izvestaj ili prezentaciju imas
krace, uredne foldere sa jednim checkpoint-om, reprezentativnim videom i
kratkim zakljuckom sta je eksperiment zapravo pokazao.

## Objasnjenje osnovnih RL pojmova

### Okruzenje

Okruzenje je simulacija u kojoj agent "zivi".

Ovde je to `BipedalWalker-v3`, znaci:

- imas robota sa dve noge
- robot dobija informacije o stanju tela i terena
- robot salje akcije motorima
- dobija reward u zavisnosti od toga kako se ponasa

### Observation

Observation je ono sto agent "vidi".

To nije slika kao kod coveka, nego niz brojeva. Ti brojevi opisuju trenutno stanje, na primer:

- ugao tela
- brzinu
- kontakt noge sa podom
- jos razne interne informacije iz simulacije

Model ne zna nista osim toga. On nema "zdrav razum". Zna samo te brojeve.

### Action

Action je ono sto model salje okruzenju.

U ovom projektu akcija je niz brojeva koji upravljaju motorima. Uprosticeno:

- pozitivan broj = guraj malo u jednom smeru
- negativan broj = guraj malo u drugom smeru
- sve to zajedno pokrece noge

### Reward

Reward je broj koji govori koliko je poslednji potez bio dobar ili los.

Uprosticeno:

- dobar potez -> veci reward
- los potez -> manji reward
- padanje i gluposti -> lose
- smisleno kretanje -> bolje

Bitna stvar: model ne "uci da hoda" direktno.
Model uci da pravi akcije koje vode ka boljem ukupnom reward-u.

### Episoda

Episoda je jedno potpuno igranje od pocetka do kraja.

Na primer:

- env se resetuje
- agent igra dok ne padne ili dok epizoda ne istekne
- to je jedna epizoda

### Timestep

Timestep je jedan korak:

- procitaj observation
- izaberi action
- env vrati reward i sledece stanje

Kada kazemo `50000 timesteps`, to znaci 50000 takvih malih koraka.

### Policy

Policy je pravilo koje kaze:

"Za ovo stanje, koju akciju da izaberem?"

U praksi, ovde je policy neuronska mreza.

### Actor

Actor je deo modela koji bira akciju.

Najprostije:

- observation udje u mrezu
- mreza vrati sta misli da treba uraditi

### Critic

Critic ne bira akciju direktno.

Njegov posao je da proceni:

- koliko je neko stanje dobro
- ili koliko je neki par stanje-akcija dobar

Zato ga mozes zamisliti kao "unutrasnjeg sudiju" koji pomaze actor-u da nauci.

### Deterministicki i stohasticki model

Deterministicki model:

- za isto stanje uglavnom bira istu akciju

Stohasticki model:

- za isto stanje bira akciju iz raspodele
- znaci ima malo kontrolisane nasumicnosti

To je bitno jer:

- PPO i SAC su stohasticki
- TD3 je deterministicki

## Sta radi svaki fajl

### `setup_env.ps1`

Ovo je PowerShell skripta za setup.

Radi tri proste stvari:

1. pravi `.venv` ako ne postoji
2. instalira pakete iz `requirements.txt`
3. kaze ti kako da aktiviras okruzenje

### `requirements.txt`

Ovde su zavisnosti projekta.

Najbitnije su:

- `gymnasium[box2d]`
  daje samo okruzenje `BipedalWalker-v3`
- `stable-baselines3`
  gotove, proverene RL implementacije
- `torch`
  neuronske mreze i tensor racun
- `numpy`
  rad sa nizovima brojeva
- `moviepy`
  potrebno za video snimanje

### `train_bipedal_walker.py`

Ovo je glavni "entry point" projekta.

Njegov posao je jednostavan:

1. procita argumente iz terminala
2. vidi da li hoces `ppo`, `sac` ili `td3`
3. napravi putanju za model i eventualni video
4. pozove odgovarajucu funkciju za trening
5. odstampa pregledan rezime i sacuva puni JSON summary na disk

To znaci da korisnik najcesce ne mora da dira ostale fajlove da bi pokrenuo projekat.

### `bipedal_workflow.py`

Ovo je sada centralni radni fajl projekta.

U njemu su skupljene sve bitne helper funkcije:

- pravljenje okruzenja
- trening preko Stable-Baselines3
- evaluacija modela
- random baseline
- snimanje videa
- tanki wrapper-i za `ppo`, `sac` i `td3`

Najvaznija prakticna poenta:

- vise nema rasutih malih fajlova za svaki algoritam
- sav glavni posao je na jednom mestu
- ako hoces da razumes sta projekat radi iza CLI-ja, gledas `bipedal_workflow.py`

Glavni tok u tom fajlu je i dalje vrlo jednostavan:

1. napravi env
2. napravi SB3 model
3. pokrene trening
4. sacuva model
5. evaluira model
6. pokrene random baseline
7. uporedi model sa random igranjem
8. po potrebi snimi video

## Kako projekat stvarno radi korak po korak

Ako pokrenes:

```powershell
.\.venv\Scripts\python.exe train_bipedal_walker.py --algo sac --timesteps 300000 --eval-episodes 5 --record-video
```

desava se ovo:

1. `train_bipedal_walker.py` procita argumente
2. vidi da si izabrao `sac`
3. pozove SAC putanju iz `bipedal_workflow.py`
4. `bipedal_workflow.py` napravi env
5. po potrebi napravi vise trening env-ova za PPO
6. napravi SB3 SAC model
7. model trenira zadati broj koraka
8. model se sacuva na disk
9. model se evaluira kroz nekoliko epizoda
10. random baseline se ili pokrene ili preskoci ako je trazeno
11. ako je trazen video, snime se `best` i `worst` evaluaciona epizoda
12. ako je `video-episodes > 1`, snime se i dodatne epizode
13. sacuva se puni JSON summary i odstampa kratak rezime

To je ceo projekat u praksi.

## Teorijski deo: sta predstavljaju PPO, SAC i TD3

### PPO

PPO znaci `Proximal Policy Optimization`.

Najgluplje receno:

- model igra neko vreme
- zapamti sta je radio
- pogleda sta je bilo dobro a sta lose
- popravi sebe malo
- ali ne dozvoli sebi da se previse naglo promeni

To "nemoj da se promenis previse naglo" je cela poenta PPO-a.

Zasto je to bitno?

Zato sto RL modeli lako polude.
Ako im dozvolis da se prebrzo promene:

- danas nauce nesto korisno
- sledeci update sve pokvari
- trening postane nestabilan

PPO zato koristi "clipping" ideju.

Uprosticeno:

- stara politika je govorila da je akcija bila ovako verovatna
- nova politika sada misli drugacije
- PPO proveri koliko se to promenilo
- ako je promena prevelika, kaze: "ne, uspori malo"

To je razlog zasto je PPO cesto zahvalan za pocetak:

- relativno je stabilan
- nije najjednostavniji matematikom, ali je dosta popularan
- cesto dobro radi kao prvi izbor

#### Kako PPO radi u ovom projektu

U ovom projektu PPO prakticno ide ovako:

1. `train_bipedal_walker.py` procita da je izabran `ppo`
2. PPO putanja iz `bipedal_workflow.py` prosledi parametre u zajednicki workflow
3. `bipedal_workflow.py` napravi SB3 PPO model
4. SB3 odradi trening svojom internom PPO implementacijom
5. posle toga se urade evaluacija, random baseline i opcionalno video

#### Sta je GAE u prostom jeziku

GAE je nacin da dobijes bolju procenu:

"Koliko je ovo sto sam uradio bilo korisno?"

Bez GAE bi taj signal cesto bio previse bucan.
Sa GAE dobijas nesto smireniji i korisniji signal za ucenje.

### SAC

SAC znaci `Soft Actor-Critic`.

Najprostije objasnjenje:

- SAC ne zeli samo dobru akciju
- SAC zeli i da agent dovoljno istrazuje

Zato u SAC-u postoji entropijski deo.

- nije dovoljno da agent uvek radi samo jednu stvar
- korisno je da bude malo raznovrstan dok jos uci

SAC zato voli stohasticku politiku.

To znaci:

- actor ne vraca samo "uradi ovo"
- actor vraca raspodelu akcija
- iz te raspodele se uzorkuje konkretna akcija

To SAC-u pomaze kod continuous control problema kao sto je BipedalWalker.

#### Zasto SAC ima dva critic-a

Zato sto Q procene umeju da budu previse optimisticne.

Uprosticeno:

- jedan critic kaze "ovo je super"
- drugi kaze "nije bas toliko super"
- uzmemo manju procenu

To pomaze stabilnosti.

#### Sta su target mreze

Target mreze su sporije kopije glavnih mreza.

Njihova poenta je:

- da meta za ucenje ne skace prebrzo
- da trening bude manje haotican

Zato postoji `soft_update`.

To znaci:

- ne kopiramo sve naglo
- nego target pomeramo malo po malo

### TD3

TD3 znaci `Twin Delayed Deep Deterministic Policy Gradient`.

Ime zvuci strasno, ali glavne ideje su samo tri.

#### 1. Twin critics

Ima dva critic-a.

Poenta:

- ako jedan pretera i kaze da je akcija predobra
- drugi ga moze "spustiti"
- uzima se manja procena

#### 2. Delayed actor updates

Actor se ne updejtuje stalno kao critic.

Poenta:

- prvo critic treba da postane koliko-toliko pametan
- tek onda ima smisla da actor mnogo slusa njegove savete

To ume da stabilizuje trening.

#### 3. Target policy smoothing

Na target akciju se dodaje malo buke.

Poenta:

- da politika ne postane previse osetljiva
- da ne "overfituje" na uske i krhke akcije

#### Zasto je TD3 deterministicki

Za razliku od SAC-a:

- SAC bira akciju iz raspodele
- TD3 uglavnom bira jednu direktnu akciju

To moze biti efikasno, ali trazi svoje stabilizacione trikove, zato TD3 ima
gore pomenute dodatke.

## Razlika izmedju PPO, SAC i TD3

| Algorithm | Characteristics                                                                                                                           |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| PPO       | - voli rollout pristup<br>- radi update iz skupljenih sekvenci iskustva<br>- poznat po stabilnosti i popularnosti<br>- dobar kao pocetna referenca |
| SAC       | - stohasticki<br>- voli istrazivanje<br>- vrlo jak za continuous action probleme<br>- cesto dobar izbor za ovakav zadatak                         |
| TD3       | - deterministicki<br>- koristi dva critic-a i target smoothing<br>- fokusiran na stabilniji deterministic actor-critic pristup                 |

## Kako da tumacis rezultate

### Ako je reward jako negativan

To obicno znaci:

- model pada brzo
- ne zna da koordinise noge
- ponasanje je lose

### Ako je video vrlo kratak i robot odmah padne

To obicno znaci:

- trening nije bio dovoljan
- model jos nije naucio nista korisno

### Ako model ne pobedi random baseline

To je ozbiljan znak da:

- nije naucio dovoljno
- ili nesto nije dobro podeseno

### Ako model jedva pobedi random baseline

To znaci:

- nesto je naucio
- ali je jos slab

### Ako je dosta bolji od random baseline-a

To znaci:

- trening ima smisla
- model je poceo da uci korisno ponasanje

## Zasto video nekad izgleda smesno

Zato sto je BipedalWalker dosta tezak problem.

Model ne razmislja kao covek.
On ne zna:

- sta je noga
- sta je ravnoteza
- kako se hoda

On samo:

- vidi brojeve
- salje brojeve
- dobija reward

Iz te tri stvari mora sam da "provali" korisno ponasanje.

Zato su pocetni video snimci cesto:

- jedan sekund
- pad odmah
- nasumicno cimanje

## Namenski Hardcore Port

Pored standardnog SB3 toka, projekat sada ima i poseban `hardcore` port koji je
pravljen kao vernija inspiracija uspesnim radovima za `BipedalWalkerHardcore-v3`.

Najbitnija ideja je sledeca:

- obican `SB3` trening sa jednom trenutnom opservacijom cesto nije dovoljan
- `hardcore` staza je parcijalno opservabilna i osetljiva na lose istrazivanje
- zato uvodimo kratku istoriju opservacija i sekvencijalne modele
- pored toga menjamo i trening protokol: `checkpoint` evaluacije, `best model`
  cuvanje i odvojeno pracenje `raw` i `shaped` reward-a

### Novi fajlovi za hardcore port

- `train_bipedal_hardcore_port.py`
  glavni CLI za custom port
- `hardcore_port/envs.py`
  wrapper-i za `frame skip`, `fall penalty` i observation history
- `hardcore_port/models.py`
  `LSTM` i `Transformer` actor-critic modeli
- `hardcore_port/agents.py`
  custom implementacije `SAC` i `TD3`
- `hardcore_port/training.py`
  episode-based trening, evaluacija i checkpoint logika
- `hardcore_port/replay.py`
  replay buffer za off-policy algoritme
- `hardcore_port/noise.py`
  exploration noise za `TD3`

### Zasto je hardcore poseban problem

Kod obicnog `BipedalWalker-v3` cesto je dovoljno da model vidi samo trenutno
stanje `s_t`. Kod `Hardcore` verzije to cesto nije dovoljno zato sto:

- prepreke dolaze u sekvenci
- jedan trenutni observation ne govori uvek dovoljno o dinamici tela
- agentu treba informacija o tome kako se stanje menja kroz vreme

Drugim recima, problem mozemo posmatrati kao parcijalno opservabilan. Umesto da
model koristi samo jedno stanje `s_t`, uvodimo istoriju:

\[
H_t = [o_{t-h+1}, o_{t-h+2}, \ldots, o_t]
\]

gde je:

- `o_t` trenutna observation opservacija
- `h` duzina istorije

U kodu su podrazumevane vrednosti:

- `SAC`: `history_length = 12`
- `TD3`: `history_length = 6`

To je namerno, jer je `SAC + LSTM + 12` najperspektivnija grana za prvi ozbiljan
pokusaj.

### Frame Skip i reward shaping

U `hardcore_port/envs.py` isti action se izvrsava vise puta, po default-u `2`.
Ako oznacimo broj ponavljanja kao `k`, onda wrapper korak koristi:

\[
a_t = a_{t,0} = a_{t,1} = \cdots = a_{t,k-1}
\]

To smanjuje efektivnu frekvenciju kontrole i cesto olaksava ucenje stabilnog
hoda.

Pored toga, terminalna kazna za pad se menja. Ako je robot pao:

\[
\bar{r}_t =
\begin{cases}
-10, & \text{ako je } dead_t = 1 \\
r_t, & \text{inace}
\end{cases}
\]

Zatim se u wrapper koraku sabiraju reward-i kroz `frame skip`:

\[
\tilde{r}_t = \sum_{j=0}^{k-1} \bar{r}_{t,j}
\]

U projektu se zato prate dve metrike:

- `raw reward`
  originalni reward iz env-a
- `shaped reward`
  reward koji stvarno koristi custom trening

To je vazno zato sto agent moze da napreduje po shaping metrici, a da se to jos
ne vidi dovoljno jasno na originalnom reward-u.

### Sekvencijalni modeli: LSTM i Transformer

U custom portu svaki agent ima actor i critic mrezu sa sekvencijalnim
encoder-om.

#### LSTM encoder

Za ulaznu sekvencu `x_1, x_2, ..., x_T`, `LSTM` racuna:

\[
(h_t, c_t) = LSTM(x_t, h_{t-1}, c_{t-1})
\]

a kao reprezentaciju istorije koristi se poslednje skriveno stanje:

\[
z_T = h_T
\]

Prakticna poenta:

- `LSTM` moze da zapamti kratkorocne i srednjerocne obrasce kretanja
- to je korisno kada agent treba da uskladi zamah, kontakt sa tlom i nagib tela

#### Transformer encoder

Kod `Transformer` varijante sekvenca se prvo embeduje i dobija poziciono
kodiranje:

\[
e_t = W_{emb} x_t + p_t
\]

Zatim se koristi self-attention:

\[
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

U ovoj implementaciji naglasak je na "last-step" citanju, tj. poslednji element
sekvence pita ostatak istorije:

\[
z_T = Attention(q_T, K, V)
\]

Prakticna poenta:

- `Transformer` moze da "pogleda unazad" i proceni koji delovi istorije su
  trenutno najvazniji
- to je alternativa `LSTM` pristupu za isti problem parcijalne opservabilnosti

### Actor-critic struktura u custom portu

Posle sekvencijalnog encoder-a i actor i critic dobijaju kompaktnu
reprezentaciju istorije `z_t`.

Critic aproksimira:

\[
Q(s_t, a_t)
\]

a actor pravi politiku:

\[
a_t = \mu_\theta(s_t)
\]

za `TD3`, odnosno stohasticku politiku:

\[
a_t \sim \pi_\theta(\cdot \mid s_t)
\]

za `SAC`.

## Teorija Custom Hardcore Porta

### Replay buffer

I `SAC` i `TD3` su off-policy algoritmi, pa koriste replay buffer:

\[
\mathcal{D} = \{(s_t, a_t, r_t, s_{t+1}, d_t)\}
\]

Iz buffer-a se nasumicno uzorkuje mini-batch:

\[
(s, a, r, s', d) \sim U(\mathcal{D})
\]

Ovo smanjuje korelaciju uzastopnih primera i poboljsava stabilnost treninga.

### SAC loss funkcije

`SAC` u custom portu koristi dva critic-a i stohasticku politiku.

Target za critic je:

\[
y_t = r_t + \gamma (1-d_t)\left(\min(Q_1'(s_{t+1}, a_{t+1}), Q_2'(s_{t+1}, a_{t+1})) - \alpha \log \pi(a_{t+1}\mid s_{t+1})\right)
\]

gde je:

\[
a_{t+1} \sim \pi(\cdot \mid s_{t+1})
\]

Critic loss je:

\[
\mathcal{L}_{Q_i} = \mathbb{E}\left[(Q_i(s_t, a_t) - y_t)^2\right]
\]

Actor loss je:

\[
\mathcal{L}_{\pi} = \mathbb{E}\left[\alpha \log \pi(a_t\mid s_t) - \min(Q_1(s_t, a_t), Q_2(s_t, a_t))\right]
\]

Ovde `alpha` kontrolise kompromis:

- veci `alpha` vise podstice istrazivanje
- manji `alpha` vise forsira direktnu optimizaciju reward-a

### TD3 loss funkcije

`TD3` koristi deterministickog actor-a, dva critic-a i delayed update.

Target akcija je:

\[
a_{t+1}' = clip(\mu_{\theta'}(s_{t+1}) + \epsilon,\; a_{min}, a_{max})
\]

gde je:

\[
\epsilon \sim clip(\mathcal{N}(0, \sigma^2), -c, c)
\]

Target za critic je:

\[
y_t = r_t + \gamma (1-d_t)\min(Q_1'(s_{t+1}, a_{t+1}'), Q_2'(s_{t+1}, a_{t+1}'))
\]

Critic loss je:

\[
\mathcal{L}_{Q_i} = \mathbb{E}\left[(Q_i(s_t, a_t) - y_t)^2\right]
\]

Actor se ne menja pri svakom koraku nego na svakih `d` critic update-a:

\[
\mathcal{L}_{\mu} = -\mathbb{E}[Q_1(s_t, \mu(s_t))]
\]

To je poenta `delayed policy updates`: prvo stabilizuj critic, pa tek onda
agresivnije pomeraj actor.

### Soft update target mreza

I u `SAC` i u `TD3` target mreze se ne kopiraju naglo, nego meko:

\[
\theta' \leftarrow \tau \theta + (1-\tau)\theta'
\]

Ovo smanjuje oscilacije target vrednosti i cini ucenje stabilnijim.

### Episode-based trening protokol

Custom port ne radi trening samo po broju `timesteps`, nego po epizodama.
Jedna epizoda ide:

1. reset env-a
2. rollout do kraja epizode ili do `max_steps = 750`
3. snimanje svih tranzicija u replay buffer
4. posle epizode se radi onoliko gradient koraka koliko je epizoda imala koraka

Ako epizoda ima `T` wrapper koraka, onda se radi:

\[
N_{update} = T
\]

Ovaj protokol je blizi referentnim `hardcore` eksperimentima nego klasicni SB3
pipeline.

### Checkpoint evaluacija

Jedna od najvaznijih dodatih stvari je periodicna evaluacija i cuvanje vise vrsta
checkpoint-a:

- `epN.pt`
  checkpoint na svakoj evaluaciji
- `best_raw.pt`
  checkpoint sa najboljim prosecnim originalnim reward-om
- `best_shaped.pt`
  checkpoint sa najboljim prosecnim shaped reward-om
- `last.pt`
  poslednji checkpoint treninga

Ako oznacimo prosecan raw reward pri evaluaciji kao:

\[
\overline{G}_{raw}^{(N)} = \frac{1}{N}\sum_{i=1}^{N} G_{raw}^{(i)}
\]

onda se `best_raw` menja kada vazi:

\[
\overline{G}_{raw}^{(N)} > \overline{G}_{raw,best}
\]

Analogno tome, za shaped reward:

\[
\overline{G}_{shaped}^{(N)} = \frac{1}{N}\sum_{i=1}^{N} G_{shaped}^{(i)}
\]

Checkpoint evaluacija je veoma vazna zato sto kod `hardcore` zadataka poslednji
model cesto nije i najbolji model.

### Prakticni podrazumevani rezim

Ako se pokrene:

```powershell
.\.venv\Scripts\python.exe train_bipedal_hardcore_port.py --algo sac --backbone lstm
```

onda se koristi:

- `env_id = BipedalWalkerHardcore-v3`
- `history_length = 12`
- `frame_skip = 2`
- `fall_penalty = -10`
- `episodes = 8000`
- `explore_episodes = 50`
- `eval_frequency = 200`
- `eval_episodes = 20`
- `final_eval_episodes = 100`
- `max_steps = 750`
- `lr = 4e-4`
- `batch_size = 64`
- `gamma = 0.98`
- `tau = 0.01`
- `alpha = 0.01`

To je trenutno najjaci "prvi pokusaj" u projektu za resavanje `hardcore`
varijante.

### Kako da tumacis rezultate custom porta

Ako vidis sledece:

- `rolling_mean_shaped_reward` raste, ali `raw` stagnira
  model uci po shaping signalu, ali jos nije dovoljno dobar na originalnom env
- `best_raw` checkpoint je znatno bolji od `last`
  trening je nestabilan i checkpoint selekcija je vazna
- `SAC + LSTM` raste stabilnije od `TD3 + LSTM`
  entropijska regularizacija verovatno pomaze na ovom sparse i teskom zadatku
- `Transformer` uci sporije od `LSTM`
  moguce je da mu treba vise podataka ili duzi trening

Upravo zato projekt sada cuva i `raw` i `shaped` metrike, umesto da gleda samo
jedan broj.
