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
- `bipedal_workflow.py`
- `requirements.txt`

Najbitniji tok rada je:

`train_bipedal_walker.py` -> izabrani algoritam (`ppo` / `sac` / `td3`) -> `bipedal_workflow.py`

To znaci:

- `train_bipedal_walker.py` cita argumente iz terminala
- bira koji algoritam hoces da koristis
- `bipedal_workflow.py` odradi pravi posao: trening, cuvanje, evaluaciju, random baseline i opcionalno video

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

Video ce biti sacuvan pod:

```text
artifacts/videos/
```

Model ce biti sacuvan pod:

```text
artifacts/models/
```

## Sta dobijas kao izlaz

Skripta stampa JSON summary. U njemu su najbitnije stvari:

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
- `random_baseline`
  rezultat potpuno random agenta
- `beats_random_baseline`
  da li je istrenirani model bolji od random igranja
- `improvement_vs_random`
  za koliko je model bolji ili gori od random baseline-a
- `video_files`
  lista snimljenih videa ako je video bio trazen
- `video_error`
  poruka o gresci ako video nije uspeo

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
5. odstampa summary kao JSON

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
5. napravi SB3 SAC model
6. model trenira zadati broj koraka
7. model se sacuva na disk
8. model se evaluira kroz nekoliko epizoda
9. pokrene se random baseline radi poredjenja
10. snimi se video ako je trazen
11. sve se vrati u jednom JSON summary-ju

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
