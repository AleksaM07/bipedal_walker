# Teaching a Small Robot to Walk

## Osnovna ideja projekta

Tema projekta je ucenje podsticanjem na Gymnasium okruzenju `BipedalWalker-v3`.
Cilj je da se uporede performanse vise RL algoritama na istom zadatku hodanja,
uz analizu okruzenja, metrike uspeha i ponasanja agenta.

U ovom projektu su testirani:

- PPO
- SAC
- TD3
- random baseline

Planirano poredjenje je sa:

- Hugging Face DRLC leaderboard-om
- radom `Teaching a Robot to Walk Using Reinforcement Learning`

Najvaznija poruka koju treba da nosi prezentacija je:

`nije dovoljno da agent bude bolji od random baseline-a; bitno je i koliko je stabilan, koliko brzo uci i kako izgleda njegovo ponasanje u samoj simulaciji`

## Kako da ispricas projekat kao pricu

Najzanimljivije je da prezentaciju ne vodis kao "nabrajanje algoritama",
nego kao malu pricu:

`zelimo da naucimo malog robota da hoda`

Ta prica moze da ide ovako:

1. Na pocetku robot ne zna nista i pada ili daje nasumicne pokrete.
2. Zatim mu damo razlicite nacine ucenja: PPO, SAC i TD3.
3. Svi oni pokusavaju da rese isti problem, ali to rade na razlicit nacin.
4. Na kraju ne gledamo samo broj poena, nego i to kako robot zaista hoda.
5. Tako dolazimo do pitanja: koji algoritam ne samo da uci, nego uci stabilno i korisno?

To je dobra "nit" kroz celu prezentaciju, jer je intuitivna i lako se prati.

## Predlog za prezentaciju od 5 minuta

Za 5 minuta je najrealnije da imas 6 slajdova.
To je dovoljno da kazes sve bitno, a da ne deluje prenatrpano.

---

## Slajd 1 - Problem i motivacija

### Sta ide na slajd

- naslov projekta
- jedna slika BipedalWalker okruzenja
- jedna kratka recenica cilja

Predlog kratkog teksta na slajdu:

`Cilj: nauciti bipedalnog agenta da hoda i uporediti PPO, SAC i TD3 na istom zadatku.`

### Sta da kazes usmeno

`U ovom projektu sam se bavio problemom locomotion-a, odnosno ucenjem robota da hoda u simulaciji. Izabrao sam Gymnasium BipedalWalker-v3, jer je to poznat continuous control zadatak i postoji i literatura i benchmark sa kojim mogu da uporedim svoje rezultate.`

### Poenta slajda

Odmah objasnjavas:

- sta je problem
- zasto je zanimljiv
- zasto je dobar za projekat

---

## Slajd 2 - Okruzenje

### Sta ide na slajd

- kratko: state, action, reward, termination
- razlika izmedju normal i hardcore moda
- eventualno jedan dijagram agent -> akcija -> env -> reward

Predlog sadrzaja:

- stanje: agent dobija informacije o telu, zglobovima i kontaktu sa tlom
- akcije: kontinualne kontrole motora nogu
- reward: podstice kretanje unapred i penalizuje lose ponasanje
- kraj epizode: pad, neuspeh ili vremenski limit
- hardcore: ladders, stumps, pitfalls

### Sta da kazes usmeno

`Ovo okruzenje nije klasican klasifikacioni problem, nego problem sekvencijalnog odlucivanja. Agent u svakom koraku dobija stanje, bira akciju i dobija reward. U normalnom modu je cilj da nauci stabilno hodanje, dok je hardcore mod dosta tezi jer uvodi prepreke kao sto su rupe, prepreke i neravni delovi terena.`

### Poenta slajda

Ovde ispunjavas profesorov zahtev da objasnis:

- sta su stanja
- koje akcije postoje
- sta je nagrada
- kada se epizoda zavrsava

---

## Slajd 3 - Metodologija

### Sta ide na slajd

- PPO, SAC, TD3, random baseline
- isti protokol za sve algoritme
- trening -> evaluacija -> video analiza

Predlog sadrzaja:

- algoritmi: PPO, SAC, TD3
- baseline: random policy
- isto okruzenje i isti eval setup
- metrika: mean reward, std reward, qualitative rollout

### Sta da kazes usmeno

`Da bi poredjenje bilo posteno, sve algoritme sam testirao na istom okruzenju i sa istim evaluacionim protokolom. Pored istreniranih modela, koristio sam i random baseline da vidim koliko modeli zaista uce, a ne samo da slucajno postizu rezultat. Osim numerickih metrika, snimao sam i video epizode da bih video kako se agent stvarno ponasa.`

### Poenta slajda

Naglasavas:

- fer poredjenje
- vise modela
- i brojke i kvalitativnu analizu

---

## Slajd 4 - Rezultati

### Sta ide na slajd

Najbolje je da ovde imas tabelu ili bar chart.

Predlog tabele:

| Algoritam | Mean reward | Std | Random baseline | Zakljucak |
|-----------|-------------|-----|-----------------|-----------|
| PPO | 125.28 | 131.29 | oko -99 | uci, ali vrlo nestabilno |
| SAC | 282.08 | 0.91 | oko -105 | vrlo stabilan i jak |
| TD3 | 299.54 | 0.46 | oko -105 | najbolji rezultat |

Napomena:

- rezultati su iz normalnog `BipedalWalker-v3`
- evaluacija je na 5 epizoda
- random baseline je jak negativan i pokazuje koliko je zadatak tezak bez ucenja

### Sta da kazes usmeno

`Glavni rezultat je da su sva tri algoritma bolja od random baseline-a, ali nisu jednako dobra. PPO jeste naucio nesto, ali ima veoma veliku varijansu i daje i dobre i lose epizode. SAC je mnogo stabilniji, a TD3 je dao najbolji srednji rezultat i najmanju varijansu u ovom eksperimentu.`

### Poenta slajda

Ovo je centralni slajd prezentacije.
Ako imas samo jedan grafikon ili tabelu, neka bude ovaj.

---

## Slajd 5 - Kako agent zapravo hoda

### Sta ide na slajd

- 2 ili 3 screenshot-a iz videa
- ili jedan kadar za PPO, jedan za SAC, jedan za TD3
- kratke etikete: unstable, stable, best

Predlog poruke:

- PPO: ume da napravi napredak, ali cesto osciluje ili pada
- SAC: kretanje je glatko i pouzdano
- TD3: najkonzistentnije i najefikasnije hodanje

Ako hoces da ukljucis i failure case:

- pokazi i jedan los PPO ili hardcore video

### Sta da kazes usmeno

`Ovo je vazno jer reward sam po sebi nije dovoljan. Kod PPO sam video da postoje i vrlo dobre i vrlo lose epizode, sto znaci da politika nije stabilna. Kod SAC i TD3 ponasanje je mnogo konzistentnije, a TD3 je dao i najbolji krajnji rezultat.`

### Poenta slajda

Ovde pokazujes da nisi samo "istrcao kod", nego da si analizirao ponasanje modela.

---

## Slajd 6 - Zakljucak, ogranicenja i dalji rad

### Sta ide na slajd

- 3 kratke poruke
- 2 ogranicenja
- 2 ideje za dalje

Predlog:

Zakljucak:

- TD3 je dao najbolji rezultat na normalnom modu
- SAC je bio gotovo jednako dobar i vrlo stabilan
- PPO je radio, ali uz veliku nestabilnost

Ogranicenja:

- jos nema eksplicitnog poredjenja sa konkretnim brojkama iz rada i leaderboard-a
- nema vise seed-ova za jacu statisticku tvrdnju
- ARS nije implementiran

Dalji rad:

- vise seed-ova i learning curves
- hardcore mod sa jacim hiperparametrima
- dodavanje ARS i direktno poredjenje sa literaturom

### Sta da kazes usmeno

`Zakljucak je da se na ovom zadatku vidi jasna razlika izmedju algoritama: nije vazno samo da agent nauci, nego i koliko stabilno uci. U mojim eksperimentima TD3 i SAC su bili znatno bolji od PPO-a. Kao sledeci korak, bilo bi vazno dodati vise seed-ova, ubaciti ARS i napraviti eksplicitno poredjenje sa referentnim radom i benchmark rezultatima.`

---

## Kratka verzija price za usmeno izlaganje

Ako zelis da zvuci kao prica, mozes ovako:

`Zeleo sam da proverim da li mogu da naucim malog robota da hoda i da pritom vidim koji RL algoritam to radi najbolje. Za to sam izabrao BipedalWalker-v3, jer je dovoljno tezak da problem bude zanimljiv, a opet postoji literatura sa kojom moze da se uporedi. Testirao sam PPO, SAC i TD3, kao i random baseline. Rezultati su pokazali da sva tri algoritma nadmasuju random agenta, ali ne na isti nacin. PPO je umeo da napravi dobar rezultat, ali je bio nestabilan. SAC je bio veoma stabilan, a TD3 je dao najbolji krajnji rezultat. To mi govori da kod locomotion problema nije vazna samo visina reward-a, nego i pouzdanost ponasanja agenta.`

To je dovoljno prirodno da ne zvuci kao citanje sa slajda.

## Kako da slajdovi budu zanimljivi

Nemoj da zatrpas slajdove tekstom.

Bolji pristup je:

- 1 jaka recenica po slajdu
- 1 slika ili 1 tabela
- usmeno objasnjenje od 20 do 40 sekundi

Za ovu temu posebno je korisno:

- screenshot okruzenja
- tabela rezultata
- 2 ili 3 frame-a iz videa
- jedan "failure" primer i jedan "success" primer

To ce delovati mnogo bolje nego 10 slajdova sa tekstom.

## Sta ti jos nedostaje da projekat bude potpuno zaokruzen

Trenutno projekat deluje dovoljno dobro za prezentaciju, ali za punu,
akademski jacu verziju jos fale tri stvari:

### 1. Eksplicitno poredjenje sa radom i benchmark-om

Ovo znaci:

- izvuci konkretne referentne brojeve iz rada
- izvuci konkretne referentne brojeve sa leaderboard-a
- napravi jednu tabelu: `moj rezultat vs literatura vs benchmark`

Ovo nije tesko za implementaciju u kodu.
Vise je stvar dopune izvestaja i slajda.

### 2. Vise seed-ova

Trenutno imas dobar osecaj kako algoritmi rade, ali sa jednim seed-om jos ne
mozes mnogo jako da tvrdis o stabilnosti.

Idealno bi bilo:

- 3 seeda po algoritmu
- prosecni rezultat preko seed-ova
- error bar ili makar mean +- std

Ovo je tehnicki lako, ali vremenski skupo zato sto zahteva vise treninga.

### 3. ARS

U uvodnoj ideji si pomenuo `ARS`, ali on jos nije u projektu.

To je jedina stvar koja je stvarno "implementaciono veca" od ove tri.
Nije nemoguce, ali je primetno vise posla od:

- dodavanja vise seed-ova
- pravljenja dodatne tabele
- dopune prezentacije i izvestaja

## Da li je tesko dopuniti ono sto fali

Najkrace:

- benchmark tabela: nije tesko
- vise seed-ova: nije tesko, ali traje dugo
- learning curves po algoritmu: srednje tesko
- ARS: srednje do teze, zavisno da li hoces svoju implementaciju ili gotovu referencu
- hardcore kao ozbiljan eksperiment: nije tesko da se pokrene, ali je racunski mnogo skuplje

## Moj iskren savet

Ako ti je cilj da projekat bude dobar za prezentaciju u razumnom roku, ja bih
prioritet stavio ovako:

1. Napravi dobar slajd sa rezultatima koje vec imas.
2. Dodaj slajd sa jednim success i jednim failure videom.
3. Napravi tabelu `moj rezultat vs literatura/benchmark`.
4. Ako stignes, pokreni jos 2 dodatna seeda za SAC i TD3.
5. ARS ostavi kao future work ako ne stignes.

Tako ces imati zaokruzen, posten i dovoljno jak projekat.

## Materijal koji vec imas i koji treba iskoristiti

Iz lokalnih rezultata trenutno imas vrlo korisne brojke za prezentaciju:

- PPO: `125.28 +- 131.29`
- SAC: `282.08 +- 0.91`
- TD3: `299.54 +- 0.46`
- random baseline: oko `-100`

To je vec sasvim dovoljno da pokazes:

- da modeli stvarno uce
- da se algoritmi razlikuju
- da stabilnost nije ista

## Reference

- Hugging Face DRLC leaderboard:
  `https://huggingface.co/datasets/huggingface-projects/drlc-leaderboard-data/viewer`
- `Teaching a Robot to Walk Using Reinforcement Learning`:
  `https://arxiv.org/pdf/2112.07031v1`
- SAC blog koji lepo objasnjava intuiciju:
  `https://adi3e08.github.io/blog/sac/`
- CleanRL docs:
  `https://docs.cleanrl.dev/`
