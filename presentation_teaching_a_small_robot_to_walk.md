0. Cilj: Želeo bih da obradim projekat iz učenja podsticanjem na "Gymnasium BipedalWalker-v3" okruženju. Rezultate svog agenta planiram da uporedim sa benchmark rezultatima nas ledećem websajtu - (https://huggingface.co/datasets/huggingface-projects/drlc-leaderboard-data/viewer), a kao referentni rad bih koristio publikaciju “Teaching a Robot to Walk Using Reinforcement Learning” (https://arxiv.org/pdf/2112.07031v1), koja je rađena nad istim okruženjem.
Cilj projekta je replikacija objavljenih rezultata i poređenje performansi različitih RL algoritama (npr. ARS, PPO, SAC), uz analizu okruženja, metrika i ponašanja agenta.

1.	Uvod
RL, locomotion, zašto je walking control težak problem.
2.	Okruženje
BipedalWalker-v3: state, action, reward, termination, normal vs hardcore. 
3.	Povezani radovi
Glavni rad 2021, zatim kratko 2024 broader locomotion paper. DQN vs ARS iz 2021. 
4.	Metodologija
PPO, SAC, TD3, random baseline; isti protocol; evaluation setup.
5.	Eksperimenti
learning curves, final mean/std, qualitative rollout analysis.
6.	Hardmode
teren ima prepreke — ladders, stumps, pitfalls — dakle stepenaste delove, panjeve/prepreke i rupe, pa je hodanje mnogo teže.
7.	Poređenje sa benchmark-om
tvoj rezultat vs HF leaderboard. 
8.	Diskusija
zašto je neki algoritam bio stabilniji, sample efficiency, failure cases.
9.	Zaključak
šta si dobio, ograničenja, future work.

