# Artifacts Layout

Sve generisane rezultate drzimo iskljucivo pod `artifacts/`.

## Glavne sekcije

- `runs/hardcore/`
  cuva sve bitne hardcore run-ove; checkpointovi, logovi i summary stoje direktno u run folderu
- `runs/standard/`
  cuva standard-env i benchmark run-ove u istom jednostavnom formatu
- `archive/`
  cuva sporedne dijagnostike, report bundle i smoke rezultate koje ne moras stalno da gledas

## Najbitniji run-ovi

- `runs/hardcore/fix_train_a001_as/`
  prekinuti trening do oko `ep3200`, best eval raw `67.62`
- `runs/hardcore/res_train_a001_as/`
  nastavljen warm-start run, najbolji pokazani eval raw `225.79` na `ep6400`
- `runs/hardcore/res_eval_best_raw/`
  video probe za `best_raw` checkpoint iz resumed run-a
- `runs/hardcore/preview_ep4600/`
  ciljane 5-episode video probe za `ep4600` checkpoint

`historical_videos/` je namerno ostavljen van ove strukture i nije diran.
