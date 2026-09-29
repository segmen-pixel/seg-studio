# Tutorial sample dataset

`seg-studio-tutorial.zip` — 38 macro photographs of SIM-card contact pads, with
their annotation masks. Import it and you have a working project in about a
minute, without needing images of your own.

Use it when you want to learn the tool, or when your own data is not producing a
result and you need to tell "the software is misconfigured" apart from "my
dataset is not ready yet".

## Import

Projects tab -> **Import** -> pick this ZIP. A project is created from the file
name and the images and masks come in together.

## What is in it

| | |
|---|---|
| Images | 38, 512x512, JPEG (quality 95, 4:4:4 — no chroma subsampling) |
| Masks | 38, PNG, one class (`class1`) |
| With a defect | 22 |
| Confirmed clean | 16 (mask is all background — what **Mark Clean** writes) |

The defects are small dark specks and fine scratches on the gold pads. They are
deliberately not easy: a defect a few pixels across is the case this tool exists
for, and a tutorial that only worked on obvious blobs would teach the wrong
lesson.

The 16 clean images are there on purpose too. An image with no mask at all is
*unconfirmed* — nobody has looked at it. An image marked clean has been
confirmed to contain no defect, and the model learns from it. Those are
different states, and the difference matters more than it first appears.

## Provenance and licence

Photographed by the project author on their own equipment, of their own parts.
No customer material and no third-party data. Distributed under the repository's
licence (Apache-2.0) together with the rest of the project.
