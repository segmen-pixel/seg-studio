# The bootstrap loop

Taking a project from nothing to a trained model that drafts the rest. Every
step is a tool; the loop is the order.

```
project_create                 → a project id
image_upload (× N)             → images in it
classes_set                    → the class list, if the default is wrong
[first annotations]            → a person, or the counting playbook if the
                                 object repeats and one teacher exists
dataset_set_split              → hold out the hard ones as test
train_start                    → a run
train_run_get / run_metrics_get→ wait for status: completed
predict_operating_points       → the recall-first threshold and min_area
predict_verdicts               → which images the run gets wrong
[fix the worst]                → mask_put on those; replacing a person's
                                 mask or clean mark needs overwrite and
                                 --policy full
prelabel_run                   → draft every unannotated image from the run
                                 (it copies aside what it replaces)
mask_stats                     → which drafts fall outside the hand masks'
                                 range: the person looks at those first
dataset_set_split / train_start→ again, with more annotations
```

## Rules the loop depends on

- **Splits are the experiment.** Changing them makes every earlier run's
  scores incomparable. `run_splits` says what a finished run actually trained
  on; read it before comparing two runs.
- **Resume, do not restart.** `predict_status` says which images already carry
  a prediction; `annotation_status` which carry a mask. Skip those.
- **Never write an empty mask for "I could not tell".** An all-background mask
  is a labelled negative. `mark_clean` is the tool for a genuinely clean image
  and it is a deliberate statement.
- **Check the class list did not drift.** `classes_reconcile_check` finds
  masks painted with ids the class list does not have -- written by
  `mask_put`, which takes any id, or left by a class list changed some other
  way. `classes_set` cannot cause it: the trainer refuses a list that
  drops or renumbers an id. `classes_reconcile_fix` adds placeholders, never
  removes.
- **Stop on a bad number.** If `predict_verdicts` or a hand check comes back
  far below what the other playbooks say to expect, report that instead
  of writing more masks. A wrong mask costs a person more than a missing one.
- **Say which images need eyes.** When your own check fails on one image --
  four objects kept where the teachers show five, every proposal rejected,
  the model naming more than you painted -- `mark_review` it with the reason.
  The list shows the flag; a save from the browser clears it.

## What to report at the end

First, the images that need a person's eyes: every one you passed to
`mark_review`, with its reason, written as a request to look at them. The
badge in the list marks them too, but the person reads your answer before
they open the project. Then the project id, how many images were labelled
by which method, the IoU or verdict summary against whatever ground truth
existed, the images left unlabelled and why, and any data problem found
(duplicate index entries, unfinished teacher masks, images the index lists
but the API cannot serve).
