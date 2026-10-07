Private review bundle: prompts contain masked trace text; keep outside Git.
No provider calls have been made. Only send to an approved model provider.
DIR below means this directory (quote paths with spaces):

atif-scan hunt --model MODEL --questions DIR --inspect-tool --jobs 8

Then rerun the SAME original scan inputs and options, replacing --questions DIR
with --answers DIR. Do not scan the whole cached job for a leaderboard-row review.
manifest.json is the local review subset, not the original scoring population.
selection.json records selected inputs and skipped/non-applicable questions.
Answers annotate evidence; they never automatically disqualify trials.
