# Active report: concise standalone version

Use **`midterm_report_standalone.tex`** as the current report. Upload this one
file to Overleaf and select it as the main document. It has no external chapter,
bibliography, or figure dependencies. It contains a 265-word abstract and sections
on the project goal, completed work, remaining work, and completion plan.

The other `.tex` files, Markdown version, and ZIP below are the older detailed
thesis-format draft, retained for reference. They have not been synchronized
with the shortened version. No local LaTeX compilation or PDF verification has
been possible. University and email placeholders still need completion.

The on-track statement expresses the student's current milestone-based assessment;
it is not an independently verified guarantee against a known submission deadline.

---

## Historical instructions for the earlier detailed draft

# Midterm report source

## Files and Overleaf import

Upload `midterm_overleaf.zip` as a new Overleaf project and select
`midterm_report.tex` as the main document. The ZIP contains all required chapter
files and `references.bib`; no external figures are required. The methodology
figure is drawn with native LaTeX boxes and arrows.

The project follows the supplied `report` structure: title page,
acknowledgements, contents, abstract, Introduction, Background, Methodology,
Results, Related Work, Conclusions, Future Work, bibliography, and appendix.
Each substantive part has its own `.tex` file. The cover preserves the requested
layout, with university and email placeholders awaiting confirmation.
`midterm_report.md` is the readable text version; the LaTeX project is the
formatting source of truth.

The abstract is 283 words, within the requested 250–300-word range. It states
the research question, intended contribution, implementation, baseline findings,
and the work still needed to answer the rule-variation question. Template
examples about microfinance and NSF support are guidance, not report content.
No grant or funding source is invented. Personal acknowledgements should be
reviewed before submission.

## Compilation status

No LaTeX compiler is installed here. Input files, environment structure,
citation keys, and labels were checked statically; the project has not been
compiled or visually verified. No PDF has been produced.

The bibliography uses `biblatex` and Biber. The supplied template's separate
`bibliographystyle` command was omitted because it belongs to the BibTeX workflow.
`xcolor` is included for the template's `blue` command. If compiling locally,
run from this directory:

```sh
pdflatex midterm_report.tex
biber midterm_report
pdflatex midterm_report.tex
pdflatex midterm_report.tex
```

## Evidence and scope

`results_provenance.json` maps reported aggregates to saved experiment settings
and outcomes. The 20-rollout guided aggregate uses 53 wins in 80 retained games.
A prior 26/40 seed-10000 batch survives in the conversation but was overwritten
on disk and is excluded from this aggregate. Report settings, not filenames,
identify runs: two filenames containing `r20` actually record five rollouts.

The report describes code at commit `55e6b04`. The 47 automated tests were rerun
successfully when preparing the draft. The 100-game random-engine validation
is a previously completed smoke test. No new gameplay experiments or agent
changes were made during report preparation. Only report files were edited;
nothing has been submitted, published, or pushed to GitHub.

Primary related-work sources checked:

- https://www.nealen.net/papers/3102071.3102105.pdf
- https://arxiv.org/abs/2009.04374
- https://cs.gettysburg.edu/~tneller/cs371/mcts-survey.pdf

Before submission, confirm the university, email addresses, report date,
required page limit, degree statement, and personal acknowledgements.
