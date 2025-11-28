# QBound Paper - LaTeX Source

This directory contains the complete LaTeX source for the QBound paper, ready for compilation or upload to Overleaf.

## Directory Structure

```
LatexDocs/
├── main.tex              # Main paper file
├── references.bib        # Bibliography
├── arxiv.sty            # ArXiv style file
├── figures/             # All experimental plots (PDFs)
│   ├── learning_curves_*.pdf               # Combined 3-panel figure
│   ├── gridworld_learning_curve_*.pdf      # GridWorld results
│   ├── frozenlake_learning_curve_*.pdf     # FrozenLake results
│   ├── cartpole_learning_curve_*.pdf       # CartPole results
│   └── comparison_bar_chart_*.pdf          # Bar chart comparison
└── README.md            # This file
```

## Compilation

### Local Compilation

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Overleaf

1. Create a ZIP file of this directory:
   ```bash
   zip -r qbound_paper.zip LatexDocs/
   ```

2. Upload `qbound_paper.zip` to Overleaf as a new project

3. Set compiler to `pdflatex`

4. Compile!

## Self-Contained

This directory is **completely self-contained** and portable:
- ✓ All LaTeX source files included
- ✓ All figures included in `figures/` directory
- ✓ All bibliography entries in `references.bib`
- ✓ ArXiv style file included

No external dependencies or paths needed!

## Experimental Results

The plots in `figures/` are generated from experiments run on:
- **Date**: 2025-10-25
- **Results**:
  - GridWorld: 20.2% faster (205 vs 257 episodes to 80% success)
  - FrozenLake: 5.0% faster (209 vs 220 episodes to 70% success)
  - CartPole: 31.5% higher cumulative reward (172,904 vs 131,438)

## Regenerating Plots

If you need to regenerate plots from experimental data:

```bash
# From the project root directory
cd /root/projects/QBound
python3 analysis/plot_paper_results.py
```

This will:
1. Load results from `results/combined/experiment_results_*.json`
2. Generate plots in `results/plots/`
3. **Automatically copy PDFs to `LatexDocs/figures/`**

## Notes

- All figure paths in `main.tex` use relative paths: `figures/filename.pdf`
- The paper includes 4 figures with experimental results
- Compilation requires standard LaTeX packages (all commonly available)
