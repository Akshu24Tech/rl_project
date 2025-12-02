# ProPS Gemini 3.0 Comparison Project

> Testing whether Gemini 3.0 performs better than previous versions at reinforcement learning using Prompted Policy Search (ProPS)

## 🎯 What is This?

This project implements a **laptop-optimized version** of the ProPS research paper to compare different Gemini models on a reinforcement learning task. You'll discover which Gemini version is best at optimizing policies through in-context learning!

### Quick Facts

- ⏱️ **Time**: 30-60 minutes for full comparison (Deep Think takes longer but reasons better!)
- 💻 **Requirements**: Python 3.8+, Gemini API key (free tier works!)
- 🎮 **Task**: Solve MountainCar environment
- 🤖 **Models**: Gemini 3 Deep Think Preview, 2.5 Flash, 2.0 Flash

## 🚀 Quick Start (3 Commands)

```bash
# 1. Install dependencies
pip install -r requirements_props.txt

# 2. Run guided experiment
python run_experiment.py

# 3. Analyze results
python analyze_results.py
```

**That's it!** The script will guide you through everything.

> **🧠 Note**: This experiment uses **Gemini 3 Deep Think Preview**, a model designed for complex reasoning. It takes longer per decision (~5-30 seconds vs 1-2 seconds) but should provide better optimization through deeper analysis. See [DEEP_THINK_NOTES.md](DEEP_THINK_NOTES.md) for details!

## 📖 Documentation

| Document | Purpose | Read Time |
|----------|---------|-----------|
| **[INDEX.md](INDEX.md)** | Complete navigation guide | 5 min |
| **[GETTING_STARTED.md](GETTING_STARTED.md)** | Step-by-step setup | 10 min |
| **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** | Full project overview | 15 min |
| **[WORKFLOW.md](WORKFLOW.md)** | Visual workflow diagrams | 10 min |
| **[README_COMPARISON.md](README_COMPARISON.md)** | Technical details | 20 min |

**New here?** Start with [GETTING_STARTED.md](GETTING_STARTED.md)

## 🎓 What is ProPS?

**Prompted Policy Search (ProPS)** is a novel RL method that uses Large Language Models as optimizers:

```
Traditional RL:          ProPS:
┌─────────────┐         ┌─────────────┐
│   Neural    │         │     LLM     │
│   Network   │         │  (Gemini)   │
│  + Gradient │         │ + Prompting │
│   Descent   │         │             │
└─────────────┘         └─────────────┘
      ↓                       ↓
  Learn policy          Learn policy
  through               through
  backprop              reasoning
```

### Key Innovation

Instead of gradient descent, ProPS:
1. Shows LLM history of (parameters → reward) pairs
2. LLM reasons about which parameters to try next
3. Evaluates suggested parameters
4. Repeats until optimal policy found

**Result**: Competitive with traditional RL, no gradient computation needed!

## 🔬 The Experiment

### Task: MountainCar

An underpowered car must reach the goal by building momentum:

```
     Goal!
      🏁
       /\
      /  \
     /    \___
    /         \
   /           \___
  /                \
 /                  \
🚗 ← Start here
```

**Challenge**: Car can't drive straight up. Must learn to swing back and forth!

### What We're Testing

**Research Question**: Does Gemini 3.0 optimize policies better than previous versions?

**Models Compared**:
- Gemini 3 Deep Think Preview ← **New! Deep reasoning model**
- Gemini 2.5 Flash
- Gemini 2.0 Flash

**Metrics**:
- Final reward achieved
- Learning speed
- Consistency
- Execution time

## 📊 Expected Results

### Good Performance
```
Episode:  1  10  20  30  40  50
Reward:  45  63  75  83  89  91  ← Learning!
```

### What Success Looks Like
- Rewards increase over episodes
- Final reward > 80
- Smooth learning curve
- One model clearly better

## 🛠️ Project Structure

```
PrPoS/
├── 📖 Documentation
│   ├── INDEX.md                    # Navigation hub
│   ├── GETTING_STARTED.md          # Quick start
│   ├── PROJECT_SUMMARY.md          # Complete overview
│   ├── WORKFLOW.md                 # Visual guide
│   └── README_COMPARISON.md        # Technical docs
│
├── 🐍 Scripts
│   ├── run_experiment.py           # Master script ⭐
│   ├── props_gemini_comparison.py  # Main experiment
│   ├── quick_test.py               # Fast validation
│   ├── test_setup.py               # Setup check
│   └── analyze_results.py          # Results analysis
│
├── ⚙️ Config
│   └── requirements_props.txt      # Dependencies
│
└── 📊 Results (generated)
    └── results/
        ├── comparison_*.png        # Plots
        ├── results_*.json          # Data
        └── analysis_*.png          # Analysis
```

## 🎮 Usage Examples

### Example 1: First Time User
```bash
# Read the guide
cat GETTING_STARTED.md

# Run guided experiment
python run_experiment.py
```

### Example 2: Quick Test
```bash
# Just verify everything works (2-5 min)
python quick_test.py
```

### Example 3: Full Comparison
```bash
# Run complete experiment (15-30 min)
python props_gemini_comparison.py

# Analyze results
python analyze_results.py
```

### Example 4: Custom Configuration
```python
# Edit props_gemini_comparison.py
CONFIG = {
    "NUM_EPISODES": 100,  # More episodes
    "GEMINI_MODELS": [
        "gemini-2.0-flash-exp"  # Test only one model
    ]
}
```

## 📈 Sample Output

```
ProPS - Gemini Model Comparison
============================================================

Running experiment with gemini-2.0-flash-exp
============================================================

Running 10 warmup episodes...
  Warmup 1/10: Reward = 42.31
  Warmup 2/10: Reward = 38.92
  ...

Starting training for 50 episodes...
Episode 1/50: Reward = 45.23, Params = [2.1, 3.4]
Episode 2/50: Reward = 52.87, Params = [2.3, 3.8]
...
Episode 50/50: Reward = 91.24, Params = [3.2, 4.1]

Completed in 487.32 seconds
Final Reward: 91.24
Max Reward: 92.15
Avg Reward: 78.43

============================================================
SUMMARY
============================================================

gemini-2.0-flash-exp:
  Final Reward: 91.24
  Max Reward: 92.15
  Avg Reward: 78.43
  Time: 487.32s

🏆 Best Model: gemini-2.0-flash-exp
```

## 🔧 Configuration

### Laptop-Optimized (Default)
```python
CONFIG = {
    "NUM_EPISODES": 50,              # 5x faster than paper
    "WARMUP_EPISODES": 10,
    "NUM_EVALUATION_EPISODES": 5,
    "MAX_TRAJ_LENGTH": 500,
}
```
**Time**: 15-30 minutes

### Ultra-Fast (Testing)
```python
CONFIG = {
    "NUM_EPISODES": 10,
    "WARMUP_EPISODES": 3,
    "NUM_EVALUATION_EPISODES": 2,
}
```
**Time**: 2-5 minutes

### High-Quality (Research)
```python
CONFIG = {
    "NUM_EPISODES": 200,
    "WARMUP_EPISODES": 20,
    "NUM_EVALUATION_EPISODES": 20,
}
```
**Time**: 1-2 hours

## 🔑 Getting API Key

1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Sign in with Google account
3. Click "Get API Key"
4. Create new API key
5. Copy key (starts with "AI...")
6. Paste when script asks

**Free tier includes**: 60 requests/minute, plenty for this experiment!

## 📊 Analysis Features

The `analyze_results.py` script provides:

- **Learning Curves**: Reward over episodes with confidence intervals
- **Model Comparison**: Bar charts comparing final/max rewards
- **Statistical Summary**: Mean, std, min, max for each model
- **Rankings**: Best model by different metrics
- **LaTeX Export**: Publication-ready tables

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | `pip install -r requirements_props.txt` |
| API key invalid | Get new key from aistudio.google.com |
| Rate limits | Wait 5-10 min or reduce NUM_EPISODES |
| Low rewards | Normal! Try more episodes |
| Out of memory | Reduce MAX_TRAJ_LENGTH |

**More help**: See [GETTING_STARTED.md](GETTING_STARTED.md) → Troubleshooting

## 🎯 Next Steps

After running the experiment:

1. **Analyze Results**
   - Which model performed best?
   - Was learning consistent?
   - How do results compare to paper?

2. **Run More Experiments**
   - Multiple runs for statistics
   - Different configurations
   - Other environments

3. **Extend the Project**
   - Implement ProPS+ (with environment descriptions)
   - Test other LLMs
   - Compare with traditional RL

4. **Share Findings**
   - Document interesting results
   - Create visualizations
   - Contribute to research

## 📚 Research Context

Based on:
**"Prompted Policy Search (ProPS): Reinforcement Learning through Linguistic and Numerical Reasoning in LLMs"**
- Authors: Zhou et al., 2025
- Paper: See `26193_Prompted_Policy_Search_R.pdf`
- Website: https://props-llm.github.io/

### Key Contributions

- First to use LLMs as RL optimizers via prompting
- Combines numerical optimization with linguistic reasoning
- Achieves competitive performance without gradients
- Enables interpretable policy search

### Your Contribution

By testing Gemini 3.0, you're:
- Extending research to newer models
- Validating ProPS on different setups
- Contributing to understanding of LLM capabilities

## 🤝 Contributing

Found interesting results? Consider:
- Documenting your findings
- Sharing plots and data
- Testing other environments
- Improving the code

## 📄 License

This project is for educational and research purposes. Original ProPS research by Zhou et al., 2025.

## 🙏 Acknowledgments

- Zhou et al. for the ProPS research
- Google for Gemini API
- OpenAI Gym/Gymnasium for environments

## 📞 Quick Links

- **Start Here**: [GETTING_STARTED.md](GETTING_STARTED.md)
- **Full Guide**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
- **Visual Guide**: [WORKFLOW.md](WORKFLOW.md)
- **Navigation**: [INDEX.md](INDEX.md)
- **API Key**: https://aistudio.google.com/
- **ProPS Website**: https://props-llm.github.io/

---

## 🚀 Ready to Start?

```bash
# Windows users
run.bat

# Everyone else
python run_experiment.py
```

**Good luck with your experiment! 🎉**

If you find that Gemini 3.0 performs better, that's a valuable contribution to the research community!
