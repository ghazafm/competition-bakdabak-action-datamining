# 📝 Quick Reference Card

## 🎯 Most Common Commands

### Generate Predictions (Choose One)

**Predict Method (Recommended):**
```bash
python predict_batch.py --batch-size 32 --verify
```

**With All Options:**
```bash
python predict_batch.py --batch-size 32 --img-size 640 --save-confidence --verify
```

---

## 📊 Analyze Results

```bash
# Basic analysis
python analyze_predictions.py

# Full analysis with comparison and report
python analyze_predictions.py --compare --report
```

---

## 🔍 Quick Checks

### In Terminal
```bash
# Count lines (should be test_images + 1 for header)
wc -l submission.csv

# View first few rows
head submission.csv

# Check specific food
grep "Rendang" submission.csv | wc -l
```

### In Python
```python
import pandas as pd
df = pd.read_csv('submission.csv')
print(len(df))  # Total predictions
print(df['label'].value_counts())  # Class distribution
```

---

## 📁 Input/Output Files

| File | Description |
|------|-------------|
| `model.pt` | Your trained YOLO model (input) |
| `data-mining-action-2025/test/test/*.jpg` | Test images (input) |
| `submission.csv` | Kaggle submission file (output) |
| `submission_detailed.csv` | With confidence scores (output) |

---

## 🍜 Food Classes (15 total)

1. Ayam Bakar
2. Ayam Betutu
3. Ayam Goreng
4. Ayam Pop
5. Bakso
6. Coto Makassar
7. Gado Gado
8. Gudeg
9. Nasi Goreng
10. Pempek
11. Rawon
12. Rendang
13. Sate Madura
14. Sate Padang
15. Soto

---

## ⚡ Speed Tips

| Batch Size | Speed | Memory |
|------------|-------|--------|
| 8 | Slow | Low |
| 16 | Medium | Medium |
| 32 | Fast | High |
| 64 | Fastest | Very High |

---

## ✅ Pre-Submission Checklist

- [ ] File named `submission.csv`
- [ ] 2 columns: `ID`, `label`
- [ ] Correct number of rows
- [ ] No missing/duplicate IDs
- [ ] Run `--verify` flag
- [ ] Analyze results

---

## 🆘 Common Issues

| Issue | Solution |
|-------|----------|
| Model not found | Check `model.pt` exists |
| Out of memory | Use `--batch-size 4` |
| No images found | Check test directory path |
| Wrong count | Run with `--verify` |

---

## 📚 Documentation Files

- `COMPLETE_GUIDE.md` - Full tutorial
- `README_USAGE.md` - Detailed usage
- `README_PREDICTION.md` - Quick start

---

## 🔗 Typical Workflow

```
1. predict_batch.py → submission.csv
2. analyze_predictions.py → check results
3. Upload to Kaggle → 🎉
```

---

**Need help?** Check `COMPLETE_GUIDE.md` for detailed instructions!
