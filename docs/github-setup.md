# GitHub setup and publication checklist

This clean folder contains no inherited `.git` directory. The original `03_code`
folder and its history/remote are preserved beside it.

1. Add the final report in `report/` using the documented filename.
2. Run the unit tests using the root README commands.
3. On GitHub create an **empty** repository named `hbos-reproduction`.
   Do not initialise it with a README, license or gitignore.
4. In PowerShell, open this folder and run:

```powershell
git init -b main
git add .
git status
git diff --cached --stat
git commit -m "Prepare HBOS reproduction for INFOMDW Assignment 2"
git remote add origin https://github.com/satyamkumarcode/hbos-reproduction.git
git push -u origin main
```

If that GitHub repository already contains commits, clone it separately and copy these
files into the clone, preserving its `.git` folder; then review, commit and push.
Do not force-push over existing work. The commands above assume an empty remote.

Suggested GitHub description: "INFOMDW Assignment 2: reproducible Python experiments for HBOS anomaly detection."
Suggested topics: `anomaly-detection`, `hbos`, `reproducibility`, `data-wrangling`, `python`.
Before publishing, check that staged files exclude virtual environments, credentials,
temporary drafts and other students' feedback. The repository ignore file covers common local files.
