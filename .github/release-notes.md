**Gameplay AutoCut __VERSION__** — download `GameplayAutoCut.zip`, extract it to a folder and run `GameplayAutoCut.exe`. No installer, no Python, no ffmpeg. The Moments tab downloads its AI models into `GameplayAutoCut-data` beside it on first use.

### Verify before you run it

The executable is unsigned, so Windows SmartScreen warns on first launch and some antivirus engines flag PyInstaller builds on sight. Check the download against the build instead:

```powershell
Get-FileHash GameplayAutoCut.zip -Algorithm SHA256   # expect the value below
```

| File | SHA-256 |
| --- | --- |
| `GameplayAutoCut.zip` | `__ZIP_HASH__` |
| `GameplayAutoCut.exe` | `__EXE_HASH__` |

`SHA256SUMS.txt` below carries the same two hashes in the usual format.

The zip also carries a GitHub build provenance attestation — proof it came out of this repository's release workflow at commit `__COMMIT__`, not off someone's machine:

```powershell
gh attestation verify GameplayAutoCut.zip --repo __REPO__
```
