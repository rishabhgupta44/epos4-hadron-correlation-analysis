# EPOS4 p-Lambda Correlation Study

This is a small ROOT based analysis using EPOS4 generated events.

The idea is to study basic femtoscopy style observables for p-Lambda and pp-Lambda systems.  
This is not an official ALICE analysis. It is only a preliminary generator level study.

---

## What this code does

The baseline macro reads an EPOS4 ROOT file and selects final particles inside a given eta range.

It studies:

- charged multiplicity
- proton and Lambda pT distributions
- same event p-Lambda pairs
- mixed event p-Lambda pairs
- same event pp-Lambda triplets
- mixed event pp-Lambda triplets
- source separation variables from EPOS4 space-time information

The main observables are:

```text
k*   -> relative momentum for p-Lambda pairs
Q3   -> three particle relative momentum for pp-Lambda
r    -> source separation in lab frame
r*   -> source separation in pair rest frame
rho  -> simple toy source size for pp-Lambda
