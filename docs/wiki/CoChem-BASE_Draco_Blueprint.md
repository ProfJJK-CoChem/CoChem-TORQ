> **Historical source — reviewed and superseded for TORQ implementation.** Use the [implementation SRS](CoChem-TORQ_Implementation_SRS.md), [method implementation contract](Method_Matrix_Implementation_Contract.md) and [decision register](review/Decisions_and_Risks.md). The original content below is preserved for traceability; unsupported claims and conflicting instructions are not normative. Source-line references in the reviews refer to commit `d7a4739a5f7d6f22ed659b32eeb4706bef16225e`, before this banner.

## Interaction Environment + Calculation Environment: Local + wsl
- [ ] [UI Cell 1] Seamless Install View
  - [ ] [Selection] Calc Env: wsl
  - [ ] [Selection] Interact Env: Local
  - [ ] [Action] Click 'Run Installation'
- [ ] [UI Cell 2] No Code Matrix View
  - [ ] [Selection] Engine: ORCA
  - [ ] [Selection] Method: B3LYP
  - [ ] [Selection] Basis Set: cc-pVTZ
  - [ ] [Action] Click 'Save/Submit Matrix'
  - [ ] [Action] Click 'Execute Pipeline'

## Interaction Environment + Calculation Environment: GitHub Codespaces + github-actions
- [ ] [UI Cell 1] Seamless Install View
  - [ ] [Selection] Calc Env: github-actions
  - [ ] [Selection] Interact Env: GitHub Codespaces
  - [ ] [Action] Click 'Run Installation'
- [ ] [UI Cell 2] No Code Matrix View
  - [ ] [Selection] Engine: CFOUR
  - [ ] [Selection] Method: CCSD(T)
  - [ ] [Selection] Basis Set: aug-cc-pVTZ
  - [ ] [Action] Click 'Save/Submit Matrix'
  - [ ] [Action] Click 'Execute Pipeline'
