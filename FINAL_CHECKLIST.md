# Final Submission Checklist — Section 19

Status is based on an actual local file/evidence review. `❌` marks an external or
student-controlled action, not a fabricated completion claim.

| Section 19 item | Evidence checked | Status |
| --- | --- | --- |
| Repository link/access correct | No Git repository/remote exists in this workspace yet; see `MANUAL_TODO.md`. | ❌ Needs student GitHub action |
| README setup works | `README.md` contains setup, `python -m pytest`, train, and generation commands; the acceptance script checks their presence. A clean public-clone install still needs student verification after push. | ❌ Needs student clean-clone check |
| Hand calculation included | `experiments/hand_calculation.py`, `outputs/figures/fig1_hand_calculated_attention.png`, report §2. | ✅ Verified |
| Causal mask shown | `outputs/figures/fig2_causal_mask.png`, `tests/test_mask.py`, report §2. | ✅ Verified |
| Multi-head shape trace included | `outputs/figures/fig3_multihead_shape_trace.png`, `outputs/logs/m2_multihead_shape_trace.txt`. | ✅ Verified |
| Positional evidence included | `outputs/figures/fig4_positional_encoding.png`, `src/positional_encoding.py`. | ✅ Verified |
| Architecture diagram included | `outputs/figures/fig_m3_transformer_block.png` and `fig_m3_minigpt_architecture.png`. | ✅ Verified |
| Training/validation curves included | `outputs/figures/fig6_loss_curve.png`; best step is marked. | ✅ Verified |
| At least five generated samples included | Five 200-token fixed-prompt files in `outputs/samples/` and `fig8_samples.png`. | ✅ Verified |
| Three required ablations complete | `outputs/logs/ablation_results.csv`, `.md`, `fig9_ablations.png`, report §7. | ✅ Verified |
| Parameter count/configuration documented | Fig5, `configs/tiny_gpt.yaml`, `m5_training_summary.*`, report §5. | ✅ Verified |
| CLI/API inference works | Real saved local CLI/API transcripts, `fig10_inference.png`, app source and API tests. Public hosting is not claimed. | ✅ Local verification |
| Report submitted as Medium article | `report/report.md` is complete, but publication/submission is student-controlled. | ❌ Needs student publication/submission |
| No secrets/unneeded large artifacts committed | `.gitignore` and static secret scan were reviewed; Git commit history cannot be checked before the repository is created/pushed. | ❌ Needs student pre-push review |
| All sources cited | README and report §Citations name assignment resources, corpus, Transformer/GPT, PyTorch, FastAPI, and adaptations. | ✅ Verified |

