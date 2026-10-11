# Manual Student Actions Before Submission

- [ ] Initialise a Git repository if needed, create an instructor-accessible GitHub
  repository, and push this workspace.
- [ ] Review `git status` and ensure the raw dataset, full/resume/tuning checkpoints,
  `.env`, generated caches, and no credentials are included. Do not alter
  `checkpoints/minigpt_final.pt` if you choose to publish it.
- [ ] Upload the 3.177 MiB `checkpoints/minigpt_final.pt` best-validation checkpoint
  as a GitHub Release asset (or an approved public direct-download location).
- [ ] Replace `GITHUB_RELEASE_CHECKPOINT_URL_PLACEHOLDER` in README and set the same
  direct URL as Render’s `CHECKPOINT_URL`; do not commit a private signed URL.
- [ ] Create a Render account, deploy using `render.yaml`, verify `/health`,
  `/model-info`, and `/generate`, then replace `LIVE_RENDER_URL_PLACEHOLDER`.
- [ ] Set the `GITHUB_URL` and `CHECKPOINT_URL` placeholders in the Colab notebook,
  then run it top-to-bottom once in Colab and record any platform-specific issue.
- [ ] Publish or submit `report/report.md` as the requested Medium-style article;
  only then add its actual URL if your instructor asks for it.
- [ ] Rehearse `DEMO_SCRIPT.md`, record the required 5–10 minute screen demo, and
  make sure any live-deployment claim is true at recording time.
- [ ] Turn `SLIDES_OUTLINE.md` into presentation slides and deliver the class
  presentation.
- [ ] Read `VIVA_QA.md`, inspect the source code, and be prepared to defend every
  design choice and measured result yourself.

