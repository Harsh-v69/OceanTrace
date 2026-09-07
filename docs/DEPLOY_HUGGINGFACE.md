# Deploy to Hugging Face Spaces (free, ~10 minutes)

A temporary public deployment for a demo. Free CPU tier, no card, HTTPS URL of
the form `https://<your-username>-oceantrace.hf.space`.

The repo already contains everything the Space needs: `Dockerfile`,
`.dockerignore`, and the HF front-matter at the top of `README.md`.

---

## 1. Create the Space

1. Sign in / sign up at <https://huggingface.co> (free).
2. <https://huggingface.co/new-space>
   - **Owner**: your username
   - **Space name**: `oceantrace`
   - **License**: your choice (e.g. `mit`)
   - **SDK**: **Docker** → **Blank**
   - **Hardware**: **CPU basic** (free, 2 vCPU / 16 GB)
   - **Visibility**: **Public**
3. **Create Space**. You now have an empty Space repo at
   `https://huggingface.co/spaces/<username>/oceantrace`.

## 2. Push this repository to the Space

The Space is a git repo. Add it as a second remote and push `main` to it.

```bash
# from the project root, on the branch you want to deploy (main)
git remote add space https://huggingface.co/spaces/<username>/oceantrace
git push space main
```

When git prompts for a password, use a **Hugging Face access token** with
*write* scope: <https://huggingface.co/settings/tokens> → **New token** → Role
**Write**. Username is your HF username.

> Large files: this repo's biggest committed file is ~1 MB (the coastline
> GeoJSON and the model binaries are all small), so a plain `git push` is fine —
> no Git LFS setup required.

The Space starts building as soon as the push lands. First build ≈ 8–12 minutes
(most of it is the CPU PyTorch wheel). Watch the **Logs** tab; it goes
`Building` → `Running`.

## 3. Set the two secrets

In the Space: **Settings → Variables and secrets → New secret**.

| Name | Value |
|---|---|
| `JWT_SECRET_KEY` | any long random string, e.g. `openssl rand -hex 32` |
| `DEFAULT_USER_PASSWORD` | the password you'll use to sign in at the demo |

Adding secrets triggers a restart. After it, the seeded accounts use your
`DEFAULT_USER_PASSWORD`.

Optional extras (only if you want live SMS / real met-ocean — the app works
without them):

| Name | Purpose |
|---|---|
| `SMS_PROVIDER` = `twilio` | switch off the mock SMS provider |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER` | Twilio creds |

## 4. Open it

`https://<username>-oceantrace.hf.space` → the Operations Console loads.

Sign in:

| Role | Email |
|---|---|
| National | `national@oceantrace.gov.in` |
| Regional | `regional@oceantrace.gov.in` |
| Pilot | `pilot@oceantrace.gov.in` |

Password = your `DEFAULT_USER_PASSWORD`.

Demo path: **Scenarios → mumbai-high-confidence → Run** for a full
detect → drift → attribute pipeline with an evidence dossier.

---

## Updating the deployment

Push again:

```bash
git push space main
```

Each push rebuilds the Space.

## Notes for the demo window

- **Ephemeral storage.** The free tier has no persistent disk. The SQLite DB
  lives inside the running container: it survives while the Space is up, but a
  rebuild or a hardware restart wipes it. Jurisdictions and the three role
  accounts re-seed automatically on boot; the four demo scenarios are
  deterministic, so they always produce the same result. Only ad-hoc
  investigations created during a session are lost on a restart.
- **Sleep.** A free Space sleeps after ~48 h of no traffic. The first request
  after sleep takes ~30–60 s to wake and lazy-load the models. Open it a few
  minutes before the jury session to warm it.
- **First pipeline run** after each boot pays a one-time model-load cost
  (~3–5 s for the SAR ensemble, ~2 s each for the AIS autoencoder and the LSTM).
  Subsequent runs are warm.
- **Tear-down.** When the demo is over: Space **Settings → Delete this Space**.
- The Space is independent of the GitHub repo — pushing to `origin` (GitHub)
  does not touch the Space, and vice versa. Push to `space` to deploy.
