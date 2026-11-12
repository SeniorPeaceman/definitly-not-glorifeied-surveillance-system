# definitly not glorified surveillance system
Vibe Coded Art Project / Game thingy

## Deploy to Neocities

The GitHub Actions workflow deploys the site to Neocities on every push to
`main`. To enable it:

1. Copy the API key for your site from [Neocities settings](https://neocities.org/settings).
2. In the GitHub repository, open **Settings > Secrets and variables > Actions**.
3. Add a repository secret named `NEOCITIES_API_KEY` with that key as its value.
4. Push to `main`, or run **Deploy to Neocities** from the repository's Actions tab.

The HTML page and stylesheet are in `frontend/`; the JavaScript modules are in
`backend/`. The workflow packages those files into the root-level layout expected
by Neocities before uploading. Pose tracks assign player numbers. Each player
has a two-minute countdown and an elapsed-play timer; recognized cues subtract
the configured number of seconds. Yellow and blue torso-color checks pause the
countdown as camouflage. These color checks are heuristic and depend on lighting
and camera quality.

MediaPipe Tasks Vision can also run its ObjectDetector in the browser with a
compatible model asset and camera permission (HTTPS or localhost). A generic
pretrained detector does not reliably identify specific garments or a
pest-doctor mask; those classes need an appropriate custom-trained model. This
app does not currently load an object-detection model. Player numbers, timers,
totals, and slider settings persist in this browser. **Delete saved data** removes
this app's local-storage keys and clears the current scoreboard and camera
session without writing zero scores back. The app does not save images or face
templates and does not identify people across separate visits.
