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
by Neocities before uploading. Face numbers, totals, and slider settings persist
in this browser. **Delete saved data** removes this app's local-storage keys and
clears the current scoreboard and camera session without writing zero scores
back. The app does not save images or face templates and does not identify
people across separate visits.

## Clothing camouflage detection

The optional clothing detector uses a Roboflow object-detection model through a
Cloudflare Worker. Train and deploy a model with the classes `yellow-jacket`,
`plague-doctor-mask`, and `blue-t-shirt`. Its bounding boxes should cover the
person wearing each item so detections can be associated with visible face
tracks. The game sends a reduced camera frame to the Worker every 1.5 seconds
while the camera is running; Roboflow processes those images.

Deploy `backend/clothing-worker.js` as a Cloudflare Worker and configure these
Worker variables:

- `ROBOFLOW_API_KEY` as a Worker secret.
- `ROBOFLOW_MODEL_ID` as the Roboflow model ID.
- `ROBOFLOW_MODEL_VERSION` as the deployed model version.
- `ALLOWED_ORIGIN` as the exact public site origin, for example
	`https://example.neocities.org`.

Set the Worker's public URL in the `clothing-api-url` meta tag in
`frontend/index.html`. Do not put the Roboflow API key in frontend code. The
Worker checks the browser origin, but a public Worker URL can still be called
outside the site; configure request limits in Cloudflare to protect API usage.
Without a trained model and configured Worker URL, camouflage is not detected
and the two-minute timer continues normally.
