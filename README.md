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
