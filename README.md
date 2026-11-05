# definitly-not-glorifeied-surveillance-system
Vibe Coded Art Project / Game thingy

## Deploy to Neocities

The GitHub Actions workflow deploys the site to Neocities on every push to
`main`. To enable it:

1. Copy the API key for your site from [Neocities settings](https://neocities.org/settings).
2. In the GitHub repository, open **Settings > Secrets and variables > Actions**.
3. Add a repository secret named `NEOCITIES_API_KEY` with that key as its value.
4. Push to `main`, or run **Deploy to Neocities** from the repository's Actions tab.

The workflow publishes the HTML entry point and its four JavaScript modules.

## In-browser game data

Face labels are temporary tracking numbers, not identities, and are not linked
to the same person across camera sessions. The browser stores only the next
number and point-slider settings; face data and game points are not saved.
