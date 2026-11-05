import {
  DrawingUtils,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
import {
  classifyFace,
  createFaceLandmarker,
  drawFaceMesh,
  getFaceBounds,
} from "./face-tracking.js";
import {
  classifyGesture,
  createHandLandmarker,
  drawHands,
} from "./hand-tracking.js";
import {
  classifyPose,
  createPoseLandmarker,
  drawPoses,
} from "./pose-tracking.js";

// Set model limits, frame cadence, track lifetime, and per-cue repeat cooldown.
const MAX_FACES = 5;
const MAX_HANDS = 4;
const MAX_POSES = 5;
const INFERENCE_INTERVAL_MS = 1000 / 12;
const TRACK_TIMEOUT_MS = 2200;
const CUE_COOLDOWN_MS = 1200;
const STORAGE_KEY = "definitly-not-glorified-surveillance-system-v1";
const LEGACY_STORAGE_KEY = "cue-carnival-game-v1";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

// Connect each detectable cue to its slider, display label, and default points.
const CUES = [
  { id: "fist", label: "Fist", input: "points-fist" },
  { id: "raised-fist", label: "Raised fist", input: "points-raised-fist" },
  { id: "middle-finger", label: "Middle finger", input: "points-middle-finger" },
  { id: "open-mouth", label: "Open mouth", input: "points-open-mouth" },
  { id: "lowered-brows", label: "Lowered brows", input: "points-lowered-brows" },
  { id: "both-arms-raised", label: "Both arms raised", input: "points-both-arms-raised" },
  { id: "arms-spread", label: "Arms spread", input: "points-arms-spread" },
];
const cueById = new Map(CUES.map((cue) => [cue.id, cue]));

// Load only saved game data and tolerate missing, invalid, or blocked storage.
function readSavedGame() {
  try {
    const savedValue = localStorage.getItem(STORAGE_KEY) ?? localStorage.getItem(LEGACY_STORAGE_KEY);
    const saved = JSON.parse(savedValue ?? "null");
    const faceScores = Array.isArray(saved?.faceScores)
      ? saved.faceScores.filter((score) => Number.isInteger(score.id) && Number.isFinite(score.points))
      : Array.isArray(saved?.players)
        ? saved.players.filter((score) => Number.isInteger(score.id) && Number.isFinite(score.points))
      : [];
    const unassignedScores = Array.isArray(saved?.unassignedScores)
      ? saved.unassignedScores.filter(Number.isFinite)
      : [];
    const highestId = faceScores.reduce((highest, score) => Math.max(highest, score.id), 0);
    return {
      nextFaceId: Math.max(highestId + 1, Number(saved?.nextFaceId) || 1),
      faceScores,
      unassignedScores,
      settings: saved?.settings && typeof saved.settings === "object" ? saved.settings : {},
    };
  } catch {
    return { nextFaceId: 1, faceScores: [], unassignedScores: [], settings: {} };
  }
}

const savedGame = readSavedGame();
const faceScores = new Map(savedGame.faceScores.map((score) => [score.id, score]));
let unassignedScores = savedGame.unassignedScores;
// Keep active camera and face-track state in memory; only the scoreboard is persisted.
let nextFaceId = savedGame.nextFaceId;
let storageEnabled = true;
let faceLandmarker;
let handLandmarker;
let poseLandmarker;
let cameraStream;
let animationFrameId;
let lastInferenceAt = 0;
const faceTracks = new Map();

// Cache the controls and output elements used by the game page.
const startButton = document.querySelector("#start-button");
const stopButton = document.querySelector("#stop-button");
const deleteSavedDataButton = document.querySelector("#delete-saved-data-button");
const statusElement = document.querySelector("#status");
const video = document.querySelector("#camera");
const overlay = document.querySelector("#tracking-overlay");
const stageIdle = document.querySelector("#stage-idle");
const overlayContext = overlay.getContext("2d");
const faceCount = document.querySelector("#face-count");
const handCount = document.querySelector("#hand-count");
const poseCount = document.querySelector("#pose-count");
const scoreList = document.querySelector("#score-list");
const drawingUtils = new DrawingUtils(overlayContext);

// Show camera, permission, and storage feedback in the page's live status region.
function setStatus(message) {
  statusElement.textContent = message;
}

// Persist the scoreboard and cue settings unless the user deleted saved data this session.
function saveGame() {
  if (!storageEnabled) return;

  try {
    const savedFaceScores = [...faceScores.values()];
    const savedData = { nextFaceId, faceScores: savedFaceScores, unassignedScores };
    savedData.settings = Object.fromEntries(
      CUES.map(({ id, input }) => [id, Number(document.querySelector(`#${input}`).value)]),
    );
    localStorage.setItem(STORAGE_KEY, JSON.stringify(savedData));
    localStorage.removeItem(LEGACY_STORAGE_KEY);
  } catch {
    setStatus("Browser storage is unavailable; this session's scores will not persist.");
  }
}

// Read the current slider value for a detected cue.
function getPoints(cueId) {
  const cue = cueById.get(cueId);
  return cue ? Number(document.querySelector(`#${cue.input}`).value) : 0;
}

// Create a zero-point scoreboard entry when a face receives its first track ID.
function ensureFaceScore(id) {
  if (!faceScores.has(id)) {
    faceScores.set(id, { id, points: 0, lastEvent: "Ready for cues" });
  }
  return faceScores.get(id);
}

// Render the saved scoreboard in descending point order and mark visible faces.
function updateScoreboard() {
  scoreList.replaceChildren();
  const activeIds = new Set(
    [...faceTracks.values()]
      .filter((track) => performance.now() - track.lastSeen < 1000)
      .map((track) => track.id),
  );
  const standings = [
    ...[...faceScores.values()].map((score) => ({ ...score, kind: "face" })),
    ...unassignedScores.map((points, index) => ({ points, index, kind: "unassigned" })),
  ].sort((first, second) =>
    second.points - first.points || (first.id ?? Infinity) - (second.id ?? Infinity),
  );

  if (standings.length === 0) {
    const empty = document.createElement("li");
    empty.className = "empty-state";
    empty.textContent = "No points yet. Start the camera to begin.";
    scoreList.append(empty);
    return;
  }

  for (const [rank, scoreEntry] of standings.entries()) {
    const item = document.createElement("li");
    item.className = "score-row";

    const identity = document.createElement("div");
    identity.className = "score-identity";
    const detail = document.createElement("span");
    detail.className = "score-detail";
    const name = document.createElement("strong");
    name.textContent = scoreEntry.kind === "face"
      ? `Face ${scoreEntry.id}`
      : `Unassigned total ${scoreEntry.index + 1}`;
    const event = document.createElement("small");
    event.textContent = scoreEntry.kind === "face"
      ? scoreEntry.lastEvent ?? "No recent cue"
      : "Face number cleared";
    detail.append(name, event);
    if (scoreEntry.kind === "face") {
      const number = document.createElement("span");
      number.className = "face-number";
      number.textContent = `#${scoreEntry.id}`;
      identity.append(number);
    }
    identity.append(detail);

    const score = document.createElement("strong");
    score.className = "score-total";
    score.textContent = String(scoreEntry.points);
    score.setAttribute("aria-label", `${scoreEntry.points} points`);

    const state = document.createElement("span");
    const isLive = scoreEntry.kind === "face" && activeIds.has(scoreEntry.id);
    state.className = isLive ? "face-score-state is-live" : "face-score-state";
    state.textContent = isLive ? "LIVE" : rank === 0 ? "LEAD" : "SAVED";

    item.append(identity, score, state);
    scoreList.append(item);
  }
}

// Mirror slider values in their outputs and optionally persist the settings.
function updatePointControls(save = true) {
  for (const cue of CUES) {
    const input = document.querySelector(`#${cue.input}`);
    const output = document.querySelector(`#${cue.input}-value`);
    output.value = `+${input.value}`;
  }
  if (save) saveGame();
}

// Match each frame's detected faces to recent tracks, assigning new saved IDs when needed.
function matchFaces(faceLandmarks, now) {
  for (const [id, track] of faceTracks) {
    if (now - track.lastSeen >= TRACK_TIMEOUT_MS) faceTracks.delete(id);
  }
  const availableTracks = new Set(
    [...faceTracks.values()].filter((track) => now - track.lastSeen < TRACK_TIMEOUT_MS),
  );

  return faceLandmarks.map((landmarks) => {
    const bounds = getFaceBounds(landmarks);
    let bestTrack;
    let bestCost = Infinity;

    for (const track of availableTracks) {
      const distance = Math.hypot(bounds.centerX - track.centerX, bounds.centerY - track.centerY);
      const sizeChange = Math.abs(Math.log((bounds.width + 0.01) / (track.width + 0.01)));
      const cost = distance + sizeChange * 0.025;
      if (distance < Math.max(0.12, track.width * 0.85) && cost < bestCost) {
        bestTrack = track;
        bestCost = cost;
      }
    }

    if (bestTrack) {
      availableTracks.delete(bestTrack);
      bestTrack.centerX = bounds.centerX;
      bestTrack.centerY = bounds.centerY;
      bestTrack.width = bounds.width;
      bestTrack.lastSeen = now;
      return bestTrack;
    }

    const track = {
      id: nextFaceId++,
      centerX: bounds.centerX,
      centerY: bounds.centerY,
      width: bounds.width,
      lastSeen: now,
      activeCues: new Set(),
      lastAwardAt: new Map(),
    };
    faceTracks.set(track.id, track);
    ensureFaceScore(track.id);
    saveGame();
    return track;
  });
}

// Associate a pose head or hand point with the nearest visible face, if close enough.
function closestFaceId(point, faceLandmarks, tracks, maxDistance) {
  if (!point) return undefined;
  let closestId;
  let closestDistance = maxDistance;

  for (let index = 0; index < faceLandmarks.length; index += 1) {
    const bounds = getFaceBounds(faceLandmarks[index]);
    const distance = Math.hypot(point.x - bounds.centerX, point.y - bounds.centerY);
    if (distance < closestDistance) {
      closestDistance = distance;
      closestId = tracks[index].id;
    }
  }
  return closestId;
}

// Estimate the head location from the pose model's face and eye landmarks.
function poseHead(pose) {
  const headPoints = pose.slice(0, 7).filter((point) => point && (point.visibility ?? 1) >= 0.4);
  if (headPoints.length === 0) return undefined;
  return {
    x: headPoints.reduce((total, point) => total + point.x, 0) / headPoints.length,
    y: headPoints.reduce((total, point) => total + point.y, 0) / headPoints.length,
  };
}

// Combine facial expressions, body poses, and hand gestures under a face track.
function collectCueEvents(faceResult, handResult, poseResult, tracks) {
  const cuesByFace = new Map(tracks.map((track) => [track.id, new Set()]));
  const addCue = (id, cue) => cuesByFace.get(id)?.add(cue);

  for (const [index, track] of tracks.entries()) {
    const blendshapes = faceResult.faceBlendshapes[index];
    if (blendshapes) {
      for (const cue of classifyFace(blendshapes)) addCue(track.id, cue);
    }
  }

  const poseOwners = poseResult.landmarks.map((pose) => {
    const id = closestFaceId(poseHead(pose), faceResult.faceLandmarks, tracks, 0.3);
    if (id !== undefined) {
      for (const cue of classifyPose(pose)) addCue(id, cue);
    }
    return id;
  });

  for (const hand of handResult.landmarks) {
    const cue = classifyGesture(hand);
    if (!cue) continue;

    const wrist = hand[0];
    let owner;
    let nearestWristDistance = 0.28;
    for (const [poseIndex, pose] of poseResult.landmarks.entries()) {
      const poseOwner = poseOwners[poseIndex];
      if (poseOwner === undefined) continue;
      for (const wristIndex of [15, 16]) {
        const poseWrist = pose[wristIndex];
        if (!poseWrist || (poseWrist.visibility ?? 1) < 0.35) continue;
        const distance = Math.hypot(wrist.x - poseWrist.x, wrist.y - poseWrist.y);
        if (distance < nearestWristDistance) {
          nearestWristDistance = distance;
          owner = poseOwner;
        }
      }
    }

    owner ??= closestFaceId(wrist, faceResult.faceLandmarks, tracks, 0.32);
    if (owner !== undefined) addCue(owner, cue);
  }

  return cuesByFace;
}

// Award points only when a cue first appears and its repeat cooldown has elapsed.
function scoreCueEvents(cuesByFace, tracks, now) {
  let changed = false;
  const visibleIds = new Set(tracks.map((track) => track.id));

  for (const track of faceTracks.values()) {
    if (!visibleIds.has(track.id)) track.activeCues.clear();
  }

  for (const [id, cues] of cuesByFace) {
    const track = faceTracks.get(id);
    const faceScore = ensureFaceScore(id);
    for (const cueId of cues) {
      if (track.activeCues.has(cueId)) continue;
      const lastAwardAt = track.lastAwardAt.get(cueId) ?? -Infinity;
      if (now - lastAwardAt < CUE_COOLDOWN_MS) continue;

      const points = getPoints(cueId);
      faceScore.points += points;
      faceScore.lastEvent = `${cueById.get(cueId).label} +${points}`;
      track.lastAwardAt.set(cueId, now);
      changed = true;
    }
    track.activeCues = cues;
  }

  if (changed) saveGame();
}

// Run throttled model inference, update counts and scores, and redraw the overlay.
function detectFrame() {
  if (!cameraStream) return;
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    animationFrameId = requestAnimationFrame(detectFrame);
    return;
  }

  const now = performance.now();
  if (now - lastInferenceAt < INFERENCE_INTERVAL_MS) {
    animationFrameId = requestAnimationFrame(detectFrame);
    return;
  }
  lastInferenceAt = now;

  try {
    const faceResult = faceLandmarker.detectForVideo(video, now);
    const handResult = handLandmarker.detectForVideo(video, now);
    const poseResult = poseLandmarker.detectForVideo(video, now);
    const tracks = matchFaces(faceResult.faceLandmarks, now);
    const cuesByFace = collectCueEvents(faceResult, handResult, poseResult, tracks);

    faceCount.value = String(faceResult.faceLandmarks.length);
    handCount.value = String(handResult.landmarks.length);
    poseCount.value = String(poseResult.landmarks.length);
    scoreCueEvents(cuesByFace, tracks, now);

    drawFaceMesh(faceResult, video, overlay, overlayContext, drawingUtils, tracks.map((track) => track.id));
    drawPoses(poseResult, drawingUtils);
    drawHands(handResult, drawingUtils);
    updateScoreboard();
  } catch (error) {
    stopCamera(`Tracking stopped: ${error.message}`);
    return;
  }

  animationFrameId = requestAnimationFrame(detectFrame);
}

// Request camera permission, initialize all three models, and start the frame loop.
async function startCamera() {
  startButton.disabled = true;
  setStatus("Waiting for camera permission…");

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera access needs HTTPS and a supported browser.");
    }

    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "user" },
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 24, max: 30 },
      },
    });
    video.srcObject = cameraStream;
    await video.play();
    stageIdle.hidden = true;

    if (!faceLandmarker || !handLandmarker || !poseLandmarker) {
      setStatus("Loading face, hand, and pose models…");
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      [faceLandmarker, handLandmarker, poseLandmarker] = await Promise.all([
        faceLandmarker ?? createFaceLandmarker(vision, MAX_FACES),
        handLandmarker ?? createHandLandmarker(vision, MAX_HANDS),
        poseLandmarker ?? createPoseLandmarker(vision, MAX_POSES),
      ]);
    }

    stopButton.disabled = false;
    lastInferenceAt = 0;
    setStatus("Camera live. Points and face numbers are saved on this device only.");
    animationFrameId = requestAnimationFrame(detectFrame);
  } catch (error) {
    stopCamera(`Could not start camera or models: ${error.message}`);
  } finally {
    startButton.disabled = false;
  }
}

// Stop the camera, clear transient tracks, and leave saved scoreboard data intact.
function stopCamera(message = "Camera off. Scores stay saved on this device.") {
  if (animationFrameId !== undefined) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = undefined;
  }
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = undefined;
  video.srcObject = null;
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  stageIdle.hidden = false;
  faceTracks.clear();
  faceCount.value = "0";
  handCount.value = "0";
  poseCount.value = "0";
  startButton.disabled = false;
  stopButton.disabled = true;
  setStatus(message);
  updateScoreboard();
}

// Delete stored game data and clear the session so the old scoreboard cannot reappear.
function deleteSavedData() {
  try {
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(LEGACY_STORAGE_KEY);
    storageEnabled = false;
    stopCamera("Saved data deleted. Camera stopped.");
    faceScores.clear();
    unassignedScores = [];
    nextFaceId = 1;
    faceTracks.clear();
    updateScoreboard();
    setStatus("Saved data deleted. Scores are cleared for this session.");
  } catch {
    setStatus("Could not delete saved data from browser storage.");
  }
}

// Restore saved slider choices when present; defaults remain unsaved on initial load.
for (const cue of CUES) {
  const input = document.querySelector(`#${cue.input}`);
  const savedValue = savedGame.settings[cue.id];
  if (Number.isFinite(savedValue)) input.value = String(Math.max(0, Math.min(10, savedValue)));
  input.addEventListener("input", updatePointControls);
}

// Connect page actions and initialize the scoreboard without rewriting cleared settings.
startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", () => stopCamera());
deleteSavedDataButton.addEventListener("click", deleteSavedData);
window.addEventListener("pagehide", () => stopCamera("Camera closed."));

updatePointControls(false);
updateScoreboard();